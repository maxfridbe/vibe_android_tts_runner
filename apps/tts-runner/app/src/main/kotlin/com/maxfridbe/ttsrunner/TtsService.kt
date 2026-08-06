package com.maxfridbe.ttsrunner

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.LinkedBlockingQueue
import kotlin.concurrent.thread
import kotlin.random.Random

/** Foreground TTS service. Runs in the separate ":engine" process so a native
 *  crash in llama.cpp kills only this process, never the UI. Generates chunk
 *  WAVs ahead of playback and streams them through one AudioTrack.
 *
 *  The model stays loaded between requests and is unloaded (process exits)
 *  after IDLE_TIMEOUT_MS with no work — same memory philosophy as the
 *  audiobook maker's GPU workers. */
class TtsService : Service() {

    companion object {
        const val ACTION_SPEAK = "com.maxfridbe.ttsrunner.SPEAK"
        const val ACTION_STOP = "com.maxfridbe.ttsrunner.STOP"
        const val STATUS_BROADCAST = "com.maxfridbe.ttsrunner.STATUS"
        const val EXTRA_TEXT = "text"
        const val EXTRA_TITLE = "title"
        const val EXTRA_VOICE = "voice"
        const val EXTRA_BACKEND = "backend"
        const val EXTRA_SAVE = "save"   // true: render to Music/TTS Runner/*.m4a instead of playing
        private const val CHANNEL = "tts"
        private const val NOTIF_ID = 1
        private const val IDLE_TIMEOUT_MS = 5 * 60_000L
        private const val SAMPLE_RATE = 24000
    }

    @Volatile private var stopRequested = false
    @Volatile private var working = false
    private var workThread: Thread? = null
    private val jobLock = Any()
    private var jobEpoch = 0
    private var wakeLock: PowerManager.WakeLock? = null
    private val idleHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val idleRunnable = Runnable { if (!working) stopSelf() }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, "Speech generation", NotificationManager.IMPORTANCE_LOW)
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopRequested = true
                TtsEngine.nCancel()
                broadcast("stopped", 0, 0, "")
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
            ACTION_SPEAK -> {
                val text = intent.getStringExtra(EXTRA_TEXT) ?: return START_NOT_STICKY
                val title = intent.getStringExtra(EXTRA_TITLE) ?: "Text to speech"
                val voice = intent.getStringExtra(EXTRA_VOICE) ?: ""
                val backend = intent.getStringExtra(EXTRA_BACKEND) ?: "cpu"
                val save = intent.getBooleanExtra(EXTRA_SAVE, false)
                startJob(text, title, voice, backend, save)
            }
        }
        return START_NOT_STICKY
    }

    private fun startJob(text: String, title: String, voiceName: String, backend: String, save: Boolean) {
        // one job at a time; a new share replaces the current playback. The
        // takeover is asynchronous: the previous thread is signalled here and
        // the NEW thread waits for it without a deadline (a 3s join on this
        // thread once expired mid-generation, leaving two threads on one
        // llama context -> "prompt processing failed").
        stopRequested = true
        TtsEngine.nCancel()
        val previous = workThread
        val epoch: Int
        synchronized(jobLock) {
            epoch = ++jobEpoch
            idleHandler.removeCallbacks(idleRunnable)
            startInForeground(notif(title, "Preparing…", 0, 0))
            wakeLock?.release()
            wakeLock = getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ttsrunner:generate")
                .apply { acquire(60 * 60_000L) }
            working = true
        }
        workThread = thread(name = "tts-generate") {
            try {
                previous?.join()
                stopRequested = false
                TtsEngine.nResetCancel()
                runJob(text, title, voiceName, backend, save)
            } catch (t: Throwable) {
                DebugLog.log(this, "TtsService", "runJob crashed", t)
                fail("Internal error: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                // runs on every survivable exit; only a process death (native
                // crash, lmkd kill) leaves the crash marker behind. Skip the
                // service-level teardown if a newer job has already taken over
                // (its runJob only starts after our join() returns, but this
                // finally may run after its startJob promoted the service).
                java.io.File(filesDir, "job-inflight").delete()
                synchronized(jobLock) {
                    if (jobEpoch == epoch) {
                        working = false
                        wakeLock?.release(); wakeLock = null
                        idleHandler.postDelayed(idleRunnable, IDLE_TIMEOUT_MS)
                        stopForeground(STOP_FOREGROUND_DETACH)
                    }
                }
            }
        }
    }

    /** Non-null end-of-stream marker: LinkedBlockingQueue rejects null. */
    private val EOS = ByteArray(0)

    private fun runJob(text: String, title: String, voiceName: String, backendWanted: String, save: Boolean) {
        // crash-loop breaker: if a previous job died without cleaning its
        // marker (native SIGABRT kills this whole process), retry on CPU
        val marker = java.io.File(filesDir, "job-inflight")
        var backend = backendWanted
        if (marker.exists() && backendWanted != "cpu") {
            backend = "cpu"
            DebugLog.log(this, "TtsService", "previous job with backend=${marker.readText()} died mid-run; falling back to cpu")
            broadcast("note", 0, 0, "Previous run died mid-generation — using CPU for this run")
        }
        marker.writeText(backend)
        DebugLog.log(this, "TtsService", "job start: ${text.length} chars, voice=$voiceName backend=$backend (wanted $backendWanted)")
        val model = ModelManager.selectedModel(this)
        if (model == null) {
            fail("No model downloaded — open TTS Runner first"); return
        }
        val voice = VoiceStore.list(this).find { it.name == voiceName } ?: VoiceStore.defaultVoice(this)
        if (voice == null) {
            fail("No voice imported — add one in TTS Runner"); return
        }
        DebugLog.log(this, "TtsService", "model=${model.id} voice=${voice.file.name} (${voice.file.length()} b)")

        if (!ensureLoaded(model, backend)) {
            if (backend == "cpu") {
                fail("Model load failed: ${TtsEngine.nLastError()}"); return
            }
            DebugLog.log(this, "TtsService", "GPU load failed (${TtsEngine.nLastError()}); falling back to cpu")
            broadcast("note", 0, 0, "GPU init failed — using CPU")
            backend = "cpu"
            marker.writeText(backend)
            if (!ensureLoaded(model, backend)) {
                fail("Model load failed: ${TtsEngine.nLastError()}"); return
            }
        }

        val chunks = Chunker.split(text)
        DebugLog.log(this, "TtsService", "chunked into ${chunks.size}: ${chunks.map { it.length }}")
        if (chunks.isEmpty()) { fail("Nothing to speak"); return }

        val queue = LinkedBlockingQueue<ByteArray>(2)
        val player = if (save) null else thread(name = "tts-play") { playLoop(queue) }
        val saver = if (!save) null else try {
            AudioSaver(this, title)
        } catch (e: Exception) {
            DebugLog.log(this, "TtsService", "AudioSaver init failed", e)
            fail("Could not create output file: ${e.message}"); return
        }

        var failed: String? = null
        for ((i, chunk) in chunks.withIndex()) {
            if (stopRequested) break
            update(notif(title, "Chunk ${i + 1}/${chunks.size}", i * 1000 / chunks.size, 1000))
            broadcast("generating", i, chunks.size, chunk.take(80))
            val t0 = System.currentTimeMillis()
            val onFrames = { framesDone: Int, framesMax: Int ->
                reportProgress(title, i, chunks.size, framesDone, framesMax)
            }
            var pcm = generatePlausible(chunk, voice.file.absolutePath, onFrames)
            if (pcm == null && !stopRequested && backend != "cpu") {
                // a caught native exception drops the engine; retry this chunk
                // on CPU in-process instead of dying and using the marker path
                DebugLog.log(this, "TtsService", "GPU generation failed (${TtsEngine.nLastError()}); retrying chunk on CPU")
                broadcast("note", 0, 0, "GPU generation failed — switching to CPU")
                backend = "cpu"
                marker.writeText(backend)
                loadedKey = null
                if (ensureLoaded(model, backend)) {
                    pcm = generatePlausible(chunk, voice.file.absolutePath, onFrames)
                }
            }
            if (pcm == null) {
                if (!stopRequested) failed = TtsEngine.nLastError().ifBlank { "generation failed" }
                DebugLog.log(this, "TtsService", "chunk ${i + 1} FAILED after ${System.currentTimeMillis() - t0} ms: $failed")
                break
            }
            DebugLog.log(this, "TtsService", "chunk ${i + 1}/${chunks.size}: ${chunk.length} chars -> " +
                "${"%.1f".format(pcm.size / 2.0 / SAMPLE_RATE)}s audio in ${System.currentTimeMillis() - t0} ms")
            if (saver != null) {
                try {
                    saver.write(pcm)
                } catch (e: Exception) {
                    DebugLog.log(this, "TtsService", "saver.write failed", e)
                    failed = "Save failed: ${e.message}"; break
                }
            } else {
                queue.put(pcm)
            }
        }
        if (player != null) {
            queue.put(EOS)
            player.join()
        }
        DebugLog.log(this, "TtsService", "job end: stopped=$stopRequested failed=$failed")

        when {
            stopRequested -> { saver?.abort(); broadcast("stopped", 0, 0, "") }
            failed != null -> { saver?.abort(); fail(failed) }
            saver != null -> {
                val path = try {
                    saver.finish()
                } catch (e: Exception) {
                    DebugLog.log(this, "TtsService", "saver.finish failed", e)
                    fail("Save failed: ${e.message}"); return
                }
                DebugLog.log(this, "TtsService", "saved: $path")
                broadcast("saved", chunks.size, chunks.size, path)
                update(notif(title, "Saved to $path", 0, 0))
            }
            else -> broadcast("done", chunks.size, chunks.size, "")
        }
    }

    /** Whole-job progress (0–1000‰), fed to both the notification and the UI.
     *  In-chunk position uses expected frames (cap / 2.2), clamped, so the bar
     *  moves smoothly instead of stalling at the safety margin. */
    private var lastNotifUpdate = 0L
    private fun reportProgress(title: String, chunkIdx: Int, totalChunks: Int, framesDone: Int, framesMax: Int) {
        val within = minOf(1.0, framesDone / (framesMax / 2.2))
        val permille = (((chunkIdx + within) / totalChunks) * 1000).toInt().coerceIn(0, 1000)
        val now = System.currentTimeMillis()
        if (now - lastFramesBroadcast > 500) {
            lastFramesBroadcast = now
            sendBroadcast(Intent(STATUS_BROADCAST).setPackage(packageName)
                .putExtra("state", "frames").putExtra("chunk", framesDone)
                .putExtra("total", framesMax).putExtra("permille", permille)
                .putExtra("message", ""))
        }
        if (now - lastNotifUpdate > 1500) {
            lastNotifUpdate = now
            update(notif(title, "Chunk ${chunkIdx + 1}/$totalChunks", permille, 1000))
        }
    }

    /** Generation with the audiobook maker's plausibility guard: reject
     *  runaway (near max_new_tokens), self-repeating (way over the expected
     *  duration for the text), or instant-EOS outputs and re-roll the seed up
     *  to 2 times. Returns raw PCM (s16 mono 24 kHz) or null. */
    private fun generatePlausible(chunk: String, voicePath: String, onFrames: (Int, Int) -> Unit): ByteArray? {
        val expectSecs = chunk.length / 16.0
        val maxFrames = ((expectSecs * 12.5 * 2.2).toInt() + 64).coerceIn(128, 2048)
        for (attempt in 0 until 3) {
            if (stopRequested) return null
            val seed = if (attempt == 0) 42 else Random.nextInt(1, 1 shl 30)
            val wav = TtsEngine.nGenerate(
                chunk, voicePath, "en", maxFrames, seed, 0.8f, 0.95f,
                object : TtsEngine.ProgressCallback {
                    override fun onProgress(framesDone: Int, framesMax: Int) {
                        // re-assert cancel: a nCancel that raced a generate-call
                        // boundary is otherwise absorbed by the next chunk
                        if (stopRequested) TtsEngine.nCancel()
                        onFrames(framesDone, framesMax)
                    }
                })
            if (wav == null) {
                DebugLog.log(this, "TtsService", "nGenerate null (attempt $attempt): ${TtsEngine.nLastError()}")
                return null
            }
            val pcm = wavToPcm(wav)
            if (pcm == null) {
                DebugLog.log(this, "TtsService", "wavToPcm failed on ${wav.size} bytes (attempt $attempt)")
                return null
            }
            val secs = pcm.size / 2.0 / SAMPLE_RATE
            val tooShort = secs < minOf(3.0, chunk.length / 40.0)
            // 2x expected + 1s catches the model reading the text twice (seen
            // on-device: 12.6s for an 82-char sentence) well before the cap
            val runaway = secs >= 0.95 * (maxFrames / 12.5) || secs > expectSecs * 2.0 + 1.0
            if (!tooShort && !runaway) return pcm
            DebugLog.log(this, "TtsService", "implausible chunk (${"%.1f".format(secs)}s for ${chunk.length} chars, seed $seed), re-rolling")
        }
        // all re-rolls implausible: use the last attempt rather than failing hard
        val wav = TtsEngine.nGenerate(chunk, voicePath, "en", maxFrames, 42, 0.8f, 0.95f, null) ?: return null
        return wavToPcm(wav)
    }

    private var loadedKey: String? = null
    private fun ensureLoaded(model: ModelManager.CatalogModel, backend: String): Boolean {
        val key = "${model.id}|$backend"
        if (loadedKey == key) return true
        TtsEngine.nUnload()
        broadcast("loading", 0, 0, model.label)
        DebugLog.log(this, "TtsService", "devices:\n${runCatching { TtsEngine.nDeviceInfo() }.getOrElse { "$it" }}")
        val threads = bigCoreCount()
        val t0 = System.currentTimeMillis()
        val ok = TtsEngine.nLoad(
            ModelManager.talkerPath(this, model), ModelManager.mmprojPath(this, model), backend, threads)
        DebugLog.log(this, "TtsService", "nLoad(${model.id}, $backend, $threads threads) -> $ok " +
            "in ${System.currentTimeMillis() - t0} ms" + if (!ok) " err=${TtsEngine.nLastError()}" else "")
        loadedKey = if (ok) key else null
        return ok
    }

    /** Threads = number of performance cores (max_freq >= 80% of the fastest
     *  core). ggml splits work statically, so one little core in the pool
     *  drags every layer down to its speed (6 threads on SD 8 Gen 2 ran
     *  slower than its 5 big cores would). */
    private fun bigCoreCount(): Int {
        val freqs = (0 until Runtime.getRuntime().availableProcessors()).mapNotNull { c ->
            runCatching {
                java.io.File("/sys/devices/system/cpu/cpu$c/cpufreq/cpuinfo_max_freq").readText().trim().toLong()
            }.getOrNull()
        }
        if (freqs.isEmpty()) return (Runtime.getRuntime().availableProcessors() - 1).coerceIn(2, 6)
        val top = freqs.max()
        val n = freqs.count { it * 10 >= top * 8 }.coerceIn(2, 6)
        DebugLog.log(this, "TtsService", "cpu max freqs $freqs -> $n threads")
        return n
    }

    private fun playLoop(queue: LinkedBlockingQueue<ByteArray>) {
        val minBuf = AudioTrack.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(SAMPLE_RATE)
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(minBuf, SAMPLE_RATE)) // >= 0.5 s
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        DebugLog.log(this, "TtsService", "AudioTrack created: state=${track.state} session=${track.audioSessionId} minBuf=$minBuf")
        track.play()
        try {
            var played = 0L
            while (true) {
                val pcm = queue.take()
                if (pcm.isEmpty()) break // EOS
                var off = 0
                while (off < pcm.size && !stopRequested) {
                    val n = track.write(pcm, off, minOf(32768, pcm.size - off))
                    if (n <= 0) {
                        DebugLog.log(this, "TtsService", "AudioTrack.write returned $n (playState=${track.playState})")
                        break
                    }
                    off += n
                }
                played += off
                if (stopRequested) break
            }
            if (!stopRequested) {
                track.stop() // drains remaining buffered audio
            }
            DebugLog.log(this, "TtsService", "playback done: ${played / 2 / SAMPLE_RATE}s written, underruns=${track.underrunCount}")
        } catch (t: Throwable) {
            DebugLog.log(this, "TtsService", "playLoop crashed", t)
        } finally {
            if (stopRequested) { try { track.pause(); track.flush() } catch (_: Exception) {} }
            track.release()
        }
    }

    /** Extract s16le PCM from a RIFF WAV produced by the engine. */
    private fun wavToPcm(wav: ByteArray): ByteArray? {
        if (wav.size < 44) return null
        val buf = ByteBuffer.wrap(wav).order(ByteOrder.LITTLE_ENDIAN)
        if (wav.copyOfRange(0, 4).decodeToString() != "RIFF") return null
        var pos = 12
        while (pos + 8 <= wav.size) {
            val id = wav.copyOfRange(pos, pos + 4).decodeToString()
            val len = buf.getInt(pos + 4)
            if (id == "data") {
                val end = minOf(pos + 8 + len, wav.size)
                return wav.copyOfRange(pos + 8, end)
            }
            pos += 8 + len + (len and 1)
        }
        return null
    }

    private fun fail(msg: String) {
        broadcast("error", 0, 0, msg)
        update(notif("TTS Runner", msg, 0, 0))
    }

    private var lastFramesBroadcast = 0L

    private fun broadcast(state: String, chunk: Int, total: Int, message: String) {
        sendBroadcast(Intent(STATUS_BROADCAST).setPackage(packageName)
            .putExtra("state", state).putExtra("chunk", chunk)
            .putExtra("total", total).putExtra("message", message))
    }

    private fun startInForeground(n: Notification) {
        if (Build.VERSION.SDK_INT >= 29) {
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK)
        } else {
            startForeground(NOTIF_ID, n)
        }
    }

    private fun update(n: Notification) =
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, n)

    private fun notif(title: String, text: String, progress: Int, max: Int): Notification {
        val stopIntent = PendingIntent.getService(
            this, 1, Intent(this, TtsService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        val b = Notification.Builder(this, CHANNEL)
            .setSmallIcon(android.R.drawable.ic_lock_silent_mode_off)
            .setContentTitle(title)
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .addAction(Notification.Action.Builder(null, "Stop", stopIntent).build())
        if (max > 0) b.setProgress(max, progress, false)
        return b.build()
    }

    override fun onDestroy() {
        stopRequested = true
        TtsEngine.nCancel()
        TtsEngine.nUnload()
        super.onDestroy()
    }
}
