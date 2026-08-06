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
        private const val CHANNEL = "tts"
        private const val NOTIF_ID = 1
        private const val IDLE_TIMEOUT_MS = 5 * 60_000L
        private const val SAMPLE_RATE = 24000
    }

    @Volatile private var stopRequested = false
    @Volatile private var working = false
    private var workThread: Thread? = null
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
                startJob(text, title, voice, backend)
            }
        }
        return START_NOT_STICKY
    }

    private fun startJob(text: String, title: String, voiceName: String, backend: String) {
        // one job at a time; a new share replaces the current playback
        stopRequested = true
        TtsEngine.nCancel()
        workThread?.join(3000)
        stopRequested = false
        idleHandler.removeCallbacks(idleRunnable)

        startInForeground(notif(title, "Preparing…", 0, 0))
        wakeLock?.release()
        wakeLock = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ttsrunner:generate")
            .apply { acquire(60 * 60_000L) }

        working = true
        workThread = thread(name = "tts-generate") {
            try {
                runJob(text, title, voiceName, backend)
            } catch (t: Throwable) {
                DebugLog.log(this, "TtsService", "runJob crashed", t)
                fail("Internal error: ${t.javaClass.simpleName}: ${t.message}")
            } finally {
                // runs on every survivable exit; only a process death (native
                // crash, lmkd kill) leaves the crash marker behind
                java.io.File(filesDir, "job-inflight").delete()
                working = false
                wakeLock?.release(); wakeLock = null
                idleHandler.postDelayed(idleRunnable, IDLE_TIMEOUT_MS)
                stopForeground(STOP_FOREGROUND_DETACH)
            }
        }
    }

    /** Non-null end-of-stream marker: LinkedBlockingQueue rejects null. */
    private val EOS = ByteArray(0)

    private fun runJob(text: String, title: String, voiceName: String, backendWanted: String) {
        // crash-loop breaker: if a previous job died without cleaning its
        // marker (native SIGABRT kills this whole process), retry on CPU
        val marker = java.io.File(filesDir, "job-inflight")
        var backend = backendWanted
        if (marker.exists() && backendWanted != "cpu") {
            backend = "cpu"
            DebugLog.log(this, "TtsService", "previous job with backend=${marker.readText()} died mid-run; falling back to cpu")
            broadcast("note", 0, 0, "GPU engine crashed last time — using CPU for this run")
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
        val player = thread(name = "tts-play") { playLoop(queue) }

        var failed: String? = null
        for ((i, chunk) in chunks.withIndex()) {
            if (stopRequested) break
            update(notif(title, "Chunk ${i + 1}/${chunks.size}", i, chunks.size))
            broadcast("generating", i, chunks.size, chunk.take(80))
            val t0 = System.currentTimeMillis()
            var pcm = generatePlausible(chunk, voice.file.absolutePath)
            if (pcm == null && !stopRequested && backend != "cpu") {
                // a caught native exception drops the engine; retry this chunk
                // on CPU in-process instead of dying and using the marker path
                DebugLog.log(this, "TtsService", "GPU generation failed (${TtsEngine.nLastError()}); retrying chunk on CPU")
                broadcast("note", 0, 0, "GPU generation failed — switching to CPU")
                backend = "cpu"
                marker.writeText(backend)
                loadedKey = null
                if (ensureLoaded(model, backend)) {
                    pcm = generatePlausible(chunk, voice.file.absolutePath)
                }
            }
            if (pcm == null) {
                if (!stopRequested) failed = TtsEngine.nLastError().ifBlank { "generation failed" }
                DebugLog.log(this, "TtsService", "chunk ${i + 1} FAILED after ${System.currentTimeMillis() - t0} ms: $failed")
                break
            }
            DebugLog.log(this, "TtsService", "chunk ${i + 1}/${chunks.size}: ${chunk.length} chars -> " +
                "${"%.1f".format(pcm.size / 2.0 / SAMPLE_RATE)}s audio in ${System.currentTimeMillis() - t0} ms")
            queue.put(pcm)
        }
        queue.put(EOS)
        player.join()
        DebugLog.log(this, "TtsService", "job end: stopped=$stopRequested failed=$failed")

        when {
            stopRequested -> broadcast("stopped", 0, 0, "")
            failed != null -> fail(failed)
            else -> broadcast("done", chunks.size, chunks.size, "")
        }
    }

    /** Generation with the audiobook maker's plausibility guard: reject
     *  runaway (near max_new_tokens) or instant-EOS outputs and re-roll the
     *  seed up to 2 times. Returns raw PCM (s16 mono 24 kHz) or null. */
    private fun generatePlausible(chunk: String, voicePath: String): ByteArray? {
        val expectSecs = chunk.length / 16.0
        val maxFrames = ((expectSecs * 12.5 * 2.2).toInt() + 64).coerceIn(128, 2048)
        for (attempt in 0 until 3) {
            if (stopRequested) return null
            val seed = if (attempt == 0) 42 else Random.nextInt(1, 1 shl 30)
            val wav = TtsEngine.nGenerate(
                chunk, voicePath, "en", maxFrames, seed, 0.8f, 0.95f,
                object : TtsEngine.ProgressCallback {
                    override fun onProgress(framesDone: Int, framesMax: Int) {
                        broadcastThrottled("frames", framesDone, framesMax)
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
            val runaway = secs >= 0.95 * (maxFrames / 12.5)
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
        val threads = (Runtime.getRuntime().availableProcessors() - 1).coerceIn(2, 6)
        val t0 = System.currentTimeMillis()
        val ok = TtsEngine.nLoad(
            ModelManager.talkerPath(this, model), ModelManager.mmprojPath(this, model), backend, threads)
        DebugLog.log(this, "TtsService", "nLoad(${model.id}, $backend, $threads threads) -> $ok " +
            "in ${System.currentTimeMillis() - t0} ms" + if (!ok) " err=${TtsEngine.nLastError()}" else "")
        loadedKey = if (ok) key else null
        return ok
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
    private fun broadcastThrottled(state: String, a: Int, b: Int) {
        val now = System.currentTimeMillis()
        if (now - lastFramesBroadcast > 500) {
            lastFramesBroadcast = now
            broadcast(state, a, b, "")
        }
    }

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
