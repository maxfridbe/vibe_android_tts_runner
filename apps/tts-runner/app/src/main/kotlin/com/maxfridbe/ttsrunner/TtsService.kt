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
        const val EXTRA_PREVIEW = "preview"  // true: cache the output as the voice's preview clip
        const val EXTRA_DESIGN = "design"    // true: no speaker ref — roll a fresh random voice
        const val EXTRA_SEED = "seed"        // base seed; 0 = default
        const val EXTRA_INSTRUCT = "instruct" // voice description (VoiceDesign model)
        const val EXTRA_JOB_ID = "job_id"    // resume this job: keep its id and its cached chunks
        private const val CHANNEL = "tts"
        private const val NOTIF_ID = 1
        private const val IDLE_TIMEOUT_MS = 5 * 60_000L
        private const val SAMPLE_RATE = 24000
    }

    @Volatile private var stopRequested = false
    @Volatile private var working = false
    @Volatile private var bound = false
    private var workThread: Thread? = null
    private val jobLock = Any()
    private var jobEpoch = 0
    private val binder = android.os.Binder()
    private var wakeLock: PowerManager.WakeLock? = null
    private val idleHandler = android.os.Handler(android.os.Looper.getMainLooper())
    private val idleRunnable = Runnable { if (!working && !bound) stopSelf() }

    // MainActivity binds while it is on screen, which keeps this process (and
    // the loaded model) alive: "model stays loaded as long as the app is open"
    override fun onBind(intent: Intent?): IBinder { bound = true; return binder }
    override fun onUnbind(intent: Intent?): Boolean {
        bound = false
        if (!working) idleHandler.postDelayed(idleRunnable, IDLE_TIMEOUT_MS)
        return false
    }

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel(CHANNEL, "Speech generation", NotificationManager.IMPORTANCE_LOW)
        )
        // This process starting with the in-flight marker still on disk means
        // the previous one died mid-job (native crash or an OS kill — nothing
        // else skips the finally). Only this process can say that with
        // authority: the UI binding to us is itself what restarts us, so it
        // cannot tell a live job from a resurrected process.
        val marker = java.io.File(filesDir, "job-inflight")
        if (marker.exists()) {
            DebugLog.log(this, "TtsService", "engine restarted with backend=${marker.readText()} in flight; job died")
            JobStore.failRunning(this, "interrupted — the engine process was killed mid-job")
            marker.delete()
            sendBroadcast(Intent(STATUS_BROADCAST).setPackage(packageName).putExtra("state", "jobs"))
        }
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
                val preview = intent.getBooleanExtra(EXTRA_PREVIEW, false)
                val design = intent.getBooleanExtra(EXTRA_DESIGN, false)
                val seed = intent.getIntExtra(EXTRA_SEED, 0)
                val instruct = intent.getStringExtra(EXTRA_INSTRUCT) ?: ""
                val resumeId = intent.getLongExtra(EXTRA_JOB_ID, 0L)
                startJob(text, title, voice, backend, save, preview, design, seed, instruct, resumeId)
            }
        }
        return START_NOT_STICKY
    }

    private fun startJob(text: String, title: String, voiceName: String, backend: String, save: Boolean,
                         preview: Boolean = false, design: Boolean = false, seed: Int = 0,
                         instruct: String = "", resumeId: Long = 0L) {
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
                runJob(text, title, voiceName, backend, save, preview, design, seed, instruct, resumeId)
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

    private fun runJob(text: String, title: String, voiceName: String, backendWanted: String, save: Boolean,
                       preview: Boolean, design: Boolean, seed: Int, instructWanted: String,
                       resumeId: Long = 0L) {
        // crash-loop breaker: if a previous job died without cleaning its
        // marker (native SIGABRT kills this whole process), retry on CPU
        val marker = java.io.File(filesDir, "job-inflight")
        // a design job with a description needs the VoiceDesign model; without
        // it we still roll (seed-random voice) on the regular model
        var instruct = if (design) instructWanted else ""
        val model = if (design && instruct.isNotBlank()) {
            val vd = ModelManager.designModel(this)
            if (vd == null) {
                broadcast("note", 0, 0, "VoiceDesign model not installed — rolling a random voice instead")
                instruct = ""
                ModelManager.selectedModel(this)
            } else vd
        } else ModelManager.selectedModel(this)
        if (model == null) {
            fail("No model downloaded — open TTS Runner first"); return
        }
        // Backends: cpu | opencl | vulkan. OpenCL (Adreno) is only worthwhile
        // with Q4_0 (tuned kernels, 1.7x talker speedup; other quants hit
        // generic kernels far slower than CPU) — gated on model.gpuCapable.
        // Vulkan works with any quant but not on Adreno drivers (they fail to
        // compile the shaders; the load error falls back to CPU cleanly).
        val wanted = when (backendWanted) { "gpu", "gpu-force" -> "opencl"; else -> backendWanted }
        var backend = when {
            backendWanted == "gpu-force" -> "opencl"
            wanted == "opencl" && model.gpuCapable -> "opencl"
            wanted == "vulkan" -> "vulkan"
            else -> "cpu"
        }
        if (wanted == "opencl" && !model.gpuCapable && backendWanted != "gpu-force") {
            DebugLog.log(this, "TtsService", "opencl requested but ${model.id} is not gpu-capable; using cpu")
            broadcast("note", 0, 0, "OpenCL GPU needs the Q4_0 model — using CPU")
        }
        // A previous run that vanished mid-generation left this marker behind
        // (process death: native crash or an lmkd kill). It is reported, not
        // acted on — the backend is the user's choice, and a failed job is
        // resumable with whichever backend they pick.
        if (marker.exists()) {
            DebugLog.log(this, "TtsService", "previous job with backend=${marker.readText()} died mid-run")
        }
        marker.writeText(backend)
        DebugLog.log(this, "TtsService", "job start: ${text.length} chars, voice=$voiceName backend=$backend (wanted $backendWanted)")
        // design mode runs with NO speaker reference: the model invents a
        // voice from the seed; the UI can then adopt the output as a voice
        val voice = VoiceStore.list(this).find { it.name == voiceName }
            ?: if (design) null else VoiceStore.defaultVoice(this)
        if (voice == null && !design) {
            fail("No voice imported — add one in TTS Runner"); return
        }
        DebugLog.log(this, "TtsService", "model=${model.id} voice=${voice?.file?.name ?: "(designing, seed=$seed)"}")

        // Register the job before loading the model: the UI switches to the
        // Jobs tab the moment a job is added, and a cold load takes ~20 s —
        // an empty list there would look like nothing happened. A resumed job
        // keeps its id, which is what makes its cached chunks findable.
        val ephemeral = preview || design   // previews/design rolls stay out of job history
        val jobId = if (resumeId != 0L) resumeId else System.currentTimeMillis()
        val chunks = Chunker.split(text)
        DebugLog.log(this, "TtsService", "chunked into ${chunks.size}: ${chunks.map { it.length }}")
        if (!ephemeral) {
            if (resumeId != 0L) JobStore.update(this, jobId) {
                it.status = "running"; it.backend = backend; it.error = ""; it.chunksTotal = chunks.size
            } else JobStore.add(this, JobStore.Job(
                id = jobId, title = title, text = text, voice = voice!!.name,
                model = model.id, backend = backend, save = save, status = "running",
                chunksTotal = chunks.size))
        }
        fun abort(reason: String) {
            persistJobResult(ephemeral, jobId, false, reason, 0, 0.0, 0, "", "")
            fail(reason)
        }
        if (chunks.isEmpty()) { abort("Nothing to speak"); return }

        // Chunk cache: every generated chunk is kept as raw PCM under
        // jobs/<id>/, so a job that dies (or is stopped) resumes from the first
        // missing chunk instead of starting over. Cleared on success.
        val jobDir = if (ephemeral) null else JobStore.jobDir(this, jobId).apply { mkdirs() }
        fun chunkFile(i: Int) = jobDir?.let { java.io.File(it, "chunk_%04d.pcm".format(i)) }
        val cached = (chunks.indices).count { chunkFile(it)?.let { f -> f.exists() && f.length() > 0 } == true }
        if (cached > 0) {
            DebugLog.log(this, "TtsService", "resuming with $cached/${chunks.size} chunks already generated")
            broadcast("note", 0, 0, "Resuming: $cached of ${chunks.size} chunks already generated")
        }

        // The backend is the user's choice: a load failure fails the job with
        // the reason, so they can resume on another backend, rather than being
        // silently moved to CPU.
        if (cached < chunks.size && !ensureLoaded(model, backend)) {
            val why = TtsEngine.nLastError().ifBlank { "unknown error" }
            abort(if (backend == "cpu") "Model load failed: $why"
                  else "$backend failed to start: $why — resume on CPU or pick another engine")
            return
        }
        // ETA is derived from generation speed, so the clock starts after the
        // one-off model load
        jobStartMs = System.currentTimeMillis()

        val queue = LinkedBlockingQueue<ByteArray>(2)
        val player = if (save) null else thread(name = "tts-play") { playLoop(queue) }
        val saver = if (!save) null else try {
            AudioSaver(this, title)
        } catch (e: Exception) {
            DebugLog.log(this, "TtsService", "AudioSaver init failed", e)
            fail("Could not create output file: ${e.message}"); return
        }
        // Samsung jails sustained-CPU processes into the /abnormal cpuset
        // (little cores only, ~10x slower) unless they are actively playing
        // audio — verified on a Fold5 by watching /proc/pid/cpuset flip. Save
        // mode has no playback, so keep a silent AudioTrack running.
        val keepalive = if (save) startSilentKeepalive() else null

        // last-audio capture for the in-app player: raw PCM appended per chunk,
        // wrapped into a WAV when the job ends
        pcmFile().delete()

        var failed: String? = null
        var audioSecs = 0.0
        var genMs = 0L
        var chunksDone = 0
        for ((i, chunk) in chunks.withIndex()) {
            if (stopRequested) break
            update(notif(title, "Chunk ${i + 1}/${chunks.size}", i * 1000 / chunks.size, 1000))
            broadcast("generating", i, chunks.size, chunk.take(80))
            val t0 = System.currentTimeMillis()
            val cf = chunkFile(i)
            // already generated by an earlier attempt: replay it, don't pay for
            // it twice
            var pcm = if (cf != null && cf.exists() && cf.length() > 0)
                runCatching { cf.readBytes() }.getOrNull() else null
            val reused = pcm != null
            if (!reused) {
                val onFrames = { framesDone: Int, framesMax: Int ->
                    reportProgress(title, i, chunks.size, framesDone, framesMax)
                }
                // design rolls skip the plausibility guard: any voice is valid, and
                // re-rolling a design triples an already slow job for nothing
                pcm = if (design) generateOnce(chunk, "", instruct, seed, onFrames)
                      else generatePlausible(chunk, voice?.file?.absolutePath ?: "", instruct, seed, onFrames)
                if (pcm == null) {
                    if (!stopRequested) {
                        val why = TtsEngine.nLastError().ifBlank { "generation failed" }
                        // the engine chose to stay on this backend; say so and let
                        // the job be resumed on another one
                        failed = if (backend == "cpu") why
                                 else "$why (on $backend) — resume on CPU to finish"
                    }
                    DebugLog.log(this, "TtsService", "chunk ${i + 1} FAILED after ${System.currentTimeMillis() - t0} ms: $failed")
                    break
                }
                runCatching { cf?.writeBytes(pcm!!) }
                genMs += System.currentTimeMillis() - t0
            }
            DebugLog.log(this, "TtsService", "chunk ${i + 1}/${chunks.size}: ${chunk.length} chars -> " +
                "${"%.1f".format(pcm!!.size / 2.0 / SAMPLE_RATE)}s audio in " +
                (if (reused) "0 ms (cached)" else "${System.currentTimeMillis() - t0} ms"))
            val bytes = pcm!!
            audioSecs += bytes.size / 2.0 / SAMPLE_RATE
            chunksDone++
            runCatching { pcmFile().appendBytes(bytes) }
            if (saver != null) {
                try {
                    saver.write(bytes)
                } catch (e: Exception) {
                    DebugLog.log(this, "TtsService", "saver.write failed", e)
                    failed = "Save failed: ${e.message}"; break
                }
            } else {
                queue.put(bytes)
            }
        }
        if (player != null) {
            queue.put(EOS)
            player.join()
        }
        keepalive?.interrupt()
        DebugLog.log(this, "TtsService", "job end: stopped=$stopRequested failed=$failed")
        finalizeLastWav()

        // stats summary: "25s audio · 3m 12s · RTF 7.6"
        val stats = if (audioSecs > 0)
            "${"%.0f".format(audioSecs)}s audio · ${fmtDuration(genMs)} · RTF ${"%.1f".format(genMs / 1000.0 / audioSecs)}"
        else ""
        var output = ""
        var outputUri = ""

        when {
            stopRequested -> {
                saver?.abort()
                persistJobResult(ephemeral, jobId, true, failed, chunksDone, audioSecs, genMs, output, outputUri)
                broadcast("stopped", 0, 0, "")
            }
            failed != null -> {
                saver?.abort()
                persistJobResult(ephemeral, jobId, false, failed, chunksDone, audioSecs, genMs, output, outputUri)
                fail(failed)
            }
            saver != null -> {
                val path = try {
                    saver.finish()
                } catch (e: Exception) {
                    DebugLog.log(this, "TtsService", "saver.finish failed", e)
                    fail("Save failed: ${e.message}"); return
                }
                output = path
                outputUri = saver.uri.toString()
                DebugLog.log(this, "TtsService", "saved: $path ($stats)")
                // persist first: the UI rebuilds its Jobs list from this file
                // as soon as the broadcast lands
                persistJobResult(ephemeral, jobId, stopRequested, failed, chunksDone, audioSecs, genMs, output, outputUri)
                jobDir?.deleteRecursively()   // finished: the chunk cache is dead weight
                broadcast("saved", chunks.size, chunks.size, "$path — $stats")
                update(notif(title, "Saved to $path", 0, 0))
            }
            else -> {
                if (preview && voice != null && audioSecs > 0) {
                    runCatching {
                        java.io.File(filesDir, "last_audio.wav")
                            .copyTo(VoiceStore.previewFile(this, voice.name, model.id), overwrite = true)
                    }
                }
                persistJobResult(ephemeral, jobId, stopRequested, failed, chunksDone, audioSecs, genMs, output, outputUri)
                jobDir?.deleteRecursively()
                broadcast(if (design) "designed" else "done", chunks.size, chunks.size, stats)
            }
        }
    }

    private fun persistJobResult(ephemeral: Boolean, jobId: Long, stopped: Boolean, failed: String?,
                                 chunks: Int, audioSecs: Double, genMs: Long, output: String, outputUri: String) {
        if (ephemeral) return
        JobStore.update(this, jobId) {
            it.status = when { stopped -> "stopped"; failed != null -> "failed"; else -> "done" }
            it.chunks = chunks
            it.audioSecs = audioSecs
            it.genMs = genMs
            it.output = output
            it.outputUri = outputUri
            it.error = failed ?: ""
        }
    }

    private fun fmtDuration(ms: Long): String {
        val s = ms / 1000
        return if (s >= 60) "${s / 60}m ${s % 60}s" else "${s}s"
    }

    /** Silent 8 kHz mono playback, ~0% CPU; exists only so the OS classifies
     *  this process as actively playing media during save-mode generation. */
    private fun startSilentKeepalive(): Thread = thread(name = "tts-keepalive") {
        val rate = 8000
        val minBuf = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(rate)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(minBuf, rate / 2))
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()
        val silence = ByteArray(rate / 2)  // 0.25 s of zeros
        try {
            track.play()
            while (!Thread.currentThread().isInterrupted) {
                track.write(silence, 0, silence.size)  // blocks at real-time rate
            }
        } catch (_: InterruptedException) {
        } catch (t: Throwable) {
            DebugLog.log(this, "TtsService", "keepalive stopped: $t")
        } finally {
            runCatching { track.stop() }; track.release()
        }
    }

    private fun pcmFile() = java.io.File(filesDir, "last_audio.pcm")

    /** Wrap the accumulated PCM into files/last_audio.wav for the in-app
     *  player (both processes share filesDir). */
    private fun finalizeLastWav() {
        val pcm = pcmFile()
        val len = pcm.length()
        if (len < 2) { pcm.delete(); return }
        try {
            java.io.File(filesDir, "last_audio.wav").outputStream().use { out ->
                val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
                h.put("RIFF".toByteArray()); h.putInt(36 + len.toInt())
                h.put("WAVE".toByteArray()); h.put("fmt ".toByteArray())
                h.putInt(16); h.putShort(1); h.putShort(1)
                h.putInt(SAMPLE_RATE); h.putInt(SAMPLE_RATE * 2)
                h.putShort(2); h.putShort(16)
                h.put("data".toByteArray()); h.putInt(len.toInt())
                out.write(h.array())
                pcm.inputStream().use { it.copyTo(out) }
            }
            pcm.delete()
        } catch (e: Exception) {
            DebugLog.log(this, "TtsService", "finalizeLastWav failed", e)
        }
    }

    /** Whole-job progress (0–1000‰) + ETA, fed to both the notification and
     *  the UI. In-chunk position uses expected frames (cap / 1.8), clamped,
     *  so the bar moves smoothly instead of stalling at the safety margin. */
    private var lastNotifUpdate = 0L
    @Volatile var jobStartMs = 0L
    private fun reportProgress(title: String, chunkIdx: Int, totalChunks: Int, framesDone: Int, framesMax: Int) {
        val within = minOf(1.0, framesDone / (framesMax / 1.8))
        val permille = (((chunkIdx + within) / totalChunks) * 1000).toInt().coerceIn(0, 1000)
        val now = System.currentTimeMillis()
        // ETA from overall progress rate; meaningless below ~5%
        val etaSecs = if (permille >= 50 && jobStartMs > 0)
            ((now - jobStartMs) * (1000 - permille) / permille / 1000).toInt() else -1
        if (now - lastFramesBroadcast > 500) {
            lastFramesBroadcast = now
            sendBroadcast(Intent(STATUS_BROADCAST).setPackage(packageName)
                .putExtra("state", "frames").putExtra("chunk", framesDone)
                .putExtra("total", framesMax).putExtra("permille", permille)
                .putExtra("eta", etaSecs)
                .putExtra("message", ""))
        }
        if (now - lastNotifUpdate > 1500) {
            lastNotifUpdate = now
            val etaText = if (etaSecs >= 0) " · ~${fmtDuration(etaSecs * 1000L)} left" else ""
            update(notif(title, "Chunk ${chunkIdx + 1}/$totalChunks$etaText", permille, 1000))
        }
    }

    /** Generation with the audiobook maker's plausibility guard: reject
     *  runaway (near max_new_tokens), self-repeating (way over the expected
     *  duration for the text), or instant-EOS outputs and re-roll the seed up
     *  to 2 times. Returns raw PCM (s16 mono 24 kHz) or null. */
    /** Single generation pass, no plausibility re-rolls. */
    private fun generateOnce(chunk: String, voicePath: String, instruct: String, seed: Int, onFrames: (Int, Int) -> Unit): ByteArray? {
        val maxFrames = ((chunk.length / 13.0 * 12.5 * 1.8).toInt() + 32).coerceIn(96, 2048)
        val wav = TtsEngine.nGenerate(chunk, voicePath, "en", instruct, maxFrames, seed, 0.9f, 1.0f,
            object : TtsEngine.ProgressCallback {
                override fun onProgress(framesDone: Int, framesMax: Int) {
                    if (stopRequested) TtsEngine.nCancel()
                    onFrames(framesDone, framesMax)
                }
            }) ?: return null
        return wavToPcm(wav)
    }

    private fun generatePlausible(chunk: String, voicePath: String, instruct: String, baseSeed: Int, onFrames: (Int, Int) -> Unit): ByteArray? {
        // measured post-double-read-fix: ~13 chars/s of speech; cap at 1.8x
        // expected so a runaway generation is bounded tightly (frame cost is
        // ~0.4-2 s on a throttled phone)
        val expectSecs = chunk.length / 13.0
        val maxFrames = ((expectSecs * 12.5 * 1.8).toInt() + 32).coerceIn(96, 2048)
        var best: ByteArray? = null   // shortest non-degenerate take so far
        for (attempt in 0 until 3) {
            if (stopRequested) return null
            val seed = if (attempt == 0) (if (baseSeed != 0) baseSeed else 42) else Random.nextInt(1, 1 shl 30)
            val wav = TtsEngine.nGenerate(
                chunk, voicePath, "en", instruct, maxFrames, seed, 0.9f, 1.0f,  // official qwen-tts talker params
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
                return best
            }
            val pcm = wavToPcm(wav)
            if (pcm == null) {
                DebugLog.log(this, "TtsService", "wavToPcm failed on ${wav.size} bytes (attempt $attempt)")
                return best
            }
            val secs = pcm.size / 2.0 / SAMPLE_RATE
            val tooShort = secs < minOf(3.0, chunk.length / 40.0)
            // 2x expected + 1s catches the model reading the text twice (seen
            // on-device: 12.6s for an 82-char sentence) well before the cap
            val runaway = secs >= 0.95 * (maxFrames / 12.5) || secs > expectSecs * 2.0 + 1.0
            if (!tooShort && !runaway) return pcm
            if (!tooShort && (best == null || pcm.size < best!!.size)) best = pcm
            DebugLog.log(this, "TtsService", "implausible chunk (${"%.1f".format(secs)}s for ${chunk.length} chars, seed $seed), re-rolling")
        }
        // all re-rolls implausible: keep the shortest overlong take (the old
        // code re-generated seed 42 — i.e. reproduced the worst attempt)
        return best
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
            .setSmallIcon(R.drawable.ic_notification)
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
