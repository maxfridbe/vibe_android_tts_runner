package com.techhurts.ttsrunner

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/** Blocking synthesis for callers that are not the UI — today the HTTP host.
 *
 *  The engine lives in another process behind a foreground service that
 *  reports progress by broadcast, so "give me a WAV for this text" means: fire
 *  the intent with an output path and a request id, wait for the terminal
 *  broadcast carrying that id, then read the file.
 *
 *  Requests are queued by the service rather than serialised here: several
 *  callers can be waiting at once, each on its own request id, and they are
 *  served in the order they arrived. There is still one generation at a time —
 *  one model, one context — but that is a queue, not a limit on callers. */
object SynthBridge {

    class Result(val wav: File?, val error: String?)

    fun synth(ctx: Context, text: String, voice: String, timeoutSecs: Long = 900): Result {
        run {
            val id = "http-" + System.nanoTime()
            val out = File(ctx.cacheDir, "http").apply { mkdirs() }.let { File(it, "$id.wav") }
            val latch = CountDownLatch(1)
            var error: String? = null
            val receiver = object : BroadcastReceiver() {
                override fun onReceive(c: Context?, i: Intent?) {
                    if (i?.getStringExtra(TtsService.EXTRA_REQ) != id) return
                    when (i.getStringExtra("state")) {
                        "done", "saved" -> latch.countDown()
                        "error" -> { error = i.getStringExtra("message") ?: "generation failed"; latch.countDown() }
                        "stopped" -> { error = "cancelled by another request"; latch.countDown() }
                    }
                }
            }
            ctx.registerReceiver(receiver, IntentFilter(TtsService.STATUS_BROADCAST),
                Context.RECEIVER_NOT_EXPORTED)
            try {
                ctx.startForegroundService(Intent(ctx, TtsService::class.java)
                    .setAction(TtsService.ACTION_SPEAK)
                    .putExtra(TtsService.EXTRA_TEXT, text)
                    .putExtra(TtsService.EXTRA_TITLE, "API request")
                    .putExtra(TtsService.EXTRA_VOICE, voice)
                    .putExtra(TtsService.EXTRA_BACKEND,
                        Backends.current(ctx, VoiceStore.engineOf(ctx, voice)))
                    // a real job while it runs, so it is visible and resumable,
                    // and removed the moment it succeeds
                    .putExtra(TtsService.EXTRA_QUEUE, true)
                    .putExtra(TtsService.EXTRA_TRANSIENT, true)
                    .putExtra(TtsService.EXTRA_SILENT, true)
                    .putExtra(TtsService.EXTRA_OUT, out.absolutePath)
                    .putExtra(TtsService.EXTRA_REQ, id)
                    .putExtra(TtsService.EXTRA_SAVE, false))
                if (!latch.await(timeoutSecs, TimeUnit.SECONDS)) {
                    return Result(null, "timed out after ${timeoutSecs}s")
                }
            } catch (t: Throwable) {
                return Result(null, t.message ?: t.javaClass.simpleName)
            } finally {
                runCatching { ctx.unregisterReceiver(receiver) }
            }
            if (error != null) return Result(null, error)
            return if (out.exists() && out.length() > 44) Result(out, null)
            else Result(null, "the engine produced no audio")
        }
    }
}
