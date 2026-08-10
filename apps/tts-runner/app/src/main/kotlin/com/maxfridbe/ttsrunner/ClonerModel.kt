package com.maxfridbe.ttsrunner

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** Download for the on-device cloning encoder.
 *
 *  It is 89 MB of ONNX for a feature that currently scores 0.22 held-out
 *  speaker similarity against 0.82 for the desktop cloner, so it does not
 *  belong in everyone's APK. It lives in the repository instead and lands in
 *  filesDir/cloner on request — the same folder a side-loaded copy uses.
 *
 *  The numbers are quoted in the UI on purpose. Someone downloading 89 MB
 *  deserves to know what it will sound like before they wait for it. */
object ClonerModel {

    /** Raw file URLs, so this needs no API token and no release plumbing. */
    private const val BASE =
        "https://raw.githubusercontent.com/maxfridbe/vibe_android_tts_runner/main/models/cloner/"

    private val FILES = listOf(
        VoiceCloner.SPK_ASSET to 84_084_349L,
        VoiceCloner.STYLE_ASSET to 4_959_295L,
    )

    val totalBytes: Long get() = FILES.sumOf { it.second }

    fun installed(ctx: Context) = FILES.all { (n, _) -> File(VoiceCloner.dir(ctx), n).length() > 0 }

    fun remove(ctx: Context) = FILES.forEach { (n, _) -> File(VoiceCloner.dir(ctx), n).delete() }

    @Volatile private var cancel = false

    fun cancel() { cancel = true }

    /** Blocking; call off the main thread. [progress] is (bytes, total). */
    fun download(ctx: Context, progress: (Long, Long) -> Unit): String? {
        cancel = false
        val dir = VoiceCloner.dir(ctx)
        var done = 0L
        for ((name, size) in FILES) {
            val dest = File(dir, name)
            if (dest.length() > 0) { done += size; progress(done, totalBytes); continue }
            val part = File(dir, "$name.part")
            try {
                val conn = URL(BASE + name).openConnection() as HttpURLConnection
                conn.connectTimeout = 20000
                conn.readTimeout = 120000
                conn.instanceFollowRedirects = true
                conn.connect()
                if (conn.responseCode !in 200..299) {
                    return "server said ${conn.responseCode} for $name"
                }
                part.outputStream().use { out ->
                    conn.inputStream.use { input ->
                        val buf = ByteArray(128 * 1024)
                        while (true) {
                            if (cancel) return "canceled"
                            val n = input.read(buf)
                            if (n < 0) break
                            out.write(buf, 0, n)
                            done += n
                            progress(done, totalBytes)
                        }
                    }
                }
                conn.disconnect()
                if (!part.renameTo(dest)) return "could not finish $name"
            } catch (e: Exception) {
                part.delete()
                DebugLog.log(ctx, "ClonerModel", "download $name failed", e)
                return e.message ?: e.javaClass.simpleName
            }
        }
        return null
    }
}
