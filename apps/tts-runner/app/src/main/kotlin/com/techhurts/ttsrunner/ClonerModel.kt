package com.techhurts.ttsrunner

import android.content.Context
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/** Download for the on-device cloning models.
 *
 *  ~185 MB of ONNX, so it does not belong in everyone's APK. It lives in the
 *  repository instead and lands in filesDir/cloner on request — the same
 *  folder a side-loaded copy uses.
 *
 *  Two analyzer variants ship together: Qwen3-TTS's own speaker encoder (the
 *  one listening preferred) and the classic ECAPA pair. */
object ClonerModel {

    /** Raw file URLs, so this needs no API token and no release plumbing. */
    private const val BASE =
        "https://raw.githubusercontent.com/maxfridbe/vibe_android_tts_runner/main/models/cloner/"

    private val FILES = listOf(
        VoiceCloner.QSPK_ASSET to 48_703_976L,
        VoiceCloner.QSTYLE_ASSET to 24_407_532L,  // fh5_d10: qwen-RELABELED 514-pair bank (flywheel), centered-qwen 0.588 held-out; versioned name — retrains keep the same size
        RefineEngine.QCENTER_ASSET to 8_192L,     // qwen population center (2048 x f32): flips the refine to centered-qwen scoring
        VoiceCloner.SPK_ASSET to 84_084_349L,
        VoiceCloner.STYLE_ASSET to 8_404_363L,
        RefineEngine.BASIS_ASSET to 19_910_684L,  // k=384 basis v5 (qwen-relabeled refit) — versioned name for the same reason
    )

    val totalBytes: Long get() = FILES.sumOf { it.second }

    /** Exact sizes, so a retrained model of the same name re-downloads. */
    fun installed(ctx: Context) = FILES.all { (n, s) -> File(VoiceCloner.dir(ctx), n).length() == s }

    fun remove(ctx: Context) {
        FILES.forEach { (n, _) -> File(VoiceCloner.dir(ctx), n).delete() }
        RefineEngine.OLD_BASIS_ASSETS.forEach { File(VoiceCloner.dir(ctx), it).delete() }
        File(VoiceCloner.dir(ctx), VoiceCloner.LEGACY_QSTYLE_ASSET).delete()
    }

    @Volatile private var cancel = false

    fun cancel() { cancel = true }

    /** Blocking; call off the main thread. [progress] is (bytes, total). */
    fun download(ctx: Context, progress: (Long, Long) -> Unit): String? {
        cancel = false
        val dir = VoiceCloner.dir(ctx)
        var done = 0L
        for ((name, size) in FILES) {
            val dest = File(dir, name)
            if (dest.length() == size) { done += size; progress(done, totalBytes); continue }
            dest.delete()   // stale build of the same name
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
