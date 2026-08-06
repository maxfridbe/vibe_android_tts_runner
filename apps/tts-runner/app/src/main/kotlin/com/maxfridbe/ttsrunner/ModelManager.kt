package com.maxfridbe.ttsrunner

import android.content.Context
import java.io.File
import java.io.RandomAccessFile
import java.net.HttpURLConnection
import java.net.URL

/** Model catalog + resumable downloader (same pattern as android_builder's
 *  bonsai app: .part file, HTTP Range resume, atomic rename). A model is a
 *  pair of GGUFs: the talker and the codec mmproj. GGUFs dropped manually
 *  into the models dir are picked up too. */
object ModelManager {

    data class CatalogModel(
        val id: String,
        val label: String,
        val talkerUrl: String,
        val talkerFile: String,
        val mmprojUrl: String,
        val mmprojFile: String,
        val totalBytes: Long,
        /** Nobody hosts this quant: derive it locally from another catalog
         *  entry's talker via llama.cpp's quantizer (talkerUrl unused). */
        val quantizeFrom: String? = null,
        val quantizeType: String? = null,
        val gpuCapable: Boolean = false,
    )

    val CATALOG = listOf(
        CatalogModel(
            id = "1.7b-q4",
            label = "Qwen3-TTS 1.7B Q4_K_M (1.5 GB, recommended)",
            talkerUrl = "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
            talkerFile = "Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf",
            mmprojUrl = "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            mmprojFile = "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            totalBytes = 1_035_965_280L + 446_422_912L,
        ),
        CatalogModel(
            id = "1.7b-q8",
            label = "Qwen3-TTS 1.7B Q8_0 (2.3 GB, higher quality)",
            talkerUrl = "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            talkerFile = "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            mmprojUrl = "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            mmprojFile = "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            totalBytes = 1_847_874_400L + 446_422_912L,
        ),
        CatalogModel(
            id = "1.7b-q4_0-gpu",
            label = "Qwen3-TTS 1.7B Q4_0 (GPU-capable, converted from Q8 on device)",
            talkerUrl = "",  // derived locally, see quantizeFrom
            talkerFile = "Qwen3-TTS-12Hz-1.7B-Base-Q4_0.gguf",
            mmprojUrl = "https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            mmprojFile = "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            totalBytes = 982_979_424L + 446_422_912L,
            quantizeFrom = "Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf",
            quantizeType = "Q4_0",
            gpuCapable = true,   // Adreno OpenCL kernels are tuned for Q4_0
        ),
    )

    fun modelsDir(ctx: Context): File {
        val ext = ctx.getExternalFilesDir("models")
        return (ext ?: File(ctx.filesDir, "models")).apply { mkdirs() }
    }

    fun isDownloaded(ctx: Context, m: CatalogModel): Boolean =
        File(modelsDir(ctx), m.talkerFile).exists() && File(modelsDir(ctx), m.mmprojFile).exists()

    fun selectedModel(ctx: Context): CatalogModel? {
        val prefs = ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
        val id = prefs.getString("model_id", null)
        val m = CATALOG.find { it.id == id } ?: CATALOG.firstOrNull { isDownloaded(ctx, it) }
        return m?.takeIf { isDownloaded(ctx, it) }
    }

    fun selectModel(ctx: Context, id: String) {
        ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE).edit().putString("model_id", id).apply()
    }

    fun talkerPath(ctx: Context, m: CatalogModel): String = File(modelsDir(ctx), m.talkerFile).absolutePath
    fun mmprojPath(ctx: Context, m: CatalogModel): String = File(modelsDir(ctx), m.mmprojFile).absolutePath

    interface DownloadListener {
        fun onProgress(file: String, doneBytes: Long, totalBytes: Long)
        fun onDone()
        fun onError(message: String)
    }

    @Volatile private var downloadCanceled = false
    fun cancelDownload() { downloadCanceled = true }

    /** Blocking; call on a worker thread. Downloads talker then mmproj,
     *  skipping files already complete, resuming partials. */
    fun download(ctx: Context, m: CatalogModel, listener: DownloadListener) {
        downloadCanceled = false
        try {
            if (m.quantizeFrom != null) {
                val src = File(modelsDir(ctx), m.quantizeFrom)
                if (!src.exists()) {
                    // fetch the source quant first (find its catalog entry for the url)
                    val srcModel = CATALOG.first { it.talkerFile == m.quantizeFrom }
                    downloadOne(ctx, srcModel.talkerUrl, srcModel.talkerFile, listener)
                }
                val dest = File(modelsDir(ctx), m.talkerFile)
                if (!dest.exists()) {
                    listener.onProgress("converting to ${m.quantizeType} (a few minutes)…", 0, 0)
                    val tmp = File(modelsDir(ctx), "${m.talkerFile}.part")
                    tmp.delete()
                    if (!TtsEngine.nQuantize(src.absolutePath, tmp.absolutePath, m.quantizeType!!)) {
                        tmp.delete()
                        throw java.io.IOException("quantize failed: ${TtsEngine.nLastError()}")
                    }
                    if (!tmp.renameTo(dest)) throw java.io.IOException("rename failed")
                }
            } else {
                downloadOne(ctx, m.talkerUrl, m.talkerFile, listener)
            }
            downloadOne(ctx, m.mmprojUrl, m.mmprojFile, listener)
            listener.onDone()
        } catch (e: InterruptedException) {
            listener.onError("canceled")
        } catch (e: Exception) {
            listener.onError(e.message ?: e.javaClass.simpleName)
        }
    }

    private fun downloadOne(ctx: Context, url: String, fileName: String, listener: DownloadListener) {
        val dest = File(modelsDir(ctx), fileName)
        if (dest.exists()) return
        val part = File(modelsDir(ctx), "$fileName.part")
        var attempt = 0
        while (true) {
            try {
                val existing = if (part.exists()) part.length() else 0L
                val conn = URL(url).openConnection() as HttpURLConnection
                conn.connectTimeout = 20000
                conn.readTimeout = 120000
                conn.instanceFollowRedirects = true
                if (existing > 0) conn.setRequestProperty("Range", "bytes=$existing-")
                conn.connect()
                val resumed = conn.responseCode == 206
                val total = if (resumed) existing + conn.contentLengthLong else conn.contentLengthLong
                val out = RandomAccessFile(part, "rw")
                out.seek(if (resumed) existing else 0L)
                var done = if (resumed) existing else 0L
                conn.inputStream.use { input ->
                    val buf = ByteArray(128 * 1024)
                    while (true) {
                        if (downloadCanceled) { out.close(); throw InterruptedException() }
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        listener.onProgress(fileName, done, total)
                    }
                }
                out.close()
                conn.disconnect()
                if (total > 0 && part.length() < total) throw java.io.IOException("short read: ${part.length()}/$total")
                if (!part.renameTo(dest)) throw java.io.IOException("rename failed")
                return
            } catch (e: InterruptedException) {
                throw e
            } catch (e: Exception) {
                attempt++
                if (attempt >= 5) throw e
                Thread.sleep(2000L * (1 shl attempt))
            }
        }
    }
}
