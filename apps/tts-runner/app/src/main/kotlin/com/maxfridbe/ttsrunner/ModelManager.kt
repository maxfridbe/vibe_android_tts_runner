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

    /** Release holding GGUFs that upstream does not publish (converted with
     *  this repo's llama.cpp patch). */
    private const val MODEL_RELEASE =
        "https://github.com/maxfridbe/vibe_android_tts_runner/releases/download/models-v1"


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
        /** VoiceDesign: no speaker encoder, so it can't clone voices — it is
         *  only used by the voice designer, never for regular TTS jobs. */
        val designOnly: Boolean = false,
        /** Supertonic runs ONNX graphs, not llama.cpp GGUFs: several files,
         *  its own engine, and style-file voices instead of reference audio. */
        val engine: String = "qwen",
        val extraFiles: List<Pair<String, String>> = emptyList(),  // url to filename
    )

    /** Supertonic 3: 99M params over four ONNX graphs (~398 MB), 44.1 kHz,
     *  well under real time on phone CPU. */
    private const val ST_BASE =
        "https://huggingface.co/Supertone/supertonic-3/resolve/main/onnx"

    val SUPERTONIC_FILES = listOf(
        "$ST_BASE/duration_predictor.onnx" to "duration_predictor.onnx",
        "$ST_BASE/text_encoder.onnx" to "text_encoder.onnx",
        "$ST_BASE/vector_estimator.onnx" to "vector_estimator.onnx",
        "$ST_BASE/vocoder.onnx" to "vocoder.onnx",
        "$ST_BASE/tts.json" to "tts.json",
        "$ST_BASE/unicode_indexer.json" to "unicode_indexer.json",
    )

    /** The published voice styles — Supertonic's equivalent of voices. */
    private const val ST_STYLES =
        "https://huggingface.co/Supertone/supertonic-3/resolve/main/voice_styles"
    val SUPERTONIC_STYLES = listOf("F1", "F2", "F3", "F4", "F5", "M1", "M2", "M3", "M4", "M5")
    fun styleUrl(name: String) = "$ST_STYLES/$name.json"

    fun supertonicDir(ctx: Context) = File(modelsDir(ctx), "supertonic-3").apply { mkdirs() }

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
        // Upstream ships no GGUF of the VoiceDesign variant, so this repo
        // hosts the conversion as release assets (see tooling/publish_models.sh).
        CatalogModel(
            id = "1.7b-vd",
            label = "Qwen3-TTS 1.7B VoiceDesign (1.4 GB — enables describe-a-voice)",
            talkerUrl = "$MODEL_RELEASE/Qwen3-TTS-VD-Q4_K_M.gguf",
            talkerFile = "Qwen3-TTS-VD-Q4_K_M.gguf",
            mmprojUrl = "$MODEL_RELEASE/mmproj-Qwen3-TTS-VD-Q8_0.gguf",
            mmprojFile = "mmproj-Qwen3-TTS-VD-Q8_0.gguf",
            totalBytes = 1_035_965_568L + 422_392_192L,
            designOnly = true,
        ),
        CatalogModel(
            id = "supertonic-3",
            label = "Supertonic 3 (400 MB — fast, 31 languages, style voices)",
            talkerUrl = "", talkerFile = "supertonic-3/vocoder.onnx",
            mmprojUrl = "", mmprojFile = "supertonic-3/vector_estimator.onnx",
            totalBytes = 398_000_000L,
            engine = "supertonic",
            extraFiles = SUPERTONIC_FILES,
        ),
    )

    fun modelsDir(ctx: Context): File {
        val ext = ctx.getExternalFilesDir("models")
        return (ext ?: File(ctx.filesDir, "models")).apply { mkdirs() }
    }

    fun isDownloaded(ctx: Context, m: CatalogModel): Boolean =
        if (m.extraFiles.isNotEmpty())
            m.extraFiles.all { (_, n) -> File(supertonicDir(ctx), n).exists() }
        else File(modelsDir(ctx), m.talkerFile).exists() && File(modelsDir(ctx), m.mmprojFile).exists()

    /** Style voices that came with the Supertonic download. */
    fun supertonicStyles(ctx: Context): List<File> =
        File(supertonicDir(ctx), "styles").listFiles { f -> f.extension == "json" }
            ?.sortedBy { it.name } ?: emptyList()

    fun selectedModel(ctx: Context): CatalogModel? {
        val prefs = ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
        val id = prefs.getString("model_id", null)
        val m = CATALOG.find { it.id == id && !it.designOnly }
            ?: CATALOG.firstOrNull { !it.designOnly && isDownloaded(ctx, it) }
        return m?.takeIf { isDownloaded(ctx, it) }
    }

    fun designModel(ctx: Context): CatalogModel? =
        CATALOG.find { it.designOnly && isDownloaded(ctx, it) }

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
            if (m.extraFiles.isNotEmpty()) {
                // multi-file model (Supertonic): graphs into their own dir,
                // then the published voice styles
                val dir = if (m.engine == "supertonic") "supertonic-3" else ""
                for ((url, name) in m.extraFiles) downloadOne(ctx, url, "$dir/$name", listener)
                if (m.engine == "supertonic") {
                    for (s in SUPERTONIC_STYLES) {
                        listener.onProgress("voice style $s", 0, 0)
                        runCatching { downloadOne(ctx, styleUrl(s), "$dir/styles/$s.json", listener) }
                    }
                }
                listener.onDone()
                return
            }
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
        dest.parentFile?.mkdirs()
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
