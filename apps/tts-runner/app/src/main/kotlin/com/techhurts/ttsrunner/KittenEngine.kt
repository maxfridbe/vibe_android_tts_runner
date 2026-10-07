package com.techhurts.ttsrunner

import org.json.JSONArray
import org.json.JSONObject
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.PyTorchAndroid
import org.pytorch.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Kitten TTS 2: CPU GGML speech tokens followed by the published S3 decoder. */
class KittenEngine : AutoCloseable {
    private var decoder: Module? = null
    private var voices = JSONObject()
    @Volatile private var canceled = false

    fun load(dir: File, threads: Int) {
        close()
        try {
            voices = JSONObject(File(dir, "voices.json").readText())
            PyTorchAndroid.setNumThreads(threads)
            decoder = Module.load(File(dir, "decoder.pt").absolutePath)
            KittenNative.load(File(dir, "model-tq2_1.gguf").absolutePath.toByteArray(Charsets.UTF_8),
                File(dir, "config.json").readBytes(), threads)
        } catch (t: Throwable) {
            close()
            throw t
        }
    }

    fun resetCancel() { canceled = false; KittenNative.resetCancel() }
    fun cancel() { canceled = true; KittenNative.cancel() }

    fun generate(text: String, voiceName: String, seed: Int): ByteArray {
        check(!canceled) { "Canceled" }
        val voice = voices.getJSONObject(voiceName.removePrefix(VOICE_PREFIX))
        // Keep non-English numbers and expression markup intact. The upstream
        // English grammar normalizer is not part of the Android runtime.
        val spoken = text.trim().replace(Regex("\\s+"), " ")
        require(spoken.isNotBlank()) { "Nothing to speak" }
        val expression = EXPRESSION.containsMatchIn(spoken)
        val tokens = KittenNative.generate(spoken.toByteArray(Charsets.UTF_8),
            voice.toString().toByteArray(Charsets.UTF_8), seed, expression)
        check(!canceled) { "Canceled" }
        val padded = LongArray(tokens.size + 3) { if (it < tokens.size) tokens[it].toLong() else 4299L }
        val refTokens = voice.getJSONArray("prompt_token").getJSONArray(0).longs()
        val rows = voice.getJSONArray("prompt_feat").getJSONArray(0)
        val features = FloatArray(rows.length() * 80) { i -> rows.getJSONArray(i / 80).getDouble(i % 80).toFloat() }
        val embedding = voice.getJSONArray("embedding").getJSONArray(0).floats()
        val module = checkNotNull(decoder) { "Kitten decoder is not loaded" }
        val audio = module.forward(
            IValue.from(Tensor.fromBlob(padded, longArrayOf(1, padded.size.toLong()))),
            IValue.from(Tensor.fromBlob(refTokens, longArrayOf(1, refTokens.size.toLong()))),
            IValue.from(Tensor.fromBlob(features, longArrayOf(1, rows.length().toLong(), 80))),
            IValue.from(Tensor.fromBlob(embedding, longArrayOf(1, embedding.size.toLong())))
        ).toTensor().dataAsFloatArray
        check(!canceled) { "Canceled" }
        check(audio.isNotEmpty() && audio.all { it.isFinite() }) { "Kitten decoder returned invalid audio" }
        return ByteBuffer.allocate(audio.size * 2).order(ByteOrder.LITTLE_ENDIAN).apply {
            audio.forEach { putShort((it.coerceIn(-1f, 1f) * 32767).toInt().toShort()) }
        }.array()
    }

    override fun close() {
        decoder?.destroy(); decoder = null
        KittenNative.unload()
        voices = JSONObject()
    }

    companion object {
        const val VOICE_PREFIX = "Kitten: "
        const val SAMPLE_RATE = 24000
        private val EXPRESSION = Regex(
            "\\[(angry|contemplative|excited|joyful|mundane|nervous|sad|stern|surprised|tender)]|" +
                "<(gasp|giggle|growl|gulp|laugh|pause|scoff|sigh|sob|um)>|\\(\\(\\([^()\\n]{1,80}\\)\\)\\)",
            RegexOption.IGNORE_CASE)
        private fun JSONArray.longs() = LongArray(length()) { getLong(it) }
        private fun JSONArray.floats() = FloatArray(length()) { getDouble(it).toFloat() }
    }
}

internal object KittenNative {
    init { System.loadLibrary("ttsrunner_jni") }
    external fun load(path: ByteArray, config: ByteArray, threads: Int)
    external fun generate(text: ByteArray, voice: ByteArray, seed: Int, expression: Boolean): IntArray
    external fun resetCancel()
    external fun cancel()
    external fun unload()
}
