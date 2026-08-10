package com.maxfridbe.ttsrunner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/** On-device voice cloning for Supertonic — EXPERIMENTAL.
 *
 *  The desktop path optimises a style against a recording over hundreds of
 *  iterations. This does it in two forward passes instead: a speaker encoder
 *  turns the recording into an identity embedding, and a small network trained
 *  on brainiac predicts the style tensors from that embedding (see
 *  docs/on-device-cloning.md). Sub-second, but only as good as its training —
 *  the desktop cloner remains the quality reference.
 *
 *  Both graphs ship in the APK under assets/. Their absence just hides the
 *  feature, so builds without a trained encoder still work. */
class VoiceCloner(private val ctx: Context) {

    private var env: OrtEnvironment? = null
    private var spk: OrtSession? = null
    private var style: OrtSession? = null

    companion object {
        const val SPK_ASSET = "spk_encoder.onnx"
        const val STYLE_ASSET = "style_encoder.onnx"
        private const val RATE = 16000

        /** Side-loaded encoder, which is how it ships while it is still being
         *  trained: 89 MB of ONNX does not belong in the APK until the thing
         *  is good enough to keep. Push the pair into this folder and the
         *  feature appears.
         *
         *      adb push spk_encoder.onnx /data/local/tmp/
         *      adb shell run-as com.maxfridbe.ttsrunner \
         *          cp /data/local/tmp/spk_encoder.onnx files/cloner/ */
        fun dir(ctx: Context): java.io.File = java.io.File(ctx.filesDir, "cloner").apply { mkdirs() }

        private fun sideloaded(ctx: Context) =
            java.io.File(dir(ctx), SPK_ASSET).exists() && java.io.File(dir(ctx), STYLE_ASSET).exists()

        private fun embedded(ctx: Context) = try {
            ctx.assets.list("")?.toSet()?.containsAll(listOf(SPK_ASSET, STYLE_ASSET)) == true
        } catch (_: Exception) { false }

        /** Whether this phone has a trained encoder at all, from either place. */
        fun available(ctx: Context): Boolean = sideloaded(ctx) || embedded(ctx)
    }

    private fun bytes(name: String): ByteArray {
        val f = java.io.File(dir(ctx), name)
        return if (f.exists()) f.readBytes() else ctx.assets.open(name).use { it.readBytes() }
    }

    private fun load(): Boolean {
        if (style != null) return true
        return try {
            val e = OrtEnvironment.getEnvironment()
            env = e
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            }
            spk = e.createSession(bytes(SPK_ASSET), opts)
            style = e.createSession(bytes(STYLE_ASSET), opts)
            true
        } catch (t: Throwable) {
            DebugLog.log(ctx, "VoiceCloner", "load failed", t as? Exception ?: Exception(t))
            close()
            false
        }
    }

    fun close() {
        runCatching { spk?.close(); style?.close() }
        spk = null; style = null
    }

    /** Reads [wav], predicts a style, and writes it to the voice library.
     *  Returns the new voice, or null with the reason logged. */
    fun cloneToStyle(wav: File, name: String): VoiceStore.Voice? {
        if (!load()) return null
        val e = env ?: return null
        return try {
            val audio = readWav16k(wav) ?: run {
                DebugLog.log(ctx, "VoiceCloner", "unreadable wav: ${wav.name}"); return null
            }
            // 16 kHz mono, capped: the encoder's identity estimate saturates
            // after a few seconds and long clips only cost time
            val n = minOf(audio.size, RATE * 12)
            val t0 = System.currentTimeMillis()
            val wavT = OnnxTensor.createTensor(e, FloatBuffer.wrap(audio, 0, n),
                longArrayOf(1, n.toLong()))
            val embOut = spk!!.run(mapOf("wav" to wavT))
            val emb = flatten(embOut[0].value)
            embOut.close(); wavT.close()

            val embT = OnnxTensor.createTensor(e, FloatBuffer.wrap(emb),
                longArrayOf(1, emb.size.toLong()))
            val styleOut = style!!.run(mapOf("embedding" to embT))
            val ttl = flatten(styleOut[0].value)
            val dp = flatten(styleOut[1].value)
            styleOut.close(); embT.close()
            DebugLog.log(ctx, "VoiceCloner",
                "cloned ${wav.name} in ${System.currentTimeMillis() - t0} ms " +
                "(${n / RATE}s audio, ttl ${ttl.size}, dp ${dp.size})")

            val json = JSONObject()
                .put("style_ttl", JSONObject()
                    .put("dims", JSONArray(listOf(1, 50, 256)))
                    .put("data", JSONArray(ttl.map { it.toDouble() })))
                .put("style_dp", JSONObject()
                    .put("dims", JSONArray(listOf(1, 8, 16)))
                    .put("data", JSONArray(dp.map { it.toDouble() })))
                .put("metadata", JSONObject()
                    .put("source", "on-device encoder (experimental)")
                    .put("reference", wav.name))
            val tmp = File(ctx.cacheDir, "cloned.json").apply { writeText(json.toString()) }
            VoiceStore.importStyle(ctx, tmp, name)
        } catch (t: Throwable) {
            DebugLog.log(ctx, "VoiceCloner", "clone failed", t as? Exception ?: Exception(t))
            null
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun flatten(v: Any?): FloatArray = when (v) {
        is FloatArray -> v
        is Array<*> -> {
            val out = ArrayList<Float>()
            fun walk(x: Any?) {
                when (x) {
                    is FloatArray -> x.forEach { out.add(it) }
                    is Array<*> -> x.forEach { walk(it) }
                }
            }
            walk(v); out.toFloatArray()
        }
        else -> FloatArray(0)
    }

    /** Minimal 16-bit PCM WAV reader with linear resampling to 16 kHz mono —
     *  the app's own recordings and imports are all this format. */
    private fun readWav16k(f: File): FloatArray? {
        val b = f.readBytes()
        if (b.size < 44 || String(b, 0, 4) != "RIFF") return null
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12
        var rate = 0; var channels = 1; var bits = 16
        var dataOff = -1; var dataLen = 0
        while (pos + 8 <= b.size) {
            val id = String(b, pos, 4)
            val size = bb.getInt(pos + 4)
            when (id) {
                "fmt " -> {
                    channels = bb.getShort(pos + 10).toInt()
                    rate = bb.getInt(pos + 12)
                    bits = bb.getShort(pos + 22).toInt()
                }
                "data" -> { dataOff = pos + 8; dataLen = size }
            }
            if (dataOff > 0) break
            pos += 8 + size + (size and 1)
        }
        if (dataOff < 0 || rate <= 0 || bits != 16) return null
        val frames = minOf(dataLen, b.size - dataOff) / 2 / channels
        val mono = FloatArray(frames)
        for (i in 0 until frames) {
            var acc = 0f
            for (c in 0 until channels) {
                acc += bb.getShort(dataOff + (i * channels + c) * 2) / 32768f
            }
            mono[i] = acc / channels
        }
        if (rate == RATE) return mono
        val outN = (frames.toLong() * RATE / rate).toInt()
        if (outN <= 1) return null
        val out = FloatArray(outN)
        for (i in 0 until outN) {
            val x = i.toDouble() * (frames - 1) / (outN - 1)
            val i0 = x.toInt().coerceAtMost(frames - 1)
            val i1 = (i0 + 1).coerceAtMost(frames - 1)
            val t = (x - i0).toFloat()
            out[i] = mono[i0] * (1 - t) + mono[i1] * t
        }
        return out
    }
}
