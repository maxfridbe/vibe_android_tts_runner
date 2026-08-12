package com.techhurts.ttsrunner

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.text.Normalizer
import kotlin.math.max
import kotlin.random.Random

/** Supertonic 3: a 99M-parameter ONNX TTS pipeline that runs far below
 *  real time on phone CPU, unlike the 1.7B Qwen model.
 *
 *  Four graphs, ported from the project's Java example (same ONNX Runtime API
 *  Android uses):
 *      text -> unicode ids -> duration_predictor -> total seconds
 *                          -> text_encoder       -> text embedding
 *      noise + text_emb + style --(vector_estimator, N flow steps)--> latent
 *      latent -> vocoder -> 44.1 kHz waveform
 *
 *  A "voice" here is a style file (style_ttl [1,50,256] + style_dp [1,8,16]),
 *  not reference audio: the published models contain no speaker encoder, so
 *  this engine cannot clone from a recording the way the Qwen path does. */
class SupertonicEngine(private val ctx: Context) {

    class Style(val ttl: Array<Array<FloatArray>>, val dp: Array<Array<FloatArray>>)

    private var env: OrtEnvironment? = null
    private var dp: OrtSession? = null
    private var textEnc: OrtSession? = null
    private var vecEst: OrtSession? = null
    private var vocoder: OrtSession? = null
    // Static-shape variants of the refine hot loop (batch 20, 92 probe tokens,
    // 128 latent frames): mobile GPU/NPU compilers refuse dynamic shapes, so
    // these are what an accelerator backend can actually place. Loaded on
    // demand by loadStatic(); generateBatch uses them when shapes match.
    private var textEncS: OrtSession? = null
    private var vecEstS: OrtSession? = null
    private var vocoderS: OrtSession? = null
    var staticBackend = "none"; private set
    private var indexer: IntArray = IntArray(0)
    var sampleRate = 44100; private set
    private var baseChunkSize = 0
    private var chunkCompressFactor = 0
    private var latentDim = 0
    /** Which execution provider actually loaded, for the UI/logs. */
    var provider = "cpu"; private set

    val loaded: Boolean get() = vocoder != null

    fun close() {
        runCatching { dp?.close(); textEnc?.close(); vecEst?.close(); vocoder?.close() }
        runCatching { textEncS?.close(); vecEstS?.close(); vocoderS?.close() }
        dp = null; textEnc = null; vecEst = null; vocoder = null
        textEncS = null; vecEstS = null; vocoderS = null; staticBackend = "none"
    }

    /** Load the static-shape refine graphs from [dir]/static with [backend]
     *  (e.g. "nnapi"). Optional: generateBatch falls back to the dynamic CPU
     *  sessions when these are absent or shapes don't match. */
    fun loadStatic(dir: File, backend: String, intraThreads: Int = 0): Boolean {
        val sdir = File(dir, "static")
        val files = listOf("text_encoder.static.onnx", "vector_estimator.static.onnx",
                           "vocoder.static.onnx").map { File(sdir, it) }
        if (files.any { !it.exists() }) return false
        return try {
            val e = OrtEnvironment.getEnvironment()
            val threads = if (intraThreads > 0) intraThreads
                else Runtime.getRuntime().availableProcessors().coerceAtMost(6)
            fun opts() = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(threads)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                // CPU_DISABLED: ops the accelerator driver rejects go back to
                // ORT's fast multithreaded CPU kernels instead of NNAPI's
                // single-threaded reference CPU (measured 2.7x SLOWER overall
                // without this). USE_FP16 lets the GPU run half precision.
                if (backend == "nnapi") addNnapi(java.util.EnumSet.of(
                    ai.onnxruntime.providers.NNAPIFlags.USE_FP16,
                    ai.onnxruntime.providers.NNAPIFlags.CPU_DISABLED))
                if (backend == "xnnpack") {
                    addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                    setIntraOpNumThreads(1)
                }
            }
            textEncS = e.createSession(files[0].absolutePath, opts())
            vecEstS = e.createSession(files[1].absolutePath, opts())
            vocoderS = e.createSession(files[2].absolutePath, opts())
            staticBackend = backend
            DebugLog.log(ctx, "Supertonic", "static refine graphs loaded ($backend)")
            true
        } catch (t: Throwable) {
            DebugLog.log(ctx, "Supertonic", "static load failed ($backend)", t as? Exception ?: Exception(t))
            runCatching { textEncS?.close(); vecEstS?.close(); vocoderS?.close() }
            textEncS = null; vecEstS = null; vocoderS = null; staticBackend = "none"
            false
        }
    }

    /** @param backend "cpu" | "nnapi" — NNAPI hands layers to the phone's
     *  accelerators (GPU/DSP/NPU) where the driver supports them and silently
     *  keeps the rest on CPU; unsupported devices fall back wholesale. */
    fun load(dir: File, backend: String, intraThreads: Int = 0): Boolean {
        close()
        return try {
            val e = OrtEnvironment.getEnvironment()
            env = e
            indexer = loadIndexer(File(dir, "unicode_indexer.json"))
            val cfg = JSONObject(File(dir, "tts.json").readText())
            sampleRate = cfg.getJSONObject("ae").getInt("sample_rate")
            baseChunkSize = cfg.getJSONObject("ae").getInt("base_chunk_size")
            chunkCompressFactor = cfg.getJSONObject("ttl").getInt("chunk_compress_factor")
            latentDim = cfg.getJSONObject("ttl").getInt("latent_dim")

            // A single utterance wants all the cores; the refine runs many
            // synths at once, so it passes a small count and parallelises across
            // candidates instead (pool × intra ≈ cores).
            val threads = if (intraThreads > 0) intraThreads
                else Runtime.getRuntime().availableProcessors().coerceAtMost(6)
            fun opts(): OrtSession.SessionOptions {
                val o = OrtSession.SessionOptions()
                o.setIntraOpNumThreads(threads)
                o.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                when (backend) {
                    // NNAPI hands ops to the GPU/NPU driver where it can. On
                    // Xclipse it loads but places nothing: measured identical
                    // to CPU, so it is offered, not preferred.
                    "nnapi" -> o.addNnapi()
                    // XNNPACK is ORT's optimised CPU kernel set; it owns its
                    // own thread pool, so the session must not also spawn one
                    "xnnpack" -> {
                        o.addXnnpack(mapOf("intra_op_num_threads" to threads.toString()))
                        o.setIntraOpNumThreads(1)
                    }
                }
                return o
            }
            provider = backend
            dp = e.createSession(File(dir, "duration_predictor.onnx").absolutePath, opts())
            textEnc = e.createSession(File(dir, "text_encoder.onnx").absolutePath, opts())
            vecEst = e.createSession(File(dir, "vector_estimator.onnx").absolutePath, opts())
            vocoder = e.createSession(File(dir, "vocoder.onnx").absolutePath, opts())
            DebugLog.log(ctx, "Supertonic", "loaded ($backend), sr=$sampleRate")
            true
        } catch (t: Throwable) {
            DebugLog.log(ctx, "Supertonic", "load failed ($backend)", t as? Exception ?: Exception(t))
            close()
            false
        }
    }

    // ---- search-path primitives (used by RefineEngine's batched CMA) -------

    /** Latent frames per second-of-audio granularity: samples per latent frame. */
    val latentChunk: Int get() = baseChunkSize * chunkCompressFactor
    /** Channel width of the flow's latent. */
    val latentWidth: Int get() = latentDim * chunkCompressFactor

    /** Text -> unicode indexer ids, the tokenisation generate() uses. */
    fun probeIds(text: String, lang: String = "en"): LongArray? {
        val prepared = preprocess(text, lang)
        if (prepared.isEmpty()) return null
        return prepared.map { indexer.getOrElse(it) { 0 }.toLong() }.toLongArray()
    }

    /** Seconds the duration predictor assigns [ids] with this style_dp. */
    fun probeDuration(ids: LongArray, dpStyle: Array<Array<FloatArray>>, speed: Float = 1.05f): Float? {
        val e = env ?: return null
        return try {
            val idsT = OnnxTensor.createTensor(e, LongBuffer.wrap(ids), longArrayOf(1, ids.size.toLong()))
            val maskT = OnnxTensor.createTensor(e, FloatBuffer.wrap(FloatArray(ids.size) { 1f }),
                longArrayOf(1, 1, ids.size.toLong()))
            val dpT = tensor3(e, dpStyle)
            val out = dp!!.run(mapOf("text_ids" to idsT, "style_dp" to dpT, "text_mask" to maskT))
            @Suppress("UNCHECKED_CAST")
            val d = (out[0].value as FloatArray)[0] / speed
            out.close(); idsT.close(); maskT.close(); dpT.close()
            d
        } catch (t: Throwable) {
            DebugLog.log(ctx, "Supertonic", "probeDuration failed", t as? Exception ?: Exception(t))
            null
        }
    }

    /** B candidate styles through text encoder, flow and vocoder as ONE batched
     *  pass (every graph input carries a dynamic batch dim — current_step and
     *  total_step included, which must be shaped [B]). The duration is frozen
     *  by the caller ([latentLen]/[wavLen]) and every candidate shares [noise]
     *  (common random numbers), so rows are comparable within a generation.
     *  Returns B float waveforms at [sampleRate], or null if a graph rejects
     *  the batch. */
    fun generateBatch(ids: LongArray, ttlsFlat: Array<FloatArray>, ttlRows: Int, ttlCols: Int,
                      latentLen: Int, wavLen: Int, noise: FloatArray, steps: Int): Array<FloatArray>? {
        val e = env ?: return null
        val bn = ttlsFlat.size
        if (bn == 0) return arrayOf()
        // Accelerator path: exact static shapes, latent mask-padded to STATIC_LL.
        if (textEncS != null && bn == STATIC_B && ids.size == STATIC_L && latentLen <= STATIC_LL) {
            generateBatchStatic(ids, ttlsFlat, ttlRows, ttlCols, latentLen, wavLen, noise, steps)
                ?.let { return it }
            // a failed static run closes those sessions so we don't retry every generation
            runCatching { textEncS?.close(); vecEstS?.close(); vocoderS?.close() }
            textEncS = null; vecEstS = null; vocoderS = null; staticBackend = "none"
            DebugLog.log(ctx, "Supertonic", "static path failed; dynamic CPU fallback")
        }
        val b = bn.toLong(); val len = ids.size
        return try {
            val idsB = LongArray(len * bn) { ids[it % len] }
            val maskB = FloatArray(len * bn) { 1f }
            val ttlB = FloatArray(ttlRows * ttlCols * bn)
            for (p in 0 until bn) System.arraycopy(ttlsFlat[p], 0, ttlB, p * ttlRows * ttlCols, ttlRows * ttlCols)
            val idsT = OnnxTensor.createTensor(e, LongBuffer.wrap(idsB), longArrayOf(b, len.toLong()))
            val maskT = OnnxTensor.createTensor(e, FloatBuffer.wrap(maskB), longArrayOf(b, 1, len.toLong()))
            val ttlT = OnnxTensor.createTensor(e, FloatBuffer.wrap(ttlB),
                longArrayOf(b, ttlRows.toLong(), ttlCols.toLong()))

            val encOut = textEnc!!.run(mapOf("text_ids" to idsT, "style_ttl" to ttlT, "text_mask" to maskT))
            val textEmb = encOut[0] as OnnxTensor

            val lDim = latentWidth
            val per = lDim * latentLen
            var xt = FloatArray(per * bn) { noise[it % per] }
            val latMaskT = OnnxTensor.createTensor(e, FloatBuffer.wrap(FloatArray(latentLen * bn) { 1f }),
                longArrayOf(b, 1, latentLen.toLong()))
            val totalT = OnnxTensor.createTensor(e, FloatBuffer.wrap(FloatArray(bn) { steps.toFloat() }),
                longArrayOf(b))
            for (step in 0 until steps) {
                val xtT = OnnxTensor.createTensor(e, FloatBuffer.wrap(xt),
                    longArrayOf(b, lDim.toLong(), latentLen.toLong()))
                val curT = OnnxTensor.createTensor(e, FloatBuffer.wrap(FloatArray(bn) { step.toFloat() }),
                    longArrayOf(b))
                val out = vecEst!!.run(mapOf(
                    "noisy_latent" to xtT, "text_emb" to textEmb, "style_ttl" to ttlT,
                    "text_mask" to maskT, "latent_mask" to latMaskT,
                    "current_step" to curT, "total_step" to totalT))
                xt = flatten(out[0].value)
                out.close(); xtT.close(); curT.close()
            }
            val latT = OnnxTensor.createTensor(e, FloatBuffer.wrap(xt),
                longArrayOf(b, lDim.toLong(), latentLen.toLong()))
            val wavOut = vocoder!!.run(mapOf("latent" to latT))
            val wav = flatten(wavOut[0].value)
            wavOut.close(); latT.close(); encOut.close()
            idsT.close(); maskT.close(); ttlT.close(); latMaskT.close(); totalT.close()

            val row = wav.size / bn
            val n = kotlin.math.min(row, max(1, wavLen))
            Array(bn) { p -> FloatArray(n) { i -> wav[p * row + i] } }
        } catch (t: Throwable) {
            DebugLog.log(ctx, "Supertonic", "generateBatch failed (B=$bn)", t as? Exception ?: Exception(t))
            null
        }
    }

    /** The static-shape twin of the dynamic batch path below: same tensors,
     *  fixed dims (STATIC_B/L/LL), latent padded past [latentLen] with zeros in
     *  the data and the mask. Returns null on any failure so the caller can
     *  fall back. */
    private fun generateBatchStatic(ids: LongArray, ttlsFlat: Array<FloatArray>, ttlRows: Int,
                                    ttlCols: Int, latentLen: Int, wavLen: Int, noise: FloatArray,
                                    steps: Int): Array<FloatArray>? {
        val e = env ?: return null
        val bn = STATIC_B; val b = bn.toLong(); val len = STATIC_L
        return try {
            val idsB = LongArray(len * bn) { ids[it % len] }
            val maskB = FloatArray(len * bn) { 1f }
            val ttlB = FloatArray(ttlRows * ttlCols * bn)
            for (p in 0 until bn) System.arraycopy(ttlsFlat[p], 0, ttlB, p * ttlRows * ttlCols, ttlRows * ttlCols)
            val idsT = OnnxTensor.createTensor(e, LongBuffer.wrap(idsB), longArrayOf(b, len.toLong()))
            val maskT = OnnxTensor.createTensor(e, FloatBuffer.wrap(maskB), longArrayOf(b, 1, len.toLong()))
            val ttlT = OnnxTensor.createTensor(e, FloatBuffer.wrap(ttlB),
                longArrayOf(b, ttlRows.toLong(), ttlCols.toLong()))
            val encOut = textEncS!!.run(mapOf("text_ids" to idsT, "style_ttl" to ttlT, "text_mask" to maskT))
            val textEmb = encOut[0] as OnnxTensor

            val lDim = latentWidth
            val perActual = lDim * latentLen
            val perPad = lDim * STATIC_LL
            // per-row: real noise laid out [lDim, latentLen] padded to [lDim, STATIC_LL]
            var xt = FloatArray(perPad * bn)
            for (p in 0 until bn) for (c in 0 until lDim)
                System.arraycopy(noise, c * latentLen, xt, p * perPad + c * STATIC_LL, latentLen)
            val latMask = FloatArray(STATIC_LL * bn)
            for (p in 0 until bn) for (i in 0 until latentLen) latMask[p * STATIC_LL + i] = 1f
            val latMaskT = OnnxTensor.createTensor(e, FloatBuffer.wrap(latMask),
                longArrayOf(b, 1, STATIC_LL.toLong()))
            val totalT = OnnxTensor.createTensor(e, FloatBuffer.wrap(FloatArray(bn) { steps.toFloat() }),
                longArrayOf(b))
            for (step in 0 until steps) {
                val xtT = OnnxTensor.createTensor(e, FloatBuffer.wrap(xt),
                    longArrayOf(b, lDim.toLong(), STATIC_LL.toLong()))
                val curT = OnnxTensor.createTensor(e, FloatBuffer.wrap(FloatArray(bn) { step.toFloat() }),
                    longArrayOf(b))
                val out = vecEstS!!.run(mapOf(
                    "noisy_latent" to xtT, "text_emb" to textEmb, "style_ttl" to ttlT,
                    "text_mask" to maskT, "latent_mask" to latMaskT,
                    "current_step" to curT, "total_step" to totalT))
                xt = flatten(out[0].value)
                out.close(); xtT.close(); curT.close()
            }
            val latT = OnnxTensor.createTensor(e, FloatBuffer.wrap(xt),
                longArrayOf(b, lDim.toLong(), STATIC_LL.toLong()))
            val wavOut = vocoderS!!.run(mapOf("latent" to latT))
            val wav = flatten(wavOut[0].value)
            wavOut.close(); latT.close(); encOut.close()
            idsT.close(); maskT.close(); ttlT.close(); latMaskT.close(); totalT.close()
            val row = wav.size / bn
            val n = kotlin.math.min(row, max(1, wavLen))
            Array(bn) { p -> FloatArray(n) { i -> wav[p * row + i] } }
        } catch (t: Throwable) {
            DebugLog.log(ctx, "Supertonic", "generateBatchStatic failed", t as? Exception ?: Exception(t))
            null
        }
    }

    /** Generates 16-bit PCM at [sampleRate] for one chunk of text. */
    fun generate(text: String, style: Style, lang: String = "en",
                 steps: Int = 8, speed: Float = 1.05f, seed: Int = 0): ByteArray? {
        val e = env ?: return null
        val prepared = preprocess(text, lang)
        if (prepared.isEmpty()) return null
        val ids = prepared.map { indexer.getOrElse(it) { 0 }.toLong() }.toLongArray()
        val len = ids.size
        val mask = FloatArray(len) { 1f }

        return try {
            val idsT = OnnxTensor.createTensor(e, LongBuffer.wrap(ids), longArrayOf(1, len.toLong()))
            val maskT = OnnxTensor.createTensor(e, FloatBuffer.wrap(mask), longArrayOf(1, 1, len.toLong()))
            val ttlT = tensor3(e, style.ttl)
            val dpT = tensor3(e, style.dp)

            val durOut = dp!!.run(mapOf("text_ids" to idsT, "style_dp" to dpT, "text_mask" to maskT))
            @Suppress("UNCHECKED_CAST")
            val duration = (durOut[0].value as FloatArray)[0] / speed
            durOut.close()

            val encOut = textEnc!!.run(mapOf("text_ids" to idsT, "style_ttl" to ttlT, "text_mask" to maskT))
            val textEmb = encOut[0] as OnnxTensor

            // noisy latent, shaped from the predicted duration
            val wavLen = (duration * sampleRate).toLong()
            val chunk = baseChunkSize * chunkCompressFactor
            val latentLen = ((wavLen + chunk - 1) / chunk).toInt().coerceAtLeast(1)
            val lDim = latentDim * chunkCompressFactor
            val rnd = if (seed != 0) Random(seed) else Random.Default
            var xt = FloatArray(lDim * latentLen) { gaussian(rnd) }
            val latentMask = FloatArray(latentLen) { 1f }
            val latentMaskT = OnnxTensor.createTensor(e, FloatBuffer.wrap(latentMask),
                longArrayOf(1, 1, latentLen.toLong()))
            val totalStepT = OnnxTensor.createTensor(e, FloatBuffer.wrap(floatArrayOf(steps.toFloat())),
                longArrayOf(1))

            for (step in 0 until steps) {
                val xtT = OnnxTensor.createTensor(e, FloatBuffer.wrap(xt),
                    longArrayOf(1, lDim.toLong(), latentLen.toLong()))
                val curT = OnnxTensor.createTensor(e, FloatBuffer.wrap(floatArrayOf(step.toFloat())),
                    longArrayOf(1))
                val out = vecEst!!.run(mapOf(
                    "noisy_latent" to xtT, "text_emb" to textEmb, "style_ttl" to ttlT,
                    "text_mask" to maskT, "latent_mask" to latentMaskT,
                    "current_step" to curT, "total_step" to totalStepT))
                xt = flatten(out[0].value)
                out.close(); xtT.close(); curT.close()
            }

            val latT = OnnxTensor.createTensor(e, FloatBuffer.wrap(xt),
                longArrayOf(1, lDim.toLong(), latentLen.toLong()))
            val wavOut = vocoder!!.run(mapOf("latent" to latT))
            val wav = flatten(wavOut[0].value)
            wavOut.close(); latT.close(); encOut.close()
            idsT.close(); maskT.close(); ttlT.close(); dpT.close()
            latentMaskT.close(); totalStepT.close()

            // trim to the predicted duration and convert to s16le
            val n = kotlin.math.min(wav.size, max(1, wavLen.toInt()))
            val pcm = ByteArray(n * 2)
            for (i in 0 until n) {
                val v = (wav[i].coerceIn(-1f, 1f) * 32767f).toInt()
                pcm[2 * i] = (v and 0xff).toByte()
                pcm[2 * i + 1] = (v shr 8 and 0xff).toByte()
            }
            pcm
        } catch (t: Throwable) {
            DebugLog.log(ctx, "Supertonic", "generate failed", t as? Exception ?: Exception(t))
            null
        }
    }

    // ---- helpers -----------------------------------------------------------

    private fun gaussian(r: Random): Float {
        // Box-Muller; the reference uses numpy randn
        val u1 = r.nextDouble().coerceAtLeast(1e-9)
        val u2 = r.nextDouble()
        return (kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) *
                kotlin.math.cos(2.0 * Math.PI * u2)).toFloat()
    }

    private fun tensor3(e: OrtEnvironment, a: Array<Array<FloatArray>>): OnnxTensor {
        val d0 = a.size; val d1 = a[0].size; val d2 = a[0][0].size
        val flat = FloatArray(d0 * d1 * d2)
        var k = 0
        for (i in 0 until d0) for (j in 0 until d1) for (l in 0 until d2) flat[k++] = a[i][j][l]
        return OnnxTensor.createTensor(e, FloatBuffer.wrap(flat),
            longArrayOf(d0.toLong(), d1.toLong(), d2.toLong()))
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
            walk(v)
            out.toFloatArray()
        }
        else -> FloatArray(0)
    }

    /** Mirrors the reference preprocessor, then maps code points to indices. */
    private fun preprocess(raw: String, lang: String): IntArray {
        var t = Normalizer.normalize(raw, Normalizer.Form.NFKD)
        t = t.replace(Regex("[\\uD800-\\uDBFF][\\uDC00-\\uDFFF]"), "")   // emoji/astral
        val repl = mapOf(
            "–" to "-", "‑" to "-", "—" to "-", "_" to " ", "“" to "\"", "”" to "\"",
            "‘" to "'", "’" to "'", "´" to "'", "`" to "'", "[" to " ", "]" to " ",
            "|" to " ", "/" to " ", "#" to " ", "→" to " ", "←" to " ")
        for ((k, v) in repl) t = t.replace(k, v)
        t = t.replace(Regex("[♥☆♡©\\\\]"), "")
        t = t.replace("@", " at ").replace("e.g.,", "for example, ").replace("i.e.,", "that is, ")
        t = t.replace(Regex(" ,"), ",").replace(Regex(" \\."), ".").replace(Regex(" !"), "!")
            .replace(Regex(" \\?"), "?").replace(Regex(" ;"), ";").replace(Regex(" :"), ":")
            .replace(Regex(" '"), "'")
        while (t.contains("\"\"")) t = t.replace("\"\"", "\"")
        while (t.contains("''")) t = t.replace("''", "'")
        t = t.replace(Regex("\\s+"), " ").trim()
        if (!Regex(".*[.!?;:,'\"\\u201C\\u201D\\u2018\\u2019)\\]}…。」』】〉》›»]$").matches(t)) t += "."
        t = "<$lang>$t</$lang>"
        return t.codePoints().toArray()
    }

    /** unicode_indexer.json is a flat array indexed by code point (-1 where a
     *  character has no embedding), exactly as the reference implementations
     *  index it. */
    private fun loadIndexer(f: File): IntArray {
        val a = org.json.JSONArray(f.readText())
        return IntArray(a.length()) { a.getInt(it) }
    }

    companion object {
        // Static refine-graph dims: batch = population, tokens of the rich
        // search probe, latent frames (~8.9 s headroom, mask-padded).
        const val STATIC_B = 20
        const val STATIC_L = 92
        const val STATIC_LL = 128

        /** Style JSON as produced by the project's voice styles / Voice Builder. */
        fun loadStyle(f: File): Style? = try {
            val o = JSONObject(f.readText())
            fun read(name: String): Array<Array<FloatArray>> {
                val s = o.getJSONObject(name)
                val dims = s.getJSONArray("dims")
                val d1 = dims.getInt(1); val d2 = dims.getInt(2)
                val data = s.getJSONArray("data")
                // data may be nested or flat; walk it into a flat list
                val flat = ArrayList<Float>(d1 * d2)
                fun walk(a: org.json.JSONArray) {
                    for (i in 0 until a.length()) {
                        val v = a.get(i)
                        if (v is org.json.JSONArray) walk(v) else flat.add((v as Number).toFloat())
                    }
                }
                walk(data)
                return Array(1) { Array(d1) { r -> FloatArray(d2) { c -> flat[r * d2 + c] } } }
            }
            Style(read("style_ttl"), read("style_dp"))
        } catch (_: Exception) { null }
    }
}
