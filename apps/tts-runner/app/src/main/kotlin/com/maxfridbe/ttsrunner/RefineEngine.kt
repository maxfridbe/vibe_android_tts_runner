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
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt

/** Forward-only voice inversion on the phone — "Refine this voice".
 *
 *  The encoder lands near a speaker, not on them. Gradient inversion would
 *  close the gap but needs autograd and gigabytes; a phone has neither. So
 *  this searches the same PCA basis with a gradient-free optimiser
 *  (separable CMA-ES): decode a candidate's coefficients to a style,
 *  synthesise a short line with Supertonic, embed it with ECAPA, score its
 *  cosine to the reference recording — a few hundred forward passes the app
 *  already runs. This is the exact loop `tooling/cma_polish.py` measured
 *  (~0.57 held-out from a ~0.49 start), ported to Kotlin.
 *
 *  Needs the models the cloner downloads (spk_encoder.onnx + style_basis.bin);
 *  runs on the caller's thread and reports progress by fraction. */
class RefineEngine(private val ctx: Context) {

    companion object {
        const val BASIS_ASSET = "style_basis.bin"
        // one short, out-of-the-ordinary probe (disjoint from what any head
        // trained or was scored on) keeps each evaluation ~1 s on phone CPU
        private const val PROBE = "A cup of coffee on the desk had long since gone cold."
        private const val RATE = 16000

        fun available(ctx: Context): Boolean =
            File(VoiceCloner.dir(ctx), BASIS_ASSET).length() > 0 &&
                VoiceCloner.variants(ctx).contains(VoiceCloner.VARIANT_ECAPA)
    }

    // ---- basis (style_basis.bin) ------------------------------------------
    private var k = 0; private var dTot = 0; private var split = 0
    private var ttlR = 0; private var ttlC = 0; private var dpR = 0; private var dpC = 0
    private lateinit var scale: FloatArray      // (k)
    private lateinit var mean: FloatArray       // (dTot)
    private lateinit var basis: FloatArray       // (k * dTot) row-major

    private var env: OrtEnvironment? = null
    private var spk: OrtSession? = null
    private var supertonic: SupertonicEngine? = null

    private fun loadBasis(): Boolean {
        val f = File(VoiceCloner.dir(ctx), BASIS_ASSET)
        if (f.length() < 28) return false
        val b = ByteBuffer.wrap(f.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
        k = b.int; dTot = b.int; split = b.int
        ttlR = b.int; ttlC = b.int; dpR = b.int; dpC = b.int
        scale = FloatArray(k) { b.float }
        mean = FloatArray(dTot) { b.float }
        basis = FloatArray(k * dTot) { b.float }
        return true
    }

    private fun load(backend: String): Boolean {
        if (supertonic?.loaded == true) return true
        if (!loadBasis()) return false
        return try {
            val e = OrtEnvironment.getEnvironment(); env = e
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
            }
            val spkBytes = File(VoiceCloner.dir(ctx), VoiceCloner.SPK_ASSET).let {
                if (it.exists()) it.readBytes() else ctx.assets.open(VoiceCloner.SPK_ASSET).use { s -> s.readBytes() }
            }
            spk = e.createSession(spkBytes, opts)
            supertonic = SupertonicEngine(ctx).also {
                if (!it.load(ModelManager.supertonicDir(ctx), backend)) return false
            }
            true
        } catch (t: Throwable) {
            DebugLog.log(ctx, "RefineEngine", "load failed", t as? Exception ?: Exception(t))
            close(); false
        }
    }

    fun close() {
        runCatching { spk?.close(); supertonic?.close() }
        spk = null; supertonic = null
    }

    // ---- basis decode / encode (match export_translation's StyleHead) ------

    /** coeffs -> style, with the per-row L2 normalisation the synthesiser wants. */
    private fun decode(c: FloatArray): SupertonicEngine.Style {
        val flat = mean.copyOf()
        for (i in 0 until k) {
            val cs = c[i] * scale[i]
            if (cs == 0f) continue
            val base = i * dTot
            for (j in 0 until dTot) flat[j] += cs * basis[base + j]
        }
        val ttl = Array(1) { Array(ttlR) { r -> FloatArray(ttlC) { col -> flat[r * ttlC + col] } } }
        val dp = Array(1) { Array(dpR) { r -> FloatArray(dpC) { col -> flat[split + r * dpC + col] } } }
        rowNorm(ttl[0]); rowNorm(dp[0])
        return SupertonicEngine.Style(ttl, dp)
    }

    /** style -> coeffs = ((flatten - mean) @ basis^T) / scale. */
    private fun encode(s: SupertonicEngine.Style): FloatArray {
        val flat = FloatArray(dTot)
        for (r in 0 until ttlR) for (col in 0 until ttlC) flat[r * ttlC + col] = s.ttl[0][r][col]
        for (r in 0 until dpR) for (col in 0 until dpC) flat[split + r * dpC + col] = s.dp[0][r][col]
        for (j in 0 until dTot) flat[j] -= mean[j]
        val c = FloatArray(k)
        for (i in 0 until k) {
            var acc = 0f; val base = i * dTot
            for (j in 0 until dTot) acc += basis[base + j] * flat[j]
            c[i] = acc / scale[i]
        }
        return c
    }

    private fun rowNorm(m: Array<FloatArray>) {
        for (row in m) {
            var n = 0f; for (v in row) n += v * v
            n = sqrt(n) + 1e-8f
            for (j in row.indices) row[j] /= n
        }
    }

    // ---- one evaluation: coeffs -> cosine to the target --------------------

    private lateinit var target: FloatArray     // reference ECAPA embedding

    private fun cosineOf(c: FloatArray): Float {
        val style = decode(c)
        val wav = supertonic?.generate(PROBE, style, steps = 4) ?: return -1f
        val emb = embed(wav) ?: return -1f
        var dot = 0f
        for (i in emb.indices) dot += emb[i] * target[i]
        return dot                              // both unit-norm out of spk_encoder
    }

    /** WAV bytes (any rate, 16-bit mono) -> unit-norm ECAPA embedding. */
    private fun embed(wavBytes: ByteArray): FloatArray? {
        val pcm = wavToMono16k(wavBytes) ?: return null
        val e = env ?: return null
        val t = OnnxTensor.createTensor(e, FloatBuffer.wrap(pcm), longArrayOf(1, pcm.size.toLong()))
        val out = spk!!.run(mapOf("wav" to t))
        val emb = flatten(out[0].value)
        out.close(); t.close()
        return emb
    }

    // ---- separable CMA-ES (port of tooling/cma_polish.sep_cma) -------------

    private class Rng(seed: Long) {
        private var s = seed
        fun next(): Float { s = s * 6364136223846793005L + 1442695040888963407L; return ((s ushr 33).toInt() / 2.147483648E9).toFloat() }
        fun gauss(): Float {   // Box–Muller
            var u = next(); var v = next()
            if (u < 1e-7f) u = 1e-7f
            return (sqrt(-2.0 * ln(u.toDouble())) * kotlin.math.cos(2.0 * Math.PI * v)).toFloat()
        }
    }

    /** Minimise f. Returns the best coeffs found. Reports progress 0..1 and
     *  bails early when [alive] turns false (user cancelled). */
    private fun sepCma(x0: FloatArray, sigma0: Float, iters: Int, pop: Int,
                       f: (FloatArray) -> Float, onProgress: (Float) -> Unit,
                       alive: () -> Boolean): FloatArray {
        val n = x0.size
        val mu = pop / 2
        val w = FloatArray(mu) { ln(mu + 0.5f) - ln((it + 1).toFloat()) }
        val wsum = w.sum(); for (i in w.indices) w[i] /= wsum
        var muEff = 0f; for (x in w) muEff += x * x; muEff = 1f / muEff
        val cs = (muEff + 2) / (n + muEff + 5)
        val ds = 1 + cs + 2 * maxOf(0f, sqrt((muEff - 1) / (n + 1)) - 1)
        val cc = (4 + muEff / n) / (n + 4 + 2 * muEff / n)
        val c1 = 2 / ((n + 1.3f) * (n + 1.3f) + muEff)
        val cmu = minOf(1 - c1, 2 * (muEff - 2 + 1 / muEff) / ((n + 2f) * (n + 2f) + muEff))
        val chiN = sqrt(n.toFloat()) * (1 - 1f / (4 * n) + 1f / (21 * n * n))
        val rng = Rng(1234567)

        val m = x0.copyOf(); var sigma = sigma0
        val C = FloatArray(n) { 1f }; val ps = FloatArray(n); val pc = FloatArray(n)
        var bestX = x0.copyOf(); var bestF = f(x0)
        var evals = 1
        val total = 1 + iters * pop
        for (g in 0 until iters) {
            if (!alive()) break
            val z = Array(pop) { FloatArray(n) { rng.gauss() } }
            val x = Array(pop) { p -> FloatArray(n) { j -> (m[j] + sigma * sqrt(C[j]) * z[p][j]).coerceIn(-3f, 3f) } }
            val scores = FloatArray(pop) { p -> f(x[p]).also { evals++; onProgress(evals.toFloat() / total) } }
            val order = (0 until pop).sortedBy { scores[it] }
            if (scores[order[0]] < bestF) { bestF = scores[order[0]]; bestX = x[order[0]].copyOf() }
            val zmean = FloatArray(n); val newM = FloatArray(n)
            for (r in 0 until mu) { val idx = order[r]; for (j in 0 until n) { zmean[j] += w[r] * z[idx][j]; newM[j] += w[r] * x[idx][j] } }
            System.arraycopy(newM, 0, m, 0, n)
            var psNorm = 0f
            for (j in 0 until n) { ps[j] = (1 - cs) * ps[j] + sqrt(cs * (2 - cs) * muEff) * zmean[j]; psNorm += ps[j] * ps[j] }
            psNorm = sqrt(psNorm)
            sigma *= exp((cs / ds) * (psNorm / chiN - 1))
            val hs = if (psNorm / sqrt(1 - Math.pow((1 - cs).toDouble(), (2.0 * (g + 1))).toFloat()) < (1.4f + 2f / (n + 1)) * chiN) 1f else 0f
            for (j in 0 until n) {
                pc[j] = (1 - cc) * pc[j] + hs * sqrt(cc * (2 - cc) * muEff) * sqrt(C[j]) * zmean[j]
                var cmuTerm = 0f
                for (r in 0 until mu) { val d = sqrt(C[j]) * z[order[r]][j]; cmuTerm += w[r] * d * d }
                C[j] = ((1 - c1 - cmu) * C[j] + c1 * pc[j] * pc[j] + cmu * cmuTerm).coerceAtLeast(1e-8f)
            }
        }
        return bestX
    }

    // ---- public entry ------------------------------------------------------

    class Result(val style: JSONObject, val startCos: Float, val endCos: Float, val evals: Int)

    /** Polish [seedStyleFile] (a style JSON) against [refWav] and return the
     *  improved style JSON plus before/after cosine. [onProgress] is 0..1. */
    fun refine(refWav: File, seedStyleFile: File, backend: String = "cpu",
               iters: Int = 25, pop: Int = 8,
               onProgress: (Float) -> Unit = {}, alive: () -> Boolean = { true }): Result? {
        if (!load(backend)) return null
        target = run {
            val pcm = wavToMono16k(refWav.readBytes()) ?: return null
            embedPcm(pcm) ?: return null
        }
        val seed = SupertonicEngine.loadStyle(seedStyleFile) ?: return null
        val x0 = encode(seed)
        val startCos = cosineOf(x0)

        var evalCount = 0
        val objective: (FloatArray) -> Float = { c ->
            evalCount++
            var l2 = 0f; for (v in c) l2 += v * v
            -(cosineOf(c) - 0.02f * l2 / c.size)      // maximise cosine, gentle pull to the manifold
        }
        val best = sepCma(x0, 0.35f, iters, pop, objective, onProgress, alive)
        val endCos = cosineOf(best)
        // keep whichever actually scored better — a search can wander
        val chosen = if (endCos >= startCos) decode(best) else decode(x0)
        return Result(styleJson(chosen, refWav.name, maxOf(startCos, endCos)),
            startCos, endCos, evalCount)
    }

    private fun embedPcm(pcm: FloatArray): FloatArray? {
        val e = env ?: return null
        val t = OnnxTensor.createTensor(e, FloatBuffer.wrap(pcm), longArrayOf(1, pcm.size.toLong()))
        val out = spk!!.run(mapOf("wav" to t))
        val emb = flatten(out[0].value); out.close(); t.close()
        return emb
    }

    private fun styleJson(s: SupertonicEngine.Style, ref: String, cos: Float): JSONObject {
        fun arr(m: Array<Array<FloatArray>>, r: Int, c: Int): Pair<JSONArray, JSONArray> {
            val data = JSONArray(); for (row in m[0]) for (v in row) data.put(v.toDouble())
            return JSONArray(listOf(1, r, c)) to data
        }
        val (td, tdata) = arr(s.ttl, ttlR, ttlC)
        val (dd, ddata) = arr(s.dp, dpR, dpC)
        return JSONObject()
            .put("style_ttl", JSONObject().put("dims", td).put("data", tdata))
            .put("style_dp", JSONObject().put("dims", dd).put("data", ddata))
            .put("metadata", JSONObject()
                .put("source", "on-device refine (CMA polish)")
                .put("reference", ref)
                .put("held_out_cos", cos.toDouble()))
    }

    // ---- small audio helpers ----------------------------------------------

    /** 16-bit PCM WAV bytes -> mono float at 16 kHz (linear resample). */
    private fun wavToMono16k(b: ByteArray): FloatArray? {
        if (b.size < 44 || String(b, 0, 4) != "RIFF") return null
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        var pos = 12; var rate = 0; var channels = 1; var bits = 16
        var off = -1; var len = 0
        while (pos + 8 <= b.size) {
            val id = String(b, pos, 4); val size = bb.getInt(pos + 4)
            when (id) {
                "fmt " -> { channels = bb.getShort(pos + 10).toInt(); rate = bb.getInt(pos + 12); bits = bb.getShort(pos + 22).toInt() }
                "data" -> { off = pos + 8; len = size }
            }
            if (off > 0) break
            pos += 8 + size + (size and 1)
        }
        if (off < 0 || rate <= 0 || bits != 16) return null
        val frames = minOf(len, b.size - off) / 2 / channels
        val mono = FloatArray(frames)
        for (i in 0 until frames) {
            var acc = 0f
            for (c in 0 until channels) acc += bb.getShort(off + (i * channels + c) * 2) / 32768f
            mono[i] = acc / channels
        }
        if (rate == RATE) return mono
        val outN = (frames.toLong() * RATE / rate).toInt()
        if (outN <= 1) return null
        return FloatArray(outN) { i ->
            val xf = i.toDouble() * (frames - 1) / (outN - 1)
            val i0 = xf.toInt().coerceAtMost(frames - 1); val i1 = (i0 + 1).coerceAtMost(frames - 1)
            val tt = (xf - i0).toFloat(); mono[i0] * (1 - tt) + mono[i1] * tt
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun flatten(v: Any?): FloatArray = when (v) {
        is FloatArray -> v
        is Array<*> -> { val out = ArrayList<Float>(); fun w(x: Any?) { when (x) { is FloatArray -> x.forEach { out.add(it) }; is Array<*> -> x.forEach { w(it) } } }; w(v); out.toFloatArray() }
        else -> FloatArray(0)
    }
}
