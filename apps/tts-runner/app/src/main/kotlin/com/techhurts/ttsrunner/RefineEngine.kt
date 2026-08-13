package com.techhurts.ttsrunner

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
 *  synthesise a short line with Supertonic, embed it, score its cosine to
 *  the reference recording — forward passes the app already runs. The judge
 *  is the qwen speaker encoder with centered cosine when its files are
 *  staged (the ear-aligned metric; desktop measured mean 0.840 over 18
 *  voices), ECAPA otherwise.
 *  The search runs at reduced fidelity (short probe, 4 flow steps, each
 *  generation synthesised as one batched pass with a frozen duration and
 *  shared noise) and stops early when it plateaus; only the seed and the
 *  final winner are scored at full fidelity. Same loop as the desktop
 *  `clonevoice --refine`, where these levers measured ~8× per evaluation
 *  (see supertonic_voice_cloner/method.md, "Making it faster").
 *
 *  Needs the models the cloner downloads (spk_encoder.onnx + style_basis.bin);
 *  runs on the caller's thread and reports progress by fraction. */
class RefineEngine(private val ctx: Context) {

    companion object {
        const val BASIS_ASSET = "style_basis.bin"
        // Full-fidelity probe (disjoint from what any head trained or was
        // scored on) — used only for the seed and final-winner scores, so the
        // reported start/end cosines stay comparable to older runs.
        private const val PROBE = "A cup of coffee on the desk had long since gone cold."
        // Search-time probe. Phonetically rich (Rainbow Passage opener) —
        // measured on a 20-voice sweep it beats a short probe by +0.061 mean
        // held-out cosine (19/20 voices, up to +0.177): broad vowel/consonant
        // coverage gives ECAPA more of the timbre spectrum to match per
        // evaluation, worth more than its extra synthesis length.
        private const val PROBE_SEARCH =
            "When sunlight strikes raindrops in the air, they act as a prism and form a rainbow."
        // Flow steps during search vs. final scoring. Fewer steps depress every
        // cosine uniformly, which is harmless for ranking — CMA only needs the
        // relative order — and the absolute numbers come from full-fidelity
        // scoring above.
        private const val SEARCH_STEPS = 4
        private const val FINAL_STEPS = 8
        // Stop when the best cosine hasn't improved for this many generations.
        // (15 measurably cut hard voices short in the sweep; 25 kept quality.)
        private const val PATIENCE = 25
        private const val RATE = 16000       // ECAPA front-end
        private const val QRATE = 24000      // qwen front-end
        // Population mean of qwen speaker features: raw qwen features share a
        // dominant common component (different speakers cosine at ~0.94), so
        // similarity is measured after subtracting this center — the metric
        // the whole qwen-first program optimises (d' 4.3 on ref pairs).
        const val QCENTER_ASSET = "qwen_center.bin"

        fun available(ctx: Context): Boolean =
            File(VoiceCloner.dir(ctx), BASIS_ASSET).length() > 0 &&
                VoiceCloner.variants(ctx).isNotEmpty()
    }

    // ---- basis (style_basis.bin) ------------------------------------------
    private var k = 0; private var dTot = 0; private var split = 0
    private var ttlR = 0; private var ttlC = 0; private var dpR = 0; private var dpC = 0
    private lateinit var scale: FloatArray      // (k)
    private lateinit var mean: FloatArray       // (dTot)
    private lateinit var basis: FloatArray       // (k * dTot) row-major

    private var env: OrtEnvironment? = null
    private var spk: OrtSession? = null
    private var qcenter: FloatArray? = null   // non-null => qwen-centered scoring
    private var scoreRate = RATE
    var analyzer = "ecapa"; private set
    private var supertonic: SupertonicEngine? = null

    // The whole population is synthesised as ONE batched ORT run per
    // generation (the graphs have dynamic batch dims), so the sessions get the
    // full intra-op pool — batching replaces the old two-candidates-in-flight
    // executor, and the batched GEMMs use the cores far better than
    // overlapping serial runs did.
    private val cores = Runtime.getRuntime().availableProcessors()
    private val evalThreads = cores.coerceIn(2, 6)

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
                setIntraOpNumThreads(evalThreads)   // small: parallelism is across candidates
            }
            // Scorer: the qwen speaker encoder + center vector when staged
            // (the ear-aligned referee); ECAPA otherwise.
            val qspkF = File(VoiceCloner.dir(ctx), VoiceCloner.QSPK_ASSET)
            val qcenF = File(VoiceCloner.dir(ctx), QCENTER_ASSET)
            if (qspkF.exists() && qcenF.length() >= 2048L * 4) {
                spk = e.createSession(qspkF.readBytes(), opts)
                val bb = ByteBuffer.wrap(qcenF.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
                qcenter = FloatArray(2048) { bb.float }
                scoreRate = QRATE
                analyzer = "qwen-centered"
            } else {
                val spkBytes = File(VoiceCloner.dir(ctx), VoiceCloner.SPK_ASSET).let {
                    if (it.exists()) it.readBytes() else ctx.assets.open(VoiceCloner.SPK_ASSET).use { s -> s.readBytes() }
                }
                spk = e.createSession(spkBytes, opts)
                qcenter = null; scoreRate = RATE; analyzer = "ecapa"
            }
            DebugLog.log(ctx, "RefineEngine", "scoring analyzer: $analyzer")
            supertonic = SupertonicEngine(ctx).also {
                // dynamic sessions stay on CPU (accelerators refuse their
                // dynamic shapes anyway); the static refine graphs, if staged
                // under supertonic-3/static, get the requested backend
                if (!it.load(ModelManager.supertonicDir(ctx), "cpu", intraThreads = evalThreads)) return false
                if (backend != "cpu")
                    it.loadStatic(ModelManager.supertonicDir(ctx), backend, intraThreads = evalThreads)
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

    /** coeffs -> just the flat, row-normalised ttl half — all the batched
     *  search synthesis needs (the dp half only sets the duration, which the
     *  search freezes from the seed). */
    private fun decodeTtlFlat(c: FloatArray): FloatArray {
        val flat = FloatArray(split)
        System.arraycopy(mean, 0, flat, 0, split)
        for (i in 0 until k) {
            val cs = c[i] * scale[i]
            if (cs == 0f) continue
            val base = i * dTot
            for (j in 0 until split) flat[j] += cs * basis[base + j]
        }
        for (r in 0 until ttlR) {
            var n = 0f
            for (col in 0 until ttlC) { val v = flat[r * ttlC + col]; n += v * v }
            n = sqrt(n) + 1e-8f
            for (col in 0 until ttlC) flat[r * ttlC + col] /= n
        }
        return flat
    }

    // ---- one evaluation: coeffs -> cosine to the target --------------------

    private lateinit var target: FloatArray     // reference ECAPA embedding

    /** Full-fidelity cosine (long probe, all flow steps, fresh duration) —
     *  seed and final winner only, so start/end numbers stay comparable to
     *  runs from before the batched search. */
    private fun cosineOf(c: FloatArray): Float {
        val style = decode(c)
        val pcmBytes = supertonic?.generate(PROBE, style, steps = FINAL_STEPS) ?: return -1f
        val srcRate = supertonic?.sampleRate ?: 44100
        val pcm = pcm16ToMono(pcmBytes, srcRate, scoreRate) ?: return -1f
        return score(pcm)
    }

    /** Embed at the active analyzer's rate; qwen features are centered (the
     *  raw space shares a dominant common component) and re-normalised so
     *  cosine spans the full range. ECAPA comes out unit-norm already. */
    private fun embedScore(pcm: FloatArray): FloatArray? {
        val emb = embedPcm(pcm) ?: return null
        val c = qcenter ?: return emb
        var n = 0f
        for (i in emb.indices) { emb[i] -= c[i]; n += emb[i] * emb[i] }
        n = sqrt(n) + 1e-8f
        for (i in emb.indices) emb[i] /= n
        return emb
    }

    private fun score(pcm: FloatArray): Float {
        val emb = embedScore(pcm) ?: return -1f
        var dot = 0f
        for (i in emb.indices) dot += emb[i] * target[i]
        return dot
    }

    /** Float wav at [srcRate] -> [dstRate] (linear), for the batched search path. */
    private fun floatToMono(wav: FloatArray, srcRate: Int, dstRate: Int): FloatArray? {
        if (wav.size < 2) return null
        if (srcRate == dstRate) return wav
        val outN = (wav.size.toLong() * dstRate / srcRate).toInt()
        if (outN <= 1) return null
        return FloatArray(outN) { i ->
            val xf = i.toDouble() * (wav.size - 1) / (outN - 1)
            val i0 = xf.toInt().coerceAtMost(wav.size - 1); val i1 = (i0 + 1).coerceAtMost(wav.size - 1)
            val tt = (xf - i0).toFloat(); wav[i0] * (1 - tt) + wav[i1] * tt
        }
    }

    /** Raw 16-bit LE mono PCM at [srcRate] -> float at [dstRate]. */
    private fun pcm16ToMono(b: ByteArray, srcRate: Int, dstRate: Int): FloatArray? {
        val frames = b.size / 2
        if (frames < 2) return null
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        val mono = FloatArray(frames) { bb.getShort(it * 2) / 32768f }
        return floatToMono(mono, srcRate, dstRate)
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

    /** Minimise f, which scores a whole population per call (one batched
     *  synthesis). Returns the best coeffs found. Reports progress 0..1, stops
     *  after [PATIENCE] generations without improvement, and bails early when
     *  [alive] turns false (user cancelled). */
    private fun sepCma(x0: FloatArray, sigma0: Float, iters: Int, pop: Int,
                       f: (Array<FloatArray>) -> FloatArray, onProgress: (Float) -> Unit,
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
        var bestX = x0.copyOf(); var bestF = f(arrayOf(x0.copyOf()))[0]
        var done = 1
        var lastGain = 0
        val total = 1 + iters * pop
        for (g in 0 until iters) {
            if (!alive()) break
            val z = Array(pop) { FloatArray(n) { rng.gauss() } }
            val x = Array(pop) { p -> FloatArray(n) { j -> (m[j] + sigma * sqrt(C[j]) * z[p][j]).coerceIn(-3f, 3f) } }
            // the whole generation is one batched synthesis + pop embeds
            val scores = f(x)
            done += pop; onProgress(done.toFloat() / total)
            val order = (0 until pop).sortedBy { scores[it] }
            if (scores[order[0]] < bestF - 1e-4f) lastGain = g
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
            if (g - lastGain >= PATIENCE) {
                DebugLog.log(ctx, "RefineEngine", "no improvement in $PATIENCE generations; stopping early at gen ${g + 1}/$iters")
                break
            }
        }
        return bestX
    }

    // ---- public entry ------------------------------------------------------

    class Result(val style: JSONObject, val startCos: Float, val endCos: Float, val evals: Int)

    /** Polish [seedStyleFile] (a style JSON) against [refWav] and return the
     *  improved style JSON plus before/after cosine. [onProgress] is 0..1. */
    fun refine(refWav: File, seedStyleFile: File, backend: String = "cpu",
               iters: Int = 20, pop: Int = 8,
               onProgress: (Float) -> Unit = {}, alive: () -> Boolean = { true }): Result? {
        if (!load(backend)) return null
        target = run {
            val pcm = wavToMono(refWav.readBytes(), scoreRate) ?: return null
            embedScore(pcm) ?: return null
        }
        val seed = SupertonicEngine.loadStyle(seedStyleFile) ?: return null
        val x0 = encode(seed)
        val startCos = cosineOf(x0)

        // Freeze the search probe's duration from the encoder seed: style_dp
        // only sets the duration scalar, and a fixed latent length is what lets
        // a whole generation run as one batch and share one noise tensor
        // (common random numbers — a deterministic objective).
        val st = supertonic ?: return null
        val searchIds = st.probeIds(PROBE_SEARCH) ?: return null
        val dur = st.probeDuration(searchIds, seed.dp) ?: return null
        val wavLen = (dur * st.sampleRate).toInt()
        val latentLen = ((wavLen + st.latentChunk - 1) / st.latentChunk).coerceAtLeast(1)
        val noiseRng = Rng(1234567)
        val noise = FloatArray(st.latentWidth * latentLen) { noiseRng.gauss() }

        val evalCount = java.util.concurrent.atomic.AtomicInteger(0)
        var batched = true
        val objective: (Array<FloatArray>) -> FloatArray = { xs ->
            evalCount.addAndGet(xs.size)
            val ttls = Array(xs.size) { decodeTtlFlat(xs[it]) }
            var wavs = if (batched)
                st.generateBatch(searchIds, ttls, ttlR, ttlC, latentLen, wavLen, noise, SEARCH_STEPS)
            else null
            if (wavs == null) {
                if (batched) { batched = false
                    DebugLog.log(ctx, "RefineEngine", "batched synthesis rejected; per-candidate fallback") }
                wavs = Array(xs.size) { p ->
                    st.generateBatch(searchIds, arrayOf(ttls[p]), ttlR, ttlC,
                        latentLen, wavLen, noise, SEARCH_STEPS)?.get(0) ?: FloatArray(0)
                }
            }
            FloatArray(xs.size) { p ->
                val w = if (wavs[p].isEmpty()) null else floatToMono(wavs[p], st.sampleRate, scoreRate)
                val cos = if (w == null) -1f else score(w)
                var l2 = 0f; for (v in xs[p]) l2 += v * v
                -(cos - 0.02f * l2 / xs[p].size)      // maximise cosine, gentle pull to the manifold
            }
        }
        val best = sepCma(x0, 0.35f, iters, pop, objective, onProgress, alive)
        val endCos = cosineOf(best)
        // keep whichever actually scored better — a search can wander
        val chosen = if (endCos >= startCos) decode(best) else decode(x0)
        return Result(styleJson(chosen, refWav.name, startCos, endCos, evalCount.get(), iters, pop),
            startCos, endCos, evalCount.get())
    }

    private fun embedPcm(pcm: FloatArray): FloatArray? {
        val e = env ?: return null
        val t = OnnxTensor.createTensor(e, FloatBuffer.wrap(pcm), longArrayOf(1, pcm.size.toLong()))
        val out = spk!!.run(mapOf("wav" to t))
        val emb = flatten(out[0].value); out.close(); t.close()
        return emb
    }

    /** Style JSON whose metadata fully explains the clone: which analyzer
     *  judged it, the objective, before/after scores, search budget, basis,
     *  reference, app version and when. */
    private fun styleJson(s: SupertonicEngine.Style, ref: String, startCos: Float,
                          endCos: Float, evals: Int, iters: Int, pop: Int): JSONObject {
        fun arr(m: Array<Array<FloatArray>>, r: Int, c: Int): Pair<JSONArray, JSONArray> {
            val data = JSONArray(); for (row in m[0]) for (v in row) data.put(v.toDouble())
            return JSONArray(listOf(1, r, c)) to data
        }
        val (td, tdata) = arr(s.ttl, ttlR, ttlC)
        val (dd, ddata) = arr(s.dp, dpR, dpC)
        val ver = runCatching {
            ctx.packageManager.getPackageInfo(ctx.packageName, 0).versionName
        }.getOrNull() ?: "?"
        return JSONObject()
            .put("style_ttl", JSONObject().put("dims", td).put("data", tdata))
            .put("style_dp", JSONObject().put("dims", dd).put("data", ddata))
            .put("metadata", JSONObject()
                .put("source", "on-device refine (sep-CMA over the k=$k style basis)")
                .put("analyzer", analyzer)
                .put("objective", if (qcenter != null)
                    "centered qwen cosine (population-mean centered, unit-norm)"
                    else "ecapa cosine")
                .put("reference", ref)
                .put("start_cos", startCos.toDouble())
                .put("end_cos", endCos.toDouble())
                .put("held_out_cos", maxOf(startCos, endCos).toDouble())
                .put("evals", evals)
                .put("budget", "${iters}x${pop}, patience $PATIENCE, search probe: rich")
                .put("app_version", ver)
                .put("created", java.text.SimpleDateFormat("yyyy-MM-dd HH:mm",
                    java.util.Locale.US).format(java.util.Date())))
    }

    // ---- small audio helpers ----------------------------------------------

    /** 16-bit PCM WAV bytes -> mono float at [dstRate] (linear resample). */
    private fun wavToMono(b: ByteArray, dstRate: Int): FloatArray? {
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
        return floatToMono(mono, rate, dstRate)
    }

    @Suppress("UNCHECKED_CAST")
    private fun flatten(v: Any?): FloatArray = when (v) {
        is FloatArray -> v
        is Array<*> -> { val out = ArrayList<Float>(); fun w(x: Any?) { when (x) { is FloatArray -> x.forEach { out.add(it) }; is Array<*> -> x.forEach { w(it) } } }; w(v); out.toFloatArray() }
        else -> FloatArray(0)
    }
}
