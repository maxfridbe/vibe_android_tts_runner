package com.techhurts.ttsrunner

import android.content.Context

/** Which compute backend to run on.
 *
 *  The two engines have nothing in common here: the Qwen models run through
 *  llama.cpp (CPU / OpenCL / Vulkan) and Supertonic runs through ONNX Runtime
 *  (CPU / NNAPI / XNNPACK). Offering one list for both was misleading — a
 *  "Vulkan" pick meant nothing to Supertonic — so the choice is per engine and
 *  so is the stored preference.
 *
 *  The recommendation is measured, not assumed. See the notes on each option. */
object Backends {

    data class Option(val id: String, val label: String, val why: String)

    private val LLAMA = listOf(
        Option("cpu", "CPU", "always works; the big cores only"),
        Option("opencl", "GPU · OpenCL", "Adreno with the Q4_0 model: ~1.7× the talker speed"),
        Option("vulkan", "GPU · Vulkan", "any quant, but Adreno drivers fail to compile the shaders"),
    )

    private val ONNX = listOf(
        Option("cpu", "CPU", "measured fastest on Xclipse: 2.71 s where XNNPACK took 5.65 s"),
        Option("nnapi", "GPU / NPU · NNAPI", "hands layers to the vendor driver; on Xclipse it placed none"),
        Option("xnnpack", "CPU · XNNPACK", "ORT's optimised kernels — slower on these graphs, kept for comparison"),
    )

    fun options(engine: String): List<Option> = if (engine == "supertonic") ONNX else LLAMA

    private fun key(engine: String) = if (engine == "supertonic") "backend_onnx" else "backend"

    fun current(ctx: Context, engine: String): String {
        val p = ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
        val stored = p.getString(key(engine), null)
        // "gpu" is the old name for the OpenCL option
        val id = if (stored == "gpu") "opencl" else stored
        return id?.takeIf { s -> options(engine).any { it.id == s } } ?: "cpu"
    }

    /** Backend for a model whose engine we do not have in hand — the compute
     *  preference is stored per engine (llama vs onnx), so this reads the
     *  llama.cpp side, which is the only one with a GPU/CPU choice worth
     *  remembering globally. Callers with a speaker in hand pass its engine. */
    fun current(ctx: Context): String = current(ctx, "llama")

    fun set(ctx: Context, engine: String, id: String) {
        ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
            .edit().putString(key(engine), id).apply()
    }

    /** @param info output of TtsEngine.nDeviceInfo()
     *  @return the option id to star, and why it is starred */
    fun recommend(engine: String, info: String, gpuCapable: Boolean): Pair<String, String> {
        val adrenoCl = info.contains("OpenCL") && info.contains("Adreno")
        val vulkanLine = info.lines().find { it.contains("Vulkan") } ?: ""
        val vulkanNonAdreno = vulkanLine.isNotEmpty() && !vulkanLine.contains("Adreno")
        if (engine == "supertonic") {
            // 99M parameters of ONNX: the CPU path is not the fallback here,
            // it is the fast one. NNAPI is worth trying on Adreno, where the
            // driver actually places graphs.
            return "cpu" to if (adrenoCl)
                "Supertonic is small enough that CPU wins; NNAPI is worth a try on your Adreno driver"
            else "Supertonic runs faster on CPU than on this phone's accelerators"
        }
        return when {
            adrenoCl && gpuCapable -> "opencl" to "your Adreno GPU has tuned OpenCL kernels for this quant"
            adrenoCl && !gpuCapable -> "cpu" to "OpenCL only helps the Q4_0 model; this one would be slower on the GPU"
            vulkanNonAdreno -> "vulkan" to "Vulkan compute is available and this GPU is not an Adreno"
            else -> "cpu" to "no usable GPU compute was detected"
        }
    }

    /** "Xclipse 940" / "Adreno" / "none", pulled out of the device report. */
    fun gpuName(info: String): String {
        val vulkanLine = info.lines().find { it.contains("Vulkan") } ?: ""
        return Regex("— ([^(]+)\\(").find(vulkanLine)?.groupValues?.get(1)?.trim()
            ?: if (info.contains("Adreno")) "Adreno" else "none"
    }
}
