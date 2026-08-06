package com.maxfridbe.ttsrunner

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Append-only debug log in filesDir, written from both the UI and :engine
 *  processes (same filesDir, append mode). Assembled together with device
 *  info and a logcat dump by "Copy debug log" so failures can be pasted back
 *  verbatim. */
object DebugLog {
    private const val MAX_BYTES = 512 * 1024L
    private val fmt = SimpleDateFormat("MM-dd HH:mm:ss.SSS", Locale.US)

    private fun file(ctx: Context) = File(ctx.filesDir, "debug.log")

    @Synchronized
    fun log(ctx: Context, tag: String, msg: String) {
        android.util.Log.i(tag, msg)
        try {
            val f = file(ctx)
            if (f.length() > MAX_BYTES) {
                val tail = f.readText().takeLast((MAX_BYTES / 2).toInt())
                f.writeText(tail)
            }
            val proc = if (isEngineProcess(ctx)) "engine" else "ui"
            f.appendText("${fmt.format(Date())} [$proc/$tag] $msg\n")
        } catch (_: Exception) {
        }
    }

    fun log(ctx: Context, tag: String, msg: String, t: Throwable) {
        log(ctx, tag, "$msg: ${t.javaClass.simpleName}: ${t.message}\n${t.stackTraceToString().take(2000)}")
    }

    private fun isEngineProcess(ctx: Context): Boolean {
        val am = ctx.getSystemService(ActivityManager::class.java)
        val pid = android.os.Process.myPid()
        return am.runningAppProcesses?.find { it.pid == pid }?.processName?.endsWith(":engine") == true
    }

    /** Full report: device facts, memory, CPU features, ggml devices (if the
     *  native lib loads), model/voice inventory, our log, then logcat. */
    fun buildReport(ctx: Context): String {
        val sb = StringBuilder()
        sb.appendLine("=== TTS Runner debug report ${fmt.format(Date())} ===")
        sb.appendLine("device: ${Build.MANUFACTURER} ${Build.MODEL} (${Build.DEVICE}) android ${Build.VERSION.RELEASE} sdk ${Build.VERSION.SDK_INT}")
        sb.appendLine("soc: ${if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER + " " + Build.SOC_MODEL else "n/a"} abis: ${Build.SUPPORTED_ABIS.joinToString()}")

        val am = ctx.getSystemService(ActivityManager::class.java)
        val mi = ActivityManager.MemoryInfo()
        am.getMemoryInfo(mi)
        sb.appendLine("ram: avail ${mi.availMem / 1048576} MB / total ${mi.totalMem / 1048576} MB, lowMemory=${mi.lowMemory} threshold ${mi.threshold / 1048576} MB")
        sb.appendLine("app memoryClass: ${am.memoryClass} MB / large ${am.largeMemoryClass} MB")

        try {
            val feat = File("/proc/cpuinfo").readLines().firstOrNull { it.startsWith("Features") }
            sb.appendLine("cpu ${Runtime.getRuntime().availableProcessors()} cores, $feat")
        } catch (e: Exception) {
            sb.appendLine("cpuinfo unreadable: $e")
        }

        try {
            sb.appendLine("--- ggml devices ---")
            sb.append(TtsEngine.nDeviceInfo())
        } catch (t: Throwable) {
            sb.appendLine("nDeviceInfo failed: $t")
        }

        sb.appendLine("--- models dir ---")
        ModelManager.modelsDir(ctx).listFiles()?.forEach { sb.appendLine("${it.name} ${it.length()} bytes") }
        sb.appendLine("--- voices ---")
        VoiceStore.list(ctx).forEach { sb.appendLine("${it.file.name} ${it.file.length()} bytes") }
        val prefs = ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
        sb.appendLine("prefs: backend=${prefs.getString("backend", "cpu")} model=${prefs.getString("model_id", null)} voice=${prefs.getString("default_voice", null)}")

        sb.appendLine("--- app log (debug.log) ---")
        try {
            sb.appendLine(file(ctx).readText().takeLast(64 * 1024))
        } catch (e: Exception) {
            sb.appendLine("no app log: $e")
        }

        sb.appendLine("--- logcat (main) ---")
        sb.appendLine(logcat("main"))
        sb.appendLine("--- logcat (crash) ---")
        sb.appendLine(logcat("crash"))
        return sb.toString()
    }

    private fun logcat(buffer: String): String = try {
        val p = Runtime.getRuntime().exec(arrayOf("logcat", "-d", "-v", "time", "-b", buffer))
        val text = p.inputStream.bufferedReader().readText()
        p.waitFor()
        // our UID only sees its own lines anyway; keep the interesting tags and the tail
        val lines = text.lines().filter { l ->
            listOf("TtsRunnerNative", "llama.cpp", "TtsService", "TtsRunner", "AndroidRuntime",
                "DEBUG", "libc", "lowmemorykiller", "ActivityManager").any { l.contains(it) } || buffer == "crash"
        }
        lines.takeLast(400).joinToString("\n")
    } catch (e: Exception) {
        "logcat failed: $e"
    }
}
