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

    /** Wipe the app log and the logcat buffers (best-effort). */
    fun clear(ctx: Context) {
        runCatching { file(ctx).delete() }
        runCatching { Runtime.getRuntime().exec(arrayOf("logcat", "-c")).waitFor() }
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
        sb.appendLine("=== TTS Runner v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE}) debug report ${fmt.format(Date())} ===")
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

        sb.appendLine("--- ggml devices ---")
        // cache only: a debug report must never poke the GPU driver itself
        sb.appendLine(DeviceProbe.cached(ctx) ?: "not probed on this OS build" +
            if (DeviceProbe.crashedBefore(ctx)) " (a previous probe crashed)" else "")

        sb.appendLine("--- models dir ---")
        ModelManager.modelsDir(ctx).listFiles()?.forEach { sb.appendLine("${it.name} ${it.length()} bytes") }
        sb.appendLine("--- voices ---")
        VoiceStore.list(ctx).forEach { sb.appendLine("${it.file.name} ${it.file.length()} bytes") }
        val prefs = ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
        sb.appendLine("prefs: backend=${prefs.getString("backend", "cpu")} model=${prefs.getString("model_id", null)} voice=${prefs.getString("default_voice", null)}")

        // process exit reasons are the ground truth for silent deaths (lmkd
        // SIGKILLs leave no tombstone); rss at kill time is the key number
        sb.appendLine("--- recent process exits ---")
        try {
            am.getHistoricalProcessExitReasons(ctx.packageName, 0, 6).forEach { e ->
                sb.appendLine("${fmt.format(Date(e.timestamp))} ${e.processName.substringAfterLast(':', "ui")} " +
                    "reason=${e.reason} (${exitReasonName(e.reason)}) rss=${e.rss / 1024} MB status=${e.status}")
            }
        } catch (e: Exception) {
            sb.appendLine("exit-info unavailable: $e")
        }

        // kept deliberately small so reports paste comfortably
        sb.appendLine("--- app log (debug.log, tail) ---")
        try {
            sb.appendLine(file(ctx).readText().lines().takeLast(60).joinToString("\n"))
        } catch (e: Exception) {
            sb.appendLine("no app log: $e")
        }

        sb.appendLine("--- logcat (main, tail) ---")
        sb.appendLine(logcat("main"))
        sb.appendLine("--- last native crash (condensed) ---")
        sb.appendLine(lastCrash())
        return sb.toString()
    }

    private fun logcat(buffer: String): String = try {
        val p = Runtime.getRuntime().exec(arrayOf("logcat", "-d", "-v", "time", "-b", buffer))
        val text = p.inputStream.bufferedReader().readText()
        p.waitFor()
        // our UID only sees its own lines anyway; keep the interesting tags and the tail
        val lines = text.lines().filter { l ->
            listOf("TtsRunnerNative", "llama.cpp", "TtsService", "TtsRunner", "AndroidRuntime",
                "DEBUG", "lowmemorykiller", "ActivityManager").any { l.contains(it) } || buffer == "crash"
        }
        lines.takeLast(80).joinToString("\n")
    } catch (e: Exception) {
        "logcat failed: $e"
    }

    /** Only the newest crash, and only its interesting lines: signal/abort
     *  header plus the first dozen backtrace frames — a full multi-crash dump
     *  was 3/4 of the old report. */
    private fun lastCrash(): String {
        val raw = logcatRaw("crash")
        val start = raw.lastIndexOf("Fatal signal")
        if (start < 0) return "(none)"
        val lines = raw.substring(start).lines()
        val keep = ArrayList<String>()
        var frames = 0
        for (l in lines) {
            val interesting = l.contains("Fatal signal") || l.contains("Abort message") ||
                l.contains("Cmdline") || l.contains("backtrace:") || l.contains(">>>")
            val isFrame = Regex("#\\d\\d pc").containsMatchIn(l)
            if (isFrame) { if (frames++ >= 14) continue }
            if (interesting || isFrame) keep.add(l)
        }
        return keep.joinToString("\n")
    }

    private fun exitReasonName(r: Int): String = when (r) {
        1 -> "EXIT_SELF"; 2 -> "SIGNALED"; 3 -> "LOW_MEMORY"; 4 -> "APP CRASH"
        5 -> "NATIVE CRASH"; 6 -> "ANR"; 10 -> "USER REQUESTED"; 13 -> "OTHER"
        16 -> "PACKAGE UPDATED"
        else -> "code $r"
    }

    private fun logcatRaw(buffer: String): String = try {
        val p = Runtime.getRuntime().exec(arrayOf("logcat", "-d", "-v", "time", "-b", buffer))
        val text = p.inputStream.bufferedReader().readText()
        p.waitFor()
        text
    } catch (e: Exception) {
        ""
    }
}
