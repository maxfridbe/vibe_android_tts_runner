package com.techhurts.ttsrunner

import android.content.Context
import android.os.Build
import android.os.Handler
import android.os.Looper
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/** One guarded probe of the ggml compute devices.
 *
 *  nDeviceInfo() looks innocent but ggml_backend_load_all() initialises every
 *  compiled backend, and the Vulkan path hands the phone's GPU driver a pile
 *  of pipelines to compile. On the S24 FE's Xclipse driver that stalls the
 *  same GPU the UI renders with — the settings screen froze — and the driver
 *  sometimes brought the whole process down from inside, where no try/catch
 *  reaches. The settings tab also re-probed on every rebuild, so two
 *  load_all() calls could race. Hence the three rules here:
 *
 *   - probe once per OS build: the result is cached in prefs keyed by
 *     Build.FINGERPRINT, so a phone re-probes after a system update, not on
 *     every settings visit;
 *   - never probe twice at once, and never after a probe died: a pending
 *     flag is committed before the native call and cleared after it, so a
 *     probe that killed the app is not repeated on the next launch;
 *   - never make the UI wait: callers get a timeout callback and proceed
 *     with the CPU recommendation, and a late answer just fills the cache. */
object DeviceProbe {
    private const val PREFS = "ttsrunner"
    private const val K_INFO = "device_info_cache"
    private const val K_FP = "device_info_fp"
    private const val K_PENDING = "device_info_pending"

    @Volatile private var memo: String? = null
    private val inFlight = AtomicBoolean(false)
    private val waiters = CopyOnWriteArrayList<(String?) -> Unit>()
    private val main = Handler(Looper.getMainLooper())

    /** Cached report for this OS build, or null if it has never been probed. */
    fun cached(ctx: Context): String? {
        memo?.let { return it }
        val p = ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val info = p.getString(K_INFO, null)
        if (info != null && p.getString(K_FP, null) == Build.FINGERPRINT) {
            memo = info
            return info
        }
        return null
    }

    /** True when a previous probe hung or took the process down mid-call. */
    fun crashedBefore(ctx: Context): Boolean =
        cached(ctx) == null && !inFlight.get() &&
            ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(K_PENDING, false)

    /** Drop the crash guard so the next probe() tries the driver again. */
    fun retry(ctx: Context) {
        ctx.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().remove(K_PENDING).apply()
    }

    /** Delivers the device report on the main thread, or null when the probe
     *  is skipped (a previous one died) or has not answered within the
     *  timeout — the caller shows the CPU-safe default either way. */
    fun probe(ctx: Context, timeoutMs: Long = 6000, cb: (String?) -> Unit) {
        cached(ctx)?.let { cb(it); return }
        val app = ctx.applicationContext
        val p = app.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (p.getBoolean(K_PENDING, false) && !inFlight.get()) {
            cb(null)
            return
        }

        // once-only delivery; both paths run on the main thread, so no race
        var delivered = false
        val once: (String?) -> Unit = { r -> if (!delivered) { delivered = true; cb(r) } }
        waiters.add { r -> main.post { once(r) } }
        main.postDelayed({ once(null) }, timeoutMs)

        if (inFlight.compareAndSet(false, true)) {
            thread(name = "gpu-probe", isDaemon = true) {
                // committed, not applied: if the driver kills us microseconds
                // from now, the flag must already be on disk
                p.edit().putBoolean(K_PENDING, true).commit()
                val info = runCatching { TtsEngine.nDeviceInfo() }.getOrDefault("")
                p.edit().putString(K_INFO, info).putString(K_FP, Build.FINGERPRINT)
                    .remove(K_PENDING).commit()
                memo = info
                inFlight.set(false)
                val ws = waiters.toList()
                waiters.clear()
                ws.forEach { it(info) }
            }
        }
    }
}
