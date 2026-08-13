package com.techhurts.ttsrunner

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/** Runs "Refine this voice" as a background job.
 *
 *  Refining is a long sustained-CPU search (hundreds of synthesis passes), so
 *  it cannot live on an Activity thread that dies when the screen goes off. This
 *  is a foreground service with the same survival kit the save-jobs use: a
 *  wake lock, and a silent AudioTrack so Samsung does not exile the process to
 *  the little cores mid-search. Progress and the result arrive as a
 *  notification and a broadcast, so the user can leave the app entirely. */
class RefineService : Service() {

    companion object {
        const val ACTION_START = "com.techhurts.ttsrunner.REFINE_START"
        const val ACTION_STOP = "com.techhurts.ttsrunner.REFINE_STOP"
        const val EXTRA_REF = "ref"      // reference recording (wav) path
        const val EXTRA_SEED = "seed"    // seed style (json) path
        const val EXTRA_NAME = "name"    // speaker name being refined
        const val STATUS = "com.techhurts.ttsrunner.REFINE_STATUS"

        // A wider 384-component basis reaches 0.8+ (matching the desktop
        // reference) but needs a bigger search — the desktop probe used ~3600
        // evaluations at k=384. The refine synthesises each generation's
        // population as one batched pass at reduced fidelity (short probe,
        // 4 flow steps, frozen duration, shared noise) and stops early on
        // plateau, so this budget costs a fraction of the old sequential
        // full-fidelity run (~8× per evaluation measured on desktop).
        const val ITERS = 120
        const val POP = 20

        private const val CHANNEL = "refine"
        private const val NOTIF_ID = 7

        @Volatile var running = false
        @Volatile var lastPermille = 0
        @Volatile var refiningName = ""
    }

    private val cancel = AtomicBoolean(false)
    private var worker: Thread? = null
    private var keepalive: Thread? = null
    private var wake: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) { cancel.set(true); return START_NOT_STICKY }
        val name = intent?.getStringExtra(EXTRA_NAME) ?: "voice"
        // Started via startForegroundService: this call must reach startForeground
        // within a few seconds even when it turns out to be a no-op, or the OS
        // throws. So go foreground first, then decide whether there is work.
        ensureChannel()
        startForegroundCompat(notif("Refining “$name”", "Starting…", 0))

        val ref = intent?.getStringExtra(EXTRA_REF)?.let { File(it) }
        val seed = intent?.getStringExtra(EXTRA_SEED)?.let { File(it) }
        if (ref == null || seed == null || !ref.exists() || !seed.exists()) {
            broadcast("error", 0, "The reference recording for this voice is gone")
            cleanup(); return START_NOT_STICKY
        }
        // Already grinding on another voice: leave it running (it is still the
        // foreground job; its next progress tick repaints the notification).
        if (running) { toast("A refine is already running"); return START_NOT_STICKY }

        running = true; refiningName = name; lastPermille = 0; cancel.set(false)
        startedAt = System.currentTimeMillis()
        wake = getSystemService(PowerManager::class.java)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "ttsrunner:refine")
            .apply { setReferenceCounted(false); acquire(90 * 60_000L) }
        keepalive = startSilentKeepalive()

        worker = thread(name = "refine") {
            val engine = RefineEngine(this)
            val result = runCatching {
                engine.refine(ref, seed,
                    backend = intent?.getStringExtra("backend") ?: "cpu",
                    iters = intent?.getIntExtra("iters", ITERS) ?: ITERS,
                    pop = intent?.getIntExtra("pop", POP) ?: POP,
                    onProgress = { f ->
                        val pm = (f * 1000).toInt().coerceIn(0, 1000)
                        if (pm - lastPermille >= 10) {   // throttle notification churn
                            lastPermille = pm
                            update(notif("Refining “$name”", "${pm / 10}% · ~${etaText(f)} left", pm))
                            broadcast("progress", pm, "")
                        }
                    },
                    alive = { !cancel.get() })
            }.getOrNull()
            engine.close()
            finish(name, ref.name, result)
        }
        return START_NOT_STICKY
    }

    private var startedAt = System.currentTimeMillis()

    private fun etaText(frac: Float): String {
        val elapsed = System.currentTimeMillis() - startedAt
        if (frac < 0.02f) return "a while"
        val totalMs = (elapsed / frac).toLong()
        val leftS = ((totalMs - elapsed) / 1000).toInt().coerceAtLeast(0)
        return if (leftS >= 60) "${leftS / 60}m" else "${leftS}s"
    }

    private fun finish(name: String, refName: String, result: RefineEngine.Result?) {
        val nm = getSystemService(NotificationManager::class.java)
        when {
            cancel.get() -> { broadcast("stopped", 0, ""); nm.cancel(NOTIF_ID) }
            result == null -> {
                broadcast("error", 0, "Refine failed")
                nm.notify(NOTIF_ID, doneNotif("Refine failed", "“$name” — see Settings → Copy log"))
            }
            else -> {
                val outName = runCatching {
                    val tmp = File(cacheDir, "refined-${System.currentTimeMillis()}.json")
                        .apply { writeText(result.style.toString()) }
                    VoiceStore.importStyle(this, tmp, "$name (refined)").name
                }.getOrDefault("$name (refined)")
                val msg = "Similarity to $refName: ${"%.2f".format(result.startCos)} → " +
                    "${"%.2f".format(result.endCos)} over ${result.evals} tries. Saved as “$outName”."
                broadcast("done", 1000, msg)
                nm.notify(NOTIF_ID, doneNotif("Refined “$name”", msg))
            }
        }
        cleanup()
    }

    private fun cleanup() {
        running = false; refiningName = ""
        keepalive?.interrupt(); keepalive = null
        runCatching { wake?.release() }; wake = null
        if (Build.VERSION.SDK_INT >= 24) stopForeground(STOP_FOREGROUND_DETACH) else @Suppress("DEPRECATION") stopForeground(false)
        stopSelf()
    }

    override fun onDestroy() {
        super.onDestroy()
        cancel.set(true)
        keepalive?.interrupt()
        runCatching { wake?.release() }
    }

    // ---- notification ------------------------------------------------------

    private fun ensureChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL, "Voice refining", NotificationManager.IMPORTANCE_LOW))
    }

    private fun openApp(): PendingIntent = PendingIntent.getActivity(
        this, 0, Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
        PendingIntent.FLAG_IMMUTABLE)

    private fun notif(title: String, text: String, permille: Int): Notification {
        val stop = PendingIntent.getService(this, 2,
            Intent(this, RefineService::class.java).setAction(ACTION_STOP), PendingIntent.FLAG_IMMUTABLE)
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title).setContentText(text)
            .setContentIntent(openApp())
            .setOnlyAlertOnce(true).setOngoing(true)
            .setProgress(1000, permille, permille <= 0)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }

    private fun doneNotif(title: String, text: String): Notification =
        Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title).setContentText(text)
            .setStyle(Notification.BigTextStyle().bigText(text))
            .setContentIntent(openApp())
            .setAutoCancel(true).build()

    private fun update(n: Notification) =
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, n)

    private fun startForegroundCompat(n: Notification) {
        if (Build.VERSION.SDK_INT >= 29)
            startForeground(NOTIF_ID, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PLAYBACK)
        else startForeground(NOTIF_ID, n)
    }

    private fun broadcast(state: String, permille: Int, message: String) {
        sendBroadcast(Intent(STATUS).setPackage(packageName)
            .putExtra("state", state).putExtra("permille", permille).putExtra("message", message))
    }

    private fun toast(t: String) = android.os.Handler(mainLooper).post {
        android.widget.Toast.makeText(this, t, android.widget.Toast.LENGTH_SHORT).show()
    }

    /** Silent audio keeps the process on the big cores while it grinds — the
     *  same trick the save-jobs use against Samsung's background CPU jail. */
    private fun startSilentKeepalive(): Thread = thread(name = "refine-keepalive") {
        val rate = 8000
        val minBuf = AudioTrack.getMinBufferSize(rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val track = AudioTrack.Builder()
            .setAudioAttributes(AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH).build())
            .setAudioFormat(AudioFormat.Builder()
                .setSampleRate(rate).setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
            .setBufferSizeInBytes(maxOf(minBuf, rate / 2))
            .setTransferMode(AudioTrack.MODE_STREAM).build()
        val silence = ByteArray(rate / 2)
        try {
            track.play()
            while (!Thread.currentThread().isInterrupted) track.write(silence, 0, silence.size)
        } catch (_: InterruptedException) {
        } catch (_: Throwable) {
        } finally { runCatching { track.stop() }; track.release() }
    }
}
