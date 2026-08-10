package com.maxfridbe.ttsrunner

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder

/** Keeps the HTTP server alive while the app is in the background.
 *
 *  A server that dies the moment you switch apps is not a server, so this is a
 *  foreground service with a notification you can stop it from. It runs in the
 *  UI process, not :engine — it hands work to the engine service the same way
 *  the UI does, so a native crash there does not take the server with it. */
class HostingService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(CHANNEL, "Speech API host", NotificationManager.IMPORTANCE_LOW))
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> { stopServer(); stopSelf(); return START_NOT_STICKY }
        }
        val port = intent?.getIntExtra(EXTRA_PORT, DEFAULT_PORT) ?: DEFAULT_PORT
        try {
            if (server?.running != true) {
                server = HttpServer(this, port).apply {
                    // every request refreshes the notification, so the phone
                    // shows what the network is asking it to say
                    onActivity = { runCatching { update() } }
                    start()
                }
                DebugLog.log(this, "HostingService", "listening on :$port")
            }
        } catch (t: Throwable) {
            DebugLog.log(this, "HostingService", "could not bind :$port", t as? Exception ?: Exception(t))
            lastBindError = t.message ?: t.javaClass.simpleName
            sendBroadcast(Intent(STATUS).setPackage(packageName))
            stopSelf()
            return START_NOT_STICKY
        }
        if (Build.VERSION.SDK_INT >= 34) {
            startForeground(NOTIF_ID, notification(), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        } else {
            startForeground(NOTIF_ID, notification())
        }
        sendBroadcast(Intent(STATUS).setPackage(packageName))
        return START_STICKY
    }

    private fun notification(): Notification {
        val s = server
        val url = "http://${HttpServer.lanAddress() ?: "127.0.0.1"}:${s?.port ?: DEFAULT_PORT}"
        val stop = PendingIntent.getService(this, 7,
            Intent(this, HostingService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_IMMUTABLE)
        val open = PendingIntent.getActivity(this, 8,
            Intent(this, MainActivity::class.java), PendingIntent.FLAG_IMMUTABLE)
        val busy = (s?.speaking ?: 0) > 0
        val detail = s?.lastActivity ?: "no requests yet"
        return Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(if (busy) "Speech API — speaking" else "Speech API is up")
            .setContentText("$url · ${s?.requests ?: 0} requests")
            .setStyle(Notification.BigTextStyle().bigText("$url\n$detail"))
            .setContentIntent(open)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .addAction(Notification.Action.Builder(null, "Stop", stop).build())
            .build()
    }

    private fun update() {
        getSystemService(NotificationManager::class.java).notify(NOTIF_ID, notification())
        sendBroadcast(Intent(STATUS).setPackage(packageName))
    }

    private fun stopServer() {
        server?.stop()
        server = null
        sendBroadcast(Intent(STATUS).setPackage(packageName))
    }

    override fun onDestroy() {
        stopServer()
        super.onDestroy()
    }

    companion object {
        const val ACTION_STOP = "com.maxfridbe.ttsrunner.HOST_STOP"
        const val STATUS = "com.maxfridbe.ttsrunner.HOST_STATUS"
        const val EXTRA_PORT = "port"
        const val DEFAULT_PORT = 8080
        private const val CHANNEL = "hosting"
        private const val NOTIF_ID = 2

        /** The one server instance. Held statically because the tab needs to
         *  ask "are you up, and on which port" without binding. */
        @Volatile var server: HttpServer? = null
            private set
        @Volatile var lastBindError: String? = null

        val running: Boolean get() = server?.running == true
        val port: Int get() = server?.port ?: DEFAULT_PORT

        fun start(ctx: Context, port: Int) {
            lastBindError = null
            ctx.startForegroundService(Intent(ctx, HostingService::class.java)
                .putExtra(EXTRA_PORT, port))
        }

        fun stop(ctx: Context) {
            ctx.startService(Intent(ctx, HostingService::class.java).setAction(ACTION_STOP))
        }
    }
}
