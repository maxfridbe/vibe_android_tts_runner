package com.maxfridbe.ttsrunner

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import java.io.File

/** Dead-man switch for save jobs. Android's sticky-service restart revives a
 *  killed engine exactly once on this Samsung — after the next kill the AMS
 *  never reschedules it. So while a save job is unfinished, an alarm re-arms
 *  itself every few minutes; each firing pokes the service, which is a no-op
 *  when the job is alive and an auto-resume when the process was killed.
 *  Runs in the :engine process, so a firing with a dead engine spawns it.
 *  The chain stops when the pending-job file disappears (job finished,
 *  stopped, or gave up after repeated kills at the same chunk). */
class ResumeReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        if (!File(ctx.filesDir, "pending-job.json").exists()) return
        TtsService.armResumeAlarm(ctx)   // keep the chain alive first
        runCatching {
            ctx.startForegroundService(
                Intent(ctx, TtsService::class.java).setAction(TtsService.ACTION_NUDGE))
        }.onFailure { DebugLog.log(ctx, "ResumeReceiver", "nudge failed: $it") }
    }
}
