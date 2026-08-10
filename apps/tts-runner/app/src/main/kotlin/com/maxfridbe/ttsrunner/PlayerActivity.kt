package com.maxfridbe.ttsrunner

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.color.DynamicColors
import java.io.File
import kotlin.concurrent.thread

/** Transport for a job that is being read out loud right now.
 *
 *  A shared article starts speaking immediately, which is the point of sharing
 *  it — but with no controls the only way to stop it was the notification.
 *  This is the screen that owns the playback: pause and resume without losing
 *  the generated audio, watch the progress, and keep the result when it turns
 *  out to be worth keeping. */
class PlayerActivity : AppCompatActivity() {

    private lateinit var titleView: TextView
    private lateinit var stateView: TextView
    private lateinit var bar: ProgressBar
    private lateinit var playBtn: Button
    private lateinit var body: TextView
    private var paused = false
    private var done = false
    private var title = "Reading"
    private var text = ""
    private var voice = ""

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, i: Intent?) {
            when (val s = i?.getStringExtra("state")) {
                "loading" -> stateView.text = "warming up the model…"
                "generating" -> {
                    val chunk = i.getIntExtra("chunk", 0)
                    val total = i.getIntExtra("total", 0)
                    stateView.text = if (total > 0) "reading — part ${chunk + 1} of $total" else "reading…"
                    if (total > 0) { bar.max = total; bar.progress = chunk; bar.isIndeterminate = false }
                }
                "frames" -> {
                    val permille = i.getIntExtra("permille", -1)
                    if (permille >= 0) { bar.max = 1000; bar.progress = permille; bar.isIndeterminate = false }
                    val eta = i.getIntExtra("eta", -1)
                    if (eta > 0) stateView.text = "reading… ~${eta / 60}m ${eta % 60}s left"
                }
                "paused" -> { paused = true; syncButtons() }
                "resumed" -> { paused = false; syncButtons() }
                "done", "saved" -> {
                    done = true
                    bar.progress = bar.max
                    stateView.text = i.getStringExtra("message")?.takeIf { it.isNotBlank() } ?: "finished"
                    syncButtons()
                }
                "stopped" -> { stateView.text = "stopped"; done = true; syncButtons() }
                "error" -> { stateView.text = "error: ${i.getStringExtra("message")}"; done = true; syncButtons() }
                else -> if (s == "note") stateView.text = i.getStringExtra("message") ?: ""
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        DynamicColors.applyToActivityIfAvailable(this)
        title = intent.getStringExtra(EXTRA_TITLE) ?: "Reading"
        text = intent.getStringExtra(EXTRA_TEXT) ?: ""
        voice = intent.getStringExtra(EXTRA_VOICE) ?: ""

        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(16), dp(20), dp(16))
        }
        titleView = TextView(this).apply {
            text = this@PlayerActivity.title
            textSize = 22f; setTypeface(typeface, Typeface.BOLD)
            maxLines = 3
        }
        col.addView(titleView)
        col.addView(TextView(this).apply {
            val words = text.trim().split(Regex("\\s+")).count { it.isNotEmpty() }
            this.text = listOfNotNull(
                voice.takeIf { it.isNotBlank() }?.let { "${VoiceStore.icon(this@PlayerActivity, it)} $it" },
                "$words words · ~${words / 150} min",
            ).joinToString(" · ")
            textSize = 13f; alpha = 0.7f
            setPadding(0, dp(2), 0, dp(10))
        })
        stateView = TextView(this).apply { textSize = 13f; alpha = 0.8f; text = "starting…" }
        col.addView(stateView)
        bar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
            isIndeterminate = true
        }
        col.addView(bar, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
        ).apply { topMargin = dp(6); bottomMargin = dp(12) })

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        playBtn = Button(this).apply {
            text = "Pause"
            setOnClickListener { togglePause() }
        }
        controls.addView(playBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        controls.addView(Button(this).apply {
            text = "Stop"
            setOnClickListener {
                startService(Intent(this@PlayerActivity, TtsService::class.java)
                    .setAction(TtsService.ACTION_STOP))
            }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        controls.addView(Button(this).apply {
            text = "Save"
            setOnClickListener { save() }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        controls.addView(Button(this).apply {
            text = "Share"
            setOnClickListener { share() }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        col.addView(controls)

        body = TextView(this).apply {
            this.text = text
            textSize = 14f; alpha = 0.75f
            setPadding(0, dp(14), 0, 0)
        }
        col.addView(ScrollView(this).apply { addView(body) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        setContentView(col)
    }

    override fun onResume() {
        super.onResume()
        registerReceiver(receiver, IntentFilter(TtsService.STATUS_BROADCAST), RECEIVER_NOT_EXPORTED)
    }

    override fun onPause() {
        super.onPause()
        runCatching { unregisterReceiver(receiver) }
    }

    private fun syncButtons() {
        playBtn.text = when {
            done -> "Play again"
            paused -> "Resume"
            else -> "Pause"
        }
    }

    private fun togglePause() {
        if (done) {
            // finished: play the captured audio back rather than regenerating
            val wav = File(filesDir, "last_audio.wav")
            if (!wav.exists()) { toast("Nothing to replay"); return }
            runCatching {
                android.media.MediaPlayer().apply {
                    setDataSource(wav.absolutePath); prepare(); start()
                }
            }.onFailure { toast("Playback failed: ${it.message}") }
            return
        }
        paused = !paused
        startService(Intent(this, TtsService::class.java)
            .setAction(if (paused) TtsService.ACTION_PAUSE else TtsService.ACTION_RESUME))
        syncButtons()
    }

    /** Saving is a re-render with save on: the live pass streams to the
     *  speaker and keeps only the tail, so there is no complete file to copy
     *  until the job has finished. Once it has, the captured WAV is the file. */
    private fun save() {
        val wav = File(filesDir, "last_audio.wav")
        if (done && wav.exists()) {
            toast("Saving…")
            thread {
                val result = runCatching {
                    val clip = Wav.read(wav) ?: error("no audio")
                    val saver = AudioSaver(this, title, clip.rate)
                    saver.write(clip.pcm)
                    saver.finish()
                }
                runOnUiThread {
                    result.onSuccess { toast("Saved to $it") }
                        .onFailure { toast("Save failed: ${it.message}") }
                }
            }
            return
        }
        toast("Rendering a copy in the background…")
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, text)
            .putExtra(TtsService.EXTRA_TITLE, title)
            .putExtra(TtsService.EXTRA_VOICE, voice)
            .putExtra(TtsService.EXTRA_BACKEND, Backends.current(this))
            .putExtra(TtsService.EXTRA_SAVE, true))
    }

    private fun share() {
        val wav = File(filesDir, "last_audio.wav")
        if (!wav.exists()) { toast("Nothing generated yet"); return }
        toast("Preparing audio…")
        thread {
            runCatching { AudioShare.shareWavAsM4a(this, wav, title) }
                .onFailure { e -> runOnUiThread { toast("Share failed: ${e.message}") } }
        }
    }

    private fun toast(t: String) = Toast.makeText(this, t, Toast.LENGTH_SHORT).show()

    companion object {
        const val EXTRA_TITLE = "title"
        const val EXTRA_TEXT = "text"
        const val EXTRA_VOICE = "voice"
    }
}
