package com.techhurts.ttsrunner

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.BackgroundColorSpan
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.HorizontalScrollView
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
 *  the generated audio, watch the waveform build as each chunk lands, follow
 *  the cleaned-up text as it is read, and keep the result when it turns out to
 *  be worth keeping. */
class PlayerActivity : AppCompatActivity() {

    private lateinit var titleView: TextView
    private lateinit var stateView: TextView
    private lateinit var bar: ProgressBar
    private lateinit var playBtn: Button
    private lateinit var body: TextView
    private lateinit var bodyScroll: ScrollView
    private lateinit var wave: LiveWaveformView
    private lateinit var waveScroll: HorizontalScrollView

    private var paused = false
    private var done = false
    private var title = "Reading"
    private var text = ""
    private var voice = ""

    private var chunks: List<String> = emptyList()
    private var starts = IntArray(0)   // char offset of each chunk in the body
    private var activeChunk = -1

    private var replay: android.media.MediaPlayer? = null
    private val ui = Handler(Looper.getMainLooper())

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun pcmFile() = File(filesDir, "last_audio.pcm")
    private fun wavFile() = File(filesDir, "last_audio.wav")

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, i: Intent?) {
            i?.getIntExtra("rate", 0)?.takeIf { it > 0 }?.let { wave.setRate(it) }
            when (val s = i?.getStringExtra("state")) {
                "loading" -> stateView.text = "warming up the model…"
                "generating" -> {
                    val chunk = i.getIntExtra("chunk", 0)
                    val total = i.getIntExtra("total", 0)
                    stateView.text = if (total > 0) "reading — part ${chunk + 1} of $total" else "reading…"
                    if (total > 0) { bar.max = total; bar.progress = chunk; bar.isIndeterminate = false }
                    setActiveChunk(chunk)
                    growWave()
                }
                "frames" -> {
                    val permille = i.getIntExtra("permille", -1)
                    if (permille >= 0) { bar.max = 1000; bar.progress = permille; bar.isIndeterminate = false }
                    val eta = i.getIntExtra("eta", -1)
                    if (eta > 0) stateView.text = "reading… ~${eta / 60}m ${eta % 60}s left"
                    growWave()
                }
                "paused" -> { paused = true; syncButtons() }
                "resumed" -> { paused = false; syncButtons() }
                "done", "saved", "designed" -> {
                    done = true
                    bar.progress = bar.max
                    stateView.text = i.getStringExtra("message")?.takeIf { it.isNotBlank() } ?: "finished"
                    setActiveChunk(chunks.size)   // everything read
                    finalizeWave()
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
        chunks = Chunker.split(text)

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
                voice.takeIf { it.isNotBlank() }?.let { VoiceStore.label(this@PlayerActivity, it) },
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
        ).apply { topMargin = dp(6); bottomMargin = dp(10) })

        // waveform that extends to the right as each chunk is generated; a
        // horizontal scroller lets a long read run off the edge and auto-follow
        wave = LiveWaveformView(this)
        waveScroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            isFillViewport = true
            addView(wave, ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT))
        }
        col.addView(waveScroll, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, dp(64)).apply { bottomMargin = dp(12) })

        val controls = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        playBtn = transportBtn(Icons.PAUSE, "Pause") { togglePause() }
        controls.addView(playBtn, weight())
        controls.addView(transportBtn(Icons.STOP, "Stop") {
            startService(Intent(this@PlayerActivity, TtsService::class.java)
                .setAction(TtsService.ACTION_STOP))
        }, weight())
        controls.addView(transportBtn(Icons.SAVE, "Save") { save() }, weight())
        controls.addView(transportBtn(Icons.SHARE, "Share") { share() }, weight())
        col.addView(controls)

        body = TextView(this).apply {
            this.text = text
            textSize = 15f
            setLineSpacing(dp(4).toFloat(), 1f)
            setPadding(0, dp(14), 0, dp(8))
        }
        bodyScroll = ScrollView(this).apply { addView(body) }
        col.addView(bodyScroll,
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        setContentView(col)
        setActiveChunk(-1)   // render the cleaned text before the first chunk
    }

    private fun weight() =
        LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            .apply { marginStart = dp(3); marginEnd = dp(3) }

    /** A transport control drawn as a FontAwesome glyph over its label. */
    private fun transportBtn(glyph: String, label: String, onClick: () -> Unit) =
        Button(this).apply {
            text = Icons.label(this@PlayerActivity, glyph, label)
            setPadding(dp(4), 0, dp(4), 0)
            contentDescription = label
            setOnClickListener { onClick() }
        }

    override fun onResume() {
        super.onResume()
        registerReceiver(receiver, IntentFilter(TtsService.STATUS_BROADCAST), RECEIVER_NOT_EXPORTED)
        // opened after the job already finished (e.g. replay): show the whole
        // shape and mark it done, rather than waiting for broadcasts that the
        // (now idle) engine will never send
        if (!done && !pcmFile().exists() && wavFile().let { it.exists() && it.length() > 44 }) {
            done = true
            stateView.text = "finished"
            bar.isIndeterminate = false; bar.max = 1; bar.progress = 1
            setActiveChunk(chunks.size)
            finalizeWave()
            syncButtons()
        }
    }

    override fun onPause() {
        super.onPause()
        runCatching { unregisterReceiver(receiver) }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopReplay()
    }

    // ---- waveform ----------------------------------------------------------

    private fun growWave() {
        if (done) return
        val f = pcmFile()
        if (!f.exists()) return
        wave.appendPcm(f)
        waveScroll.post { waveScroll.fullScroll(View.FOCUS_RIGHT) }
    }

    private fun finalizeWave() {
        val f = wavFile()
        if (!f.exists() || f.length() <= 44) return
        thread {
            wave.loadWav(f)
            runOnUiThread { waveScroll.fullScroll(View.FOCUS_LEFT); wave.setProgress(-1f) }
        }
    }

    // ---- text follow -------------------------------------------------------

    /** Rebuild the body with the cleaned-up chunk text, dimming what has been
     *  read, highlighting the chunk in flight, and scrolling it into view — the
     *  karaoke line that makes a long read easy to keep your place in. */
    private fun setActiveChunk(active: Int) {
        activeChunk = active
        if (chunks.isEmpty()) return
        val sb = SpannableStringBuilder()
        starts = IntArray(chunks.size)
        for ((i, c) in chunks.withIndex()) {
            starts[i] = sb.length
            sb.append(c)
            val end = sb.length
            when {
                i < active -> sb.setSpan(ForegroundColorSpan(0xFF9E9E9E.toInt()),
                    starts[i], end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                i == active -> {
                    sb.setSpan(BackgroundColorSpan(0x553F51B5), starts[i], end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                    sb.setSpan(StyleSpan(Typeface.BOLD), starts[i], end, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                }
            }
            if (i < chunks.size - 1) sb.append("\n\n")
        }
        body.text = sb
        if (active in chunks.indices) scrollChunkIntoView(active)
    }

    private fun scrollChunkIntoView(active: Int) {
        body.post {
            val layout = body.layout ?: return@post
            if (active !in starts.indices) return@post
            val line = layout.getLineForOffset(starts[active])
            val y = layout.getLineTop(line) + body.paddingTop
            bodyScroll.smoothScrollTo(0, (y - bodyScroll.height / 3).coerceAtLeast(0))
        }
    }

    private fun syncButtons() {
        val (glyph, label) = when {
            done -> Icons.PLAY to "Play again"
            paused -> Icons.PLAY to "Resume"
            else -> Icons.PAUSE to "Pause"
        }
        playBtn.text = Icons.label(this, glyph, label)
        playBtn.contentDescription = label
    }

    private fun togglePause() {
        if (done) { toggleReplay(); return }
        paused = !paused
        startService(Intent(this, TtsService::class.java)
            .setAction(if (paused) TtsService.ACTION_PAUSE else TtsService.ACTION_RESUME))
        syncButtons()
    }

    // ---- replay of the finished audio --------------------------------------

    private fun toggleReplay() {
        replay?.let { mp ->
            if (mp.isPlaying) { mp.pause(); ui.removeCallbacks(tick) }
            else { mp.start(); ui.post(tick) }
            return
        }
        val wav = wavFile()
        if (!wav.exists()) { toast("Nothing to replay"); return }
        runCatching {
            replay = android.media.MediaPlayer().apply {
                setDataSource(wav.absolutePath); prepare(); start()
                setOnCompletionListener { stopReplay() }
            }
            ui.post(tick)
        }.onFailure { toast("Playback failed: ${it.message}") }
    }

    /** Drive the waveform cursor and the karaoke highlight off the replay clock. */
    private val tick = object : Runnable {
        override fun run() {
            val mp = replay ?: return
            val dur = mp.duration.coerceAtLeast(1)
            val frac = (mp.currentPosition.toFloat() / dur).coerceIn(0f, 1f)
            wave.setProgress(frac)
            // keep the moving cursor on screen
            val target = (frac * wave.width - waveScroll.width / 2f).toInt().coerceAtLeast(0)
            waveScroll.smoothScrollTo(target, 0)
            if (chunks.isNotEmpty()) {
                val idx = (frac * chunks.size).toInt().coerceIn(0, chunks.size - 1)
                if (idx != activeChunk) setActiveChunk(idx)
            }
            if (mp.isPlaying) ui.postDelayed(this, 200)
        }
    }

    private fun stopReplay() {
        ui.removeCallbacks(tick)
        replay?.runCatching { stop(); release() }
        replay = null
        wave.setProgress(-1f)
    }

    /** Saving is a re-render with save on: the live pass streams to the
     *  speaker and keeps only the tail, so there is no complete file to copy
     *  until the job has finished. Once it has, the captured WAV is the file. */
    private fun save() {
        val wav = wavFile()
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
        val wav = wavFile()
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
