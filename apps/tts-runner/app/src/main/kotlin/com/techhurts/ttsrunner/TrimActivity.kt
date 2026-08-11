package com.techhurts.ttsrunner

import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.max

/** Pick a section of a picked recording before it becomes a speaker.
 *
 *  A dropped-in file is often minutes long with silence or music at the ends;
 *  cloning wants a clean ~10 s of one voice. This decodes the file, draws its
 *  waveform with two drag handles, plays the selection, and returns a trimmed
 *  mono WAV. The caller (MainActivity) then imports or clones that.
 *
 *  In and out:
 *    EXTRA_SOURCE_URI  (in)  content uri of the picked audio
 *    EXTRA_RESULT_PATH (out) path of the trimmed wav in cacheDir */
class TrimActivity : AppCompatActivity() {

    private lateinit var wave: TrimWaveformView
    private lateinit var rangeLabel: TextView
    private var pcm: AudioDecoder.Pcm? = null
    private var player: android.media.MediaPlayer? = null
    private val ui = android.os.Handler(android.os.Looper.getMainLooper())
    private var cursorTick: Runnable? = null

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "Trim recording"
        val uri = intent.getParcelableExtra<Uri>(EXTRA_SOURCE_URI)
        if (uri == null) { finish(); return }

        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(20), dp(20), dp(20))
        }
        col.addView(TextView(this).apply {
            text = "Drag the two handles to keep just the part you want — about ten " +
                "seconds of clear speech clones best."
            textSize = 13f; alpha = 0.75f
            setPadding(0, 0, 0, dp(14))
        })
        wave = TrimWaveformView(this)
        col.addView(wave, LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, dp(160)))
        rangeLabel = TextView(this).apply {
            textSize = 13f; setPadding(0, dp(10), 0, dp(10))
            text = "Decoding…"
        }
        col.addView(rangeLabel)

        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val playBtn = Button(this).apply {
            text = Icons.label(context, Icons.PLAY, "Play selection"); isEnabled = false
            setOnClickListener { playSelection() }
        }
        val useBtn = Button(this).apply {
            text = Icons.label(context, Icons.CHECK, "Use selection"); isEnabled = false
            setOnClickListener { useSelection() }
        }
        row.addView(playBtn, LinearLayout.LayoutParams(0,
            LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(useBtn, LinearLayout.LayoutParams(0,
            LinearLayout.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(8) })
        col.addView(row)
        setContentView(col)

        wave.onSelection = { _, _ -> updateLabel() }

        thread {
            val decoded = AudioDecoder.decode(this, uri)
            ui.post {
                if (decoded == null || decoded.samples.isEmpty()) {
                    rangeLabel.text = "Could not decode that file."
                    return@post
                }
                pcm = decoded
                wave.setPeaks(peaksOf(decoded.samples))
                // default to a sensible middle window, ≤ 12 s
                val dur = decoded.seconds
                if (dur > 12.0) {
                    val a = ((dur - 12.0) / 2.0 / dur).toFloat()
                    wave.setSelection(a, a + (12.0 / dur).toFloat())
                }
                playBtn.isEnabled = true
                useBtn.isEnabled = true
                updateLabel()
            }
        }
    }

    private fun peaksOf(s: ShortArray, buckets: Int = 160): IntArray {
        val out = IntArray(buckets)
        if (s.isEmpty()) return out
        val per = max(1, s.size / buckets)
        for (b in 0 until buckets) {
            var peak = 0
            var i = b * per
            val end = minOf(s.size, i + per)
            val step = max(1, per / 64)
            while (i < end) { peak = max(peak, kotlin.math.abs(s[i].toInt())); i += step }
            out[b] = peak
        }
        return out
    }

    private fun sample(frac: Float) = (frac * (pcm?.samples?.size ?: 0)).toInt()

    private fun updateLabel() {
        val p = pcm ?: return
        val a = sample(wave.selStart); val b = sample(wave.selEnd)
        val sel = (b - a).toDouble() / p.rate
        rangeLabel.text = "Selection: ${"%.1f".format(sel)} s of ${"%.1f".format(p.seconds)} s" +
            if (sel > 13) "  (long clips slow cloning and may run out of memory)" else ""
    }

    private fun playSelection() {
        val p = pcm ?: return
        stopPlayer()
        val clip = File(cacheDir, "trim_preview.wav")
        AudioDecoder.writeTrimmed(clip, p, sample(wave.selStart), sample(wave.selEnd))
        player = android.media.MediaPlayer().apply {
            setDataSource(clip.absolutePath)
            setOnCompletionListener { stopPlayer() }
            prepare(); start()
        }
        val startFrac = wave.selStart; val span = wave.selEnd - wave.selStart
        cursorTick = object : Runnable {
            override fun run() {
                val mp = player ?: return
                val frac = startFrac + span * (mp.currentPosition.toFloat() /
                    max(1, mp.duration))
                wave.setCursor(frac)
                ui.postDelayed(this, 30)
            }
        }.also { ui.post(it) }
    }

    private fun stopPlayer() {
        cursorTick?.let { ui.removeCallbacks(it) }; cursorTick = null
        wave.setCursor(-1f)
        player?.let { runCatching { it.stop() }; it.release() }
        player = null
    }

    private fun useSelection() {
        val p = pcm ?: return
        val out = File(cacheDir, "trimmed_clone.wav")
        AudioDecoder.writeTrimmed(out, p, sample(wave.selStart), sample(wave.selEnd))
        stopPlayer()
        setResult(RESULT_OK, intent.putExtra(EXTRA_RESULT_PATH, out.absolutePath))
        finish()
    }

    override fun onStop() { super.onStop(); stopPlayer() }

    companion object {
        const val EXTRA_SOURCE_URI = "source_uri"
        const val EXTRA_RESULT_PATH = "result_path"
    }
}
