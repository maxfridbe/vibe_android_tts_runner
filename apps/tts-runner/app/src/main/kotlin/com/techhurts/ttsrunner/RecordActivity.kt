package com.techhurts.ttsrunner

import android.graphics.Typeface
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.text.Spannable
import android.text.SpannableStringBuilder
import android.text.style.ForegroundColorSpan
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.DynamicColors
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Record a reference voice by reading a standard passage aloud.
 *
 *  The passage has two sections: an expressive one (questions, exclamations,
 *  quoted speech — intonation range) and a steady one to be read flat and
 *  even. The steady text is the opening of the Rainbow Passage (Fairbanks
 *  1960), the phonetically balanced standard reading in speech research, so
 *  the recording covers the great majority of English mouth-shape sounds.
 *  Both sections are editable before recording.
 *
 *  While recording, the passage highlights word by word, advancing only while
 *  the mic actually hears speech — a pace guide, not speech recognition (live
 *  per-word tracking would need on-device whisper, a documented follow-up).
 *  Because the text read is KNOWN, saving also writes it as the voice's
 *  transcript — exactly what future ICL cloning / whisper validation needs. */
class RecordActivity : AppCompatActivity() {

    private lateinit var passageView: TextView
    private lateinit var passScroll: ScrollView
    private lateinit var levelBar: ProgressBar
    private lateinit var elapsedText: TextView
    private lateinit var recordBtn: Button
    private lateinit var playBtn: Button
    private lateinit var saveBtn: Button

    private var recorder: AudioRecord? = null
    @Volatile private var recording = false
    private var recThread: Thread? = null
    private var player: android.media.MediaPlayer? = null
    private val ui = android.os.Handler(android.os.Looper.getMainLooper())
    private lateinit var denoiseBox: android.widget.CheckBox
    private val effects = mutableListOf<android.media.audiofx.AudioEffect>()

    // karaoke state: char range + syllable weight per readable word
    private data class Word(val start: Int, val end: Int, val weight: Int)
    private var words = listOf<Word>()
    private var passage = SpannableStringBuilder()
    private var highlightSpans = mutableListOf<Any>()
    @Volatile private var spokenMs = 0L
    @Volatile private var totalMs = 0L
    private var noiseFloor = 300.0
    private var hangoverMs = 0

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun themeColor(attr: Int): Int {
        val tv = android.util.TypedValue()
        theme.resolveAttribute(attr, tv, true)
        return if (tv.resourceId != 0) getColor(tv.resourceId) else tv.data
    }

    private fun wavFile() = File(filesDir, "recorded_voice.wav")
    private fun prefs() = getSharedPreferences("ttsrunner", MODE_PRIVATE)
    private fun expressiveText() = prefs().getString("record_expressive", null) ?: DEFAULT_EXPRESSIVE
    private fun steadyText() = prefs().getString("record_steady", null) ?: DEFAULT_STEADY

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        DynamicColors.applyToActivityIfAvailable(this)

        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(16), dp(20), dp(16))
        }
        col.addView(TextView(this).apply {
            text = "Record a voice"
            textSize = 24f; setTypeface(typeface, Typeface.BOLD)
        })
        col.addView(TextView(this).apply {
            text = "Read the passage aloud — with feeling first, then flat and even. " +
                "Quiet room, 20–30 cm from the phone. The highlight follows your pace."
            textSize = 13f; alpha = 0.7f; setPadding(0, dp(2), 0, dp(10))
        })

        passageView = TextView(this).apply {
            textSize = 19f
            setLineSpacing(0f, 1.25f)
            setPadding(dp(16), dp(12), dp(16), dp(12))
        }
        passScroll = ScrollView(this).apply { addView(passageView) }
        col.addView(MaterialCardView(this).apply {
            radius = dp(16).toFloat(); cardElevation = 0f; strokeWidth = 1
            addView(passScroll)
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        val meterRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(10), 0, 0)
        }
        levelBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        meterRow.addView(levelBar, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        elapsedText = TextView(this).apply {
            textSize = 13f; alpha = 0.8f
            text = "aim for 20–40 s"
            setPadding(dp(10), 0, 0, 0)
        }
        meterRow.addView(elapsedText)
        col.addView(meterRow)

        denoiseBox = android.widget.CheckBox(this).apply {
            text = "Clean up mic input (noise suppression + level)"
            isChecked = prefs().getBoolean("record_denoise", false)
            setOnCheckedChangeListener { _, on ->
                prefs().edit().putBoolean("record_denoise", on).apply()
                if (recording) Toast.makeText(this@RecordActivity,
                    "Applies to the next recording", Toast.LENGTH_SHORT).show()
            }
        }
        col.addView(denoiseBox)

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        recordBtn = Button(this).apply {
            text = "Record"
            setOnClickListener { if (recording) stopRecording() else startRecording() }
        }
        buttons.addView(recordBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        playBtn = Button(this).apply {
            text = "Play"; isEnabled = wavFile().exists()
            setOnClickListener { togglePlayback() }
        }
        buttons.addView(playBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            .apply { marginStart = dp(6) })
        saveBtn = Button(this).apply {
            text = "Save voice"; isEnabled = wavFile().exists()
            setOnClickListener { saveDialog() }
        }
        buttons.addView(saveBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            .apply { marginStart = dp(6) })
        col.addView(buttons)
        col.addView(Button(this).apply {
            text = "Edit the passage"
            setOnClickListener { editDialog() }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))

        setContentView(col)
        rebuildPassage()

        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.RECORD_AUDIO), 21)
        }
    }

    override fun onPause() {
        super.onPause()
        if (recording) stopRecording()
        stopPlayback()
    }

    // ---- passage display + karaoke ----------------------------------------

    /** One TextView: styled section headers with the readable words' char
     *  ranges collected for the highlighter (headers are not readable text). */
    private fun rebuildPassage() {
        val sb = SpannableStringBuilder()
        val collected = mutableListOf<Word>()
        fun header(t: String) {
            val s = sb.length
            sb.append(t).append("\n")
            sb.setSpan(StyleSpan(Typeface.BOLD), s, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            sb.setSpan(RelativeSizeSpan(0.62f), s, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            sb.setSpan(ForegroundColorSpan(themeColor(com.google.android.material.R.attr.colorPrimary)),
                s, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        }
        fun body(t: String) {
            var i = sb.length
            sb.append(t)
            // words = maximal non-space runs; weight = vowel-group count
            Regex("\\S+").findAll(t).forEach { m ->
                val weight = max(1, Regex("[aeiouyAEIOUY]+").findAll(m.value).count())
                collected.add(Word(i + m.range.first, i + m.range.last + 1, weight))
            }
        }
        header("WITH FEELING")
        body(expressiveText())
        sb.append("\n\n")
        header("FLAT AND EVEN")
        body(steadyText())
        passage = sb
        words = collected
        passageView.text = passage
        highlightSpans.clear()
    }

    /** Advance the highlight through the words in proportion to how much
     *  actual speech the mic has heard, weighted by syllables per word. */
    private fun updateHighlight() {
        val targetSyllables = spokenMs / 1000.0 * SYLLABLES_PER_SEC
        var acc = 0.0
        var current = -1
        for ((i, w) in words.withIndex()) {
            if (acc + w.weight <= targetSyllables) { acc += w.weight; current = i } else break
        }
        highlightSpans.forEach { passage.removeSpan(it) }
        highlightSpans.clear()
        val done = themeColor(com.google.android.material.R.attr.colorPrimary)
        if (current >= 0) {
            val span = ForegroundColorSpan(done)
            passage.setSpan(span, words[0].start, words[current].end, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            highlightSpans.add(span)
        }
        val next = current + 1
        if (next < words.size) {
            val bold = StyleSpan(Typeface.BOLD)
            val color = ForegroundColorSpan(done)
            passage.setSpan(bold, words[next].start, words[next].end, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            passage.setSpan(color, words[next].start, words[next].end, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
            highlightSpans.add(bold); highlightSpans.add(color)
            // keep the current word in the upper third of the view
            passageView.layout?.let { layout ->
                val y = layout.getLineTop(layout.getLineForOffset(words[next].start))
                passScroll.smoothScrollTo(0, max(0, y - passScroll.height / 3))
            }
        }
        passageView.text = passage
    }

    // ---- recording ---------------------------------------------------------

    private fun startRecording() {
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) !=
            android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.RECORD_AUDIO), 21)
            return
        }
        stopPlayback()
        wavFile().delete()
        spokenMs = 0; totalMs = 0; noiseFloor = 300.0; hangoverMs = 0
        rebuildPassage()
        val minBuf = AudioRecord.getMinBufferSize(RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = try {
            // VOICE_RECOGNITION: no aggressive AGC/processing — closest to the
            // clean reference the speaker encoder wants
            AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, RATE,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, max(minBuf, RATE))
        } catch (e: Exception) {
            Toast.makeText(this, "Microphone unavailable: ${e.message}", Toast.LENGTH_LONG).show()
            return
        }
        recorder = rec
        // optional platform cleanup on the capture session: hardware/OS noise
        // suppression, echo cancel and gain levelling. Off by default — the
        // speaker encoder generally prefers raw audio, but a noisy room can
        // hurt more than the processing does.
        if (denoiseBox.isChecked) {
            runCatching {
                if (android.media.audiofx.NoiseSuppressor.isAvailable())
                    android.media.audiofx.NoiseSuppressor.create(rec.audioSessionId)
                        ?.also { it.enabled = true; effects.add(it) }
                if (android.media.audiofx.AcousticEchoCanceler.isAvailable())
                    android.media.audiofx.AcousticEchoCanceler.create(rec.audioSessionId)
                        ?.also { it.enabled = true; effects.add(it) }
                if (android.media.audiofx.AutomaticGainControl.isAvailable())
                    android.media.audiofx.AutomaticGainControl.create(rec.audioSessionId)
                        ?.also { it.enabled = true; effects.add(it) }
            }
        }
        recording = true
        recordBtn.text = "Stop"
        playBtn.isEnabled = false
        saveBtn.isEnabled = false
        rec.startRecording()
        recThread = thread(name = "voice-record") {
            val pcm = java.io.ByteArrayOutputStream()
            val buf = ShortArray(RATE / 10)   // 100 ms
            while (recording) {
                val n = recorder?.read(buf, 0, buf.size) ?: break
                if (n <= 0) continue
                val bytes = ByteArray(n * 2)
                for (i in 0 until n) {
                    bytes[2 * i] = (buf[i].toInt() and 0xff).toByte()
                    bytes[2 * i + 1] = (buf[i].toInt() shr 8).toByte()
                }
                pcm.write(bytes)
                var sum = 0.0
                for (i in 0 until n) sum += buf[i].toDouble() * buf[i]
                val rms = sqrt(sum / n)
                // Forgiving VAD — the guide getting stuck mid-passage is worse
                // than it running slightly ahead: low threshold, floor adapts
                // downward fast and upward only slowly, and brief gaps between
                // words keep counting as speech (hangover).
                val speaking = rms > max(250.0, noiseFloor * 2)
                noiseFloor = if (rms < noiseFloor) noiseFloor * 0.7 + rms * 0.3
                             else min(noiseFloor * 1.005, 2000.0)
                if (speaking) hangoverMs = 400
                val counts = speaking || hangoverMs > 0
                if (!speaking && hangoverMs > 0) hangoverMs -= (n * 1000 / RATE).toInt()
                if (counts) spokenMs += n * 1000L / RATE
                totalMs += n * 1000L / RATE
                ui.post {
                    levelBar.progress = min(100.0, rms / 60.0).toInt()
                    elapsedText.text = "%d:%02d · aim for 20–40 s".format(totalMs / 60000, totalMs / 1000 % 60)
                    updateHighlight()
                }
            }
            runCatching { writeWav(wavFile(), pcm.toByteArray(), RATE) }
            ui.post {
                val ok = wavFile().length() > 44
                playBtn.isEnabled = ok
                saveBtn.isEnabled = ok
                levelBar.progress = 0
            }
        }
    }

    private fun stopRecording() {
        recording = false
        recordBtn.text = "Record again"
        runCatching { recorder?.stop() }
        runCatching { recorder?.release() }
        recorder = null
        effects.forEach { runCatching { it.release() } }
        effects.clear()
        recThread?.join(2000)
        // whatever the pace guide thought, the take is over: show the whole
        // passage as read rather than leaving it stuck mid-word
        spokenMs = Long.MAX_VALUE / 2
        updateHighlight()
    }

    private fun writeWav(f: File, pcm: ByteArray, rate: Int) {
        java.io.DataOutputStream(f.outputStream().buffered()).use { o ->
            fun le32(v: Int) { o.write(v and 0xff); o.write(v shr 8 and 0xff); o.write(v shr 16 and 0xff); o.write(v shr 24 and 0xff) }
            fun le16(v: Int) { o.write(v and 0xff); o.write(v shr 8 and 0xff) }
            o.writeBytes("RIFF"); le32(36 + pcm.size); o.writeBytes("WAVE")
            o.writeBytes("fmt "); le32(16); le16(1); le16(1); le32(rate); le32(rate * 2); le16(2); le16(16)
            o.writeBytes("data"); le32(pcm.size)
            o.write(pcm)
        }
    }

    // ---- playback / save / edit -------------------------------------------

    private fun togglePlayback() {
        val p = player
        if (p != null) { stopPlayback(); return }
        try {
            player = android.media.MediaPlayer().apply {
                setDataSource(wavFile().absolutePath)
                prepare()
                setOnCompletionListener { stopPlayback() }
                start()
            }
            playBtn.text = "Stop playback"
        } catch (e: Exception) {
            Toast.makeText(this, "Playback failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun stopPlayback() {
        player?.let { runCatching { it.stop() }; it.release() }
        player = null
        playBtn.text = "Play"
    }

    private fun saveDialog() {
        val edit = EditText(this).apply {
            hint = "Voice name"; setSingleLine()
            setPadding(dp(20), dp(12), dp(20), dp(12))
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("Save recorded voice")
            .setView(edit)
            .setPositiveButton("Save") { _, _ ->
                val name = edit.text.toString().trim().ifBlank { "recorded" }
                try {
                    val v = VoiceStore.adopt(this, wavFile(), name)
                    // the transcript is known exactly — it is the passage read
                    VoiceStore.transcriptFile(this, v.name)
                        .writeText(expressiveText() + "\n\n" + steadyText())
                    Toast.makeText(this, "Voice “${v.name}” saved", Toast.LENGTH_LONG).show()
                    finish()
                } catch (e: Exception) {
                    Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun editDialog() {
        if (recording) { Toast.makeText(this, "Stop recording first", Toast.LENGTH_SHORT).show(); return }
        val pad = dp(20)
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, dp(8), pad, dp(4))
        }
        col.addView(TextView(this).apply { text = "With feeling"; textSize = 12f; alpha = 0.7f })
        val exp = EditText(this).apply { setText(expressiveText()); minLines = 3 }
        col.addView(exp)
        col.addView(TextView(this).apply {
            text = "Flat and even"; textSize = 12f; alpha = 0.7f; setPadding(0, dp(10), 0, 0)
        })
        val steady = EditText(this).apply { setText(steadyText()); minLines = 3 }
        col.addView(steady)
        MaterialAlertDialogBuilder(this)
            .setTitle("Edit the passage")
            .setView(ScrollView(this).apply { addView(col) })
            .setPositiveButton("Save") { _, _ ->
                prefs().edit()
                    .putString("record_expressive", exp.text.toString().trim().ifBlank { DEFAULT_EXPRESSIVE })
                    .putString("record_steady", steady.text.toString().trim().ifBlank { DEFAULT_STEADY })
                    .apply()
                rebuildPassage()
            }
            .setNeutralButton("Reset to standard") { _, _ ->
                prefs().edit().remove("record_expressive").remove("record_steady").apply()
                rebuildPassage()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    companion object {
        private const val RATE = 16000
        private const val SYLLABLES_PER_SEC = 3.2

        /** Intonation range: exclamation, question, quoted speech, an aside. */
        const val DEFAULT_EXPRESSIVE =
            "Oh, that's wonderful news! Are you absolutely sure? I could hardly believe it " +
            "when I heard. “Come quickly,” she called, “before the light fades!” " +
            "Well then — what are we waiting for?"

        /** Opening of the Rainbow Passage (Fairbanks 1960): the phonetically
         *  balanced standard reading used across speech research, covering the
         *  large majority of English phonemes in a few sentences. */
        const val DEFAULT_STEADY =
            "When the sunlight strikes raindrops in the air, they act as a prism and form " +
            "a rainbow. The rainbow is a division of white light into many beautiful colors. " +
            "These take the shape of a long round arch, with its path high above, and its " +
            "two ends apparently beyond the horizon."
    }
}
