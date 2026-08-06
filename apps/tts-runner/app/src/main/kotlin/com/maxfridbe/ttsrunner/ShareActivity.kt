package com.maxfridbe.ttsrunner

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import kotlin.concurrent.thread

/** The "send to" target: receives shared text / URLs / selected text, cleans
 *  it (fetching + extracting the article when a URL was shared), then pops a
 *  voice selector and hands off to TtsService. Dialog-themed so it floats
 *  over the sharing app. */
class ShareActivity : Activity() {

    private lateinit var status: TextView
    private lateinit var preview: TextView
    private lateinit var voices: RadioGroup
    private lateinit var speakBtn: Button
    private var cleaned: TextCleaner.Cleaned? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }
        root.addView(TextView(this).apply {
            text = getString(R.string.app_name)
            textSize = 20f
            setTypeface(typeface, android.graphics.Typeface.BOLD)
        })
        status = TextView(this).apply { text = "Cleaning text…" }
        root.addView(status)
        preview = TextView(this).apply { maxLines = 4; setPadding(0, pad / 2, 0, pad / 2) }
        root.addView(preview)
        root.addView(ProgressBar(this).apply { isIndeterminate = true; id = PROGRESS_ID })
        voices = RadioGroup(this)
        root.addView(ScrollView(this).apply { addView(voices) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        buttons.addView(Button(this).apply {
            text = "Cancel"
            setOnClickListener { finish() }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        speakBtn = Button(this).apply {
            text = "Speak"
            isEnabled = false
            setOnClickListener { speak() }
        }
        buttons.addView(speakBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        root.addView(buttons)

        setContentView(root)

        val shared = extractSharedText()
        if (shared.isNullOrBlank()) {
            status.text = "Nothing shareable received"
            return
        }
        populateVoices()
        thread {
            val result = try {
                TextCleaner.clean(shared)
            } catch (e: Exception) {
                DebugLog.log(this, "ShareActivity", "clean failed for ${shared.take(120)}", e)
                null
            }
            runOnUiThread {
                findViewById<ProgressBar>(PROGRESS_ID).visibility = View.GONE
                if (result == null || result.text.isBlank()) {
                    status.text = "Could not extract readable text"
                    return@runOnUiThread
                }
                cleaned = result
                val words = result.text.split(Regex("\\s+")).size
                status.text = listOfNotNull(result.title, "$words words · ~${words / 150} min").joinToString(" — ")
                preview.text = result.text.take(300)
                speakBtn.isEnabled = voices.childCount > 0
            }
        }
    }

    private fun extractSharedText(): String? {
        return when (intent?.action) {
            Intent.ACTION_SEND -> intent.getStringExtra(Intent.EXTRA_TEXT)
            Intent.ACTION_PROCESS_TEXT -> intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
            else -> null
        }
    }

    private fun populateVoices() {
        val list = VoiceStore.list(this)
        if (list.isEmpty()) {
            status.text = "No voices yet — open TTS Runner and import one first"
            return
        }
        val def = VoiceStore.defaultVoice(this)
        for ((i, v) in list.withIndex()) {
            voices.addView(RadioButton(this).apply {
                text = v.name
                id = i
                isChecked = v.name == def?.name
            })
        }
        if (voices.checkedRadioButtonId == -1) voices.check(0)
    }

    private fun speak() {
        val c = cleaned ?: return
        val list = VoiceStore.list(this)
        val voice = list.getOrNull(voices.checkedRadioButtonId) ?: return
        VoiceStore.setDefault(this, voice)
        val backend = getSharedPreferences("ttsrunner", MODE_PRIVATE).getString("backend", "cpu") ?: "cpu"
        startForegroundService(
            Intent(this, TtsService::class.java)
                .setAction(TtsService.ACTION_SPEAK)
                .putExtra(TtsService.EXTRA_TEXT, c.text)
                .putExtra(TtsService.EXTRA_TITLE, c.title ?: "Shared text")
                .putExtra(TtsService.EXTRA_VOICE, voice.name)
                .putExtra(TtsService.EXTRA_BACKEND, backend))
        finish()
    }

    companion object { private const val PROGRESS_ID = 0x7f0a0001 }
}
