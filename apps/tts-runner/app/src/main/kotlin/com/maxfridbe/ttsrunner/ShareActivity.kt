package com.maxfridbe.ttsrunner

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
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/** The "send to" target: receives shared text / URLs / selected text, cleans
 *  it (fetching + extracting the article when a URL was shared), then pops a
 *  voice selector and hands off to TtsService. Dialog-themed so it floats
 *  over the sharing app; AppCompat so the Material dialog theme resolves the
 *  same way it does in MainActivity. */
class ShareActivity : AppCompatActivity() {

    private lateinit var status: TextView
    private lateinit var preview: TextView
    private lateinit var voices: RadioGroup
    private lateinit var speakBtn: Button
    private lateinit var saveBtn: Button
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
        // enough of the extraction to judge whether the readability pass got
        // the article and not the navigation
        preview = TextView(this).apply {
            maxLines = 8
            textSize = 13f
            alpha = 0.8f
            ellipsize = android.text.TextUtils.TruncateAt.END
            setPadding(0, pad / 2, 0, pad / 2)
        }
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
        saveBtn = Button(this).apply {
            text = "Save"   // render to Music/TTS Runner/*.m4a in the background
            isEnabled = false
            setOnClickListener { speak(save = true) }
        }
        buttons.addView(saveBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        speakBtn = Button(this).apply {
            text = "Speak"
            isEnabled = false
            setOnClickListener { speak(save = false) }
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
                DebugLog.log(this, "ShareActivity",
                    "extracted ${result.text.length} chars | ${result.text.take(140)} … ${result.text.takeLast(140)}")
                val words = result.text.split(Regex("\\s+")).size
                status.text = listOfNotNull(result.title, "$words words · ~${words / 150} min").joinToString(" — ")
                preview.text = result.text.take(300)
                speakBtn.isEnabled = voices.childCount > 0
                saveBtn.isEnabled = voices.childCount > 0
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

    /** The engines take different kinds of voice — Supertonic reads style
     *  JSONs, Qwen reads reference recordings — so the picker has to follow the
     *  selected model. Listing only recordings left this dialog empty (and
     *  Speak greyed out) for anyone running Supertonic. */
    private fun voiceList(): List<VoiceStore.Voice> {
        val supertonic = ModelManager.selectedModel(this)?.engine == "supertonic"
        return if (supertonic) VoiceStore.styleList(this) else VoiceStore.list(this)
    }

    private fun populateVoices() {
        val list = voiceList()
        if (list.isEmpty()) {
            val supertonic = ModelManager.selectedModel(this)?.engine == "supertonic"
            status.text = if (supertonic)
                "No Supertonic speakers yet — open TTS Runner and import a style"
            else "No voices yet — open TTS Runner and import one first"
            return
        }
        val supertonic = ModelManager.selectedModel(this)?.engine == "supertonic"
        val def = VoiceStore.defaultFor(this, supertonic)
        for ((i, v) in list.withIndex()) {
            voices.addView(RadioButton(this).apply {
                text = "${VoiceStore.icon(this@ShareActivity, v.name)}  ${v.name}"
                id = i
                isChecked = v.name == def?.name
            })
        }
        if (voices.checkedRadioButtonId == -1) voices.check(0)
    }

    private fun speak(save: Boolean) {
        val c = cleaned ?: return
        val voice = voiceList().getOrNull(voices.checkedRadioButtonId) ?: return
        VoiceStore.setDefaultFor(this, voice,
            ModelManager.selectedModel(this)?.engine == "supertonic")
        val backend = getSharedPreferences("ttsrunner", MODE_PRIVATE).getString("backend", "cpu") ?: "cpu"
        startForegroundService(
            Intent(this, TtsService::class.java)
                .setAction(TtsService.ACTION_SPEAK)
                .putExtra(TtsService.EXTRA_TEXT, c.text)
                .putExtra(TtsService.EXTRA_TITLE, c.title ?: "Shared text")
                .putExtra(TtsService.EXTRA_VOICE, voice.name)
                .putExtra(TtsService.EXTRA_BACKEND, backend)
                .putExtra(TtsService.EXTRA_SAVE, save))
        finish()
    }

    companion object { private const val PROGRESS_ID = 0x7f0a0001 }
}
