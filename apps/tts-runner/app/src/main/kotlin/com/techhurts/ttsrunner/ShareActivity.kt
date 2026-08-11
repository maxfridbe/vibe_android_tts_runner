package com.techhurts.ttsrunner

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/** The "send to" target: receives shared text / URLs / selected text, cleans
 *  it (fetching + extracting the article when a URL was shared), shows the
 *  result for review, then hands off to TtsService. Dialog-themed so it floats
 *  over the sharing app; AppCompat so the Material dialog theme resolves the
 *  same way it does in MainActivity.
 *
 *  The extraction is editable on purpose. Readability heuristics get the
 *  article most of the time and the navigation menu the rest of the time, and
 *  the only person who can tell the difference in one glance is the reader —
 *  so they get the text, not a preview of it. */
class ShareActivity : AppCompatActivity() {

    private lateinit var status: TextView
    private lateinit var editor: EditText
    private lateinit var voices: RadioGroup
    private lateinit var speakBtn: Button
    private lateinit var saveBtn: Button
    private lateinit var liveBtn: Button
    private var cleaned: TextCleaner.Cleaned? = null

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = dp(16)
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
        root.addView(ProgressBar(this).apply { isIndeterminate = true; id = PROGRESS_ID })

        root.addView(TextView(this).apply {
            text = "Check the text before it is read — trim menus or footers the extractor kept."
            textSize = 12f; alpha = 0.7f
            setPadding(0, dp(6), 0, dp(4))
        })
        editor = EditText(this).apply {
            textSize = 13f
            gravity = android.view.Gravity.TOP or android.view.Gravity.START
            isVerticalScrollBarEnabled = true
            setPadding(dp(10), dp(8), dp(10), dp(8))
        }
        // the editor takes the room: it is the thing being judged
        root.addView(editor, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 3f))

        root.addView(TextView(this).apply {
            text = "Speaker"
            textSize = 12f; alpha = 0.7f
            setPadding(0, dp(8), 0, 0)
        })
        voices = RadioGroup(this)
        root.addView(ScrollView(this).apply { addView(voices) },
            LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 2f))

        val buttons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        buttons.addView(Button(this).apply {
            text = Icons.label(context, Icons.CLOSE, "Cancel")
            setOnClickListener { finish() }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        saveBtn = Button(this).apply {
            text = Icons.label(context, Icons.SAVE, "Save")   // render to Music/TTS Runner/*.m4a in the background
            isEnabled = false
            setOnClickListener { speak(save = true, live = false) }
        }
        buttons.addView(saveBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        speakBtn = Button(this).apply {
            text = Icons.label(context, Icons.VOLUME, "Speak")
            isEnabled = false
            setOnClickListener { speak(save = false, live = false) }
        }
        buttons.addView(speakBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        liveBtn = Button(this).apply {
            text = Icons.label(context, Icons.PLAY, "Play live")
            isEnabled = false
            setOnClickListener { speak(save = false, live = true) }
        }
        buttons.addView(liveBtn, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
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
                    status.text = "Could not extract readable text — paste it here instead"
                    enableActions()
                    return@runOnUiThread
                }
                cleaned = result
                DebugLog.log(this, "ShareActivity",
                    "extracted ${result.text.length} chars | ${result.text.take(140)} … ${result.text.takeLast(140)}")
                editor.setText(result.text)
                showCount(result.title)
                editor.addTextChangedListener(object : android.text.TextWatcher {
                    override fun afterTextChanged(s: android.text.Editable?) = showCount(result.title)
                    override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                    override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
                })
                enableActions()
            }
        }
    }

    private fun showCount(title: String?) {
        val words = editor.text.toString().trim().split(Regex("\\s+")).count { it.isNotEmpty() }
        status.text = listOfNotNull(title, "$words words · ~${words / 150} min").joinToString(" — ")
    }

    private fun enableActions() {
        val ok = voices.childCount > 0
        speakBtn.isEnabled = ok
        saveBtn.isEnabled = ok
        liveBtn.isEnabled = ok
    }

    private fun extractSharedText(): String? {
        return when (intent?.action) {
            Intent.ACTION_SEND -> intent.getStringExtra(Intent.EXTRA_TEXT)
            Intent.ACTION_PROCESS_TEXT -> intent.getCharSequenceExtra(Intent.EXTRA_PROCESS_TEXT)?.toString()
            else -> null
        }
    }

    /** Every speaker with a model to voice it — Supertonic styles when a
     *  Supertonic model is installed, Qwen recordings when a Qwen one is. No
     *  global model gates this; the voice picked decides the engine. */
    private fun voiceList(): List<VoiceStore.Voice> = buildList {
        if (ModelManager.modelForEngine(this@ShareActivity, "supertonic") != null)
            addAll(VoiceStore.styleList(this@ShareActivity))
        if (ModelManager.modelForEngine(this@ShareActivity, "qwen") != null)
            addAll(VoiceStore.list(this@ShareActivity))
    }

    private fun populateVoices() {
        val list = voiceList()
        if (list.isEmpty()) {
            status.text = if (ModelManager.anyModel(this) == null)
                "No model yet — open TTS Runner and download one"
            else "No voices yet — open TTS Runner and add a speaker"
            return
        }
        val def = VoiceStore.defaultVoice(this)
        for ((i, v) in list.withIndex()) {
            voices.addView(RadioButton(this).apply {
                text = VoiceStore.label(this@ShareActivity, v.name)
                id = i
                isChecked = v.name == def?.name
            })
        }
        if (voices.checkedRadioButtonId == -1) voices.check(0)
    }

    private fun speak(save: Boolean, live: Boolean) {
        val text = editor.text.toString().trim()
        if (text.isBlank()) { status.text = "Nothing to read"; return }
        val voice = voiceList().getOrNull(voices.checkedRadioButtonId) ?: return
        VoiceStore.setDefaultFor(this, voice, VoiceStore.isStyle(this, voice.name))
        val title = cleaned?.title ?: "Shared text"
        startForegroundService(
            Intent(this, TtsService::class.java)
                .setAction(TtsService.ACTION_SPEAK)
                .putExtra(TtsService.EXTRA_TEXT, text)
                .putExtra(TtsService.EXTRA_TITLE, title)
                .putExtra(TtsService.EXTRA_VOICE, voice.name)
                .putExtra(TtsService.EXTRA_ENGINE,
                    if (voice.file.extension.equals("json", true)) "supertonic" else "qwen")
                .putExtra(TtsService.EXTRA_BACKEND,
                    Backends.current(this, VoiceStore.engineOf(this, voice.name)))
                .putExtra(TtsService.EXTRA_SAVE, save))
        // "Play live" hands the reader a transport instead of dropping them
        // back into the page they shared from with audio coming out of nowhere
        if (live) {
            startActivity(Intent(this, PlayerActivity::class.java)
                .putExtra(PlayerActivity.EXTRA_TITLE, title)
                .putExtra(PlayerActivity.EXTRA_TEXT, text)
                .putExtra(PlayerActivity.EXTRA_VOICE, voice.name)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        }
        finish()
    }

    companion object { private const val PROGRESS_ID = 0x7f0a0001 }
}
