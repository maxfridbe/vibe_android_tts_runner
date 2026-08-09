package com.maxfridbe.ttsrunner

import android.content.BroadcastReceiver
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.DynamicColors

/** Live conversation mode: type a line, press enter, hear it immediately.
 *
 *  Only worthwhile with Supertonic — at RTF ~0.5 a sentence starts speaking
 *  in about a second, where the 1.7B model would take half a minute. Lines
 *  are spoken through the same engine service (so the model stays warm
 *  between lines) but marked ephemeral, keeping them out of the job history:
 *  a chat is not something you re-run or save — unless you ask, which is what
 *  the per-line menu and the transcript actions are for. */
class TalkActivity : AppCompatActivity() {

    private lateinit var transcript: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var input: EditText
    private lateinit var status: TextView
    private var voice: String? = null
    private val queued = ArrayDeque<String>()
    private val said = mutableListOf<String>()
    @Volatile private var speaking = false

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun prefs() = getSharedPreferences("ttsrunner", MODE_PRIVATE)
    private fun themeColor(attr: Int): Int {
        val tv = android.util.TypedValue()
        theme.resolveAttribute(attr, tv, true)
        return if (tv.resourceId != 0) getColor(tv.resourceId) else tv.data
    }

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, i: Intent?) {
            when (i?.getStringExtra("state")) {
                "loading" -> status.text = "warming up the model…"
                "frames", "generating" -> status.text = "speaking…"
                "done", "saved" -> { speaking = false; status.text = "ready"; pump() }
                "error" -> {
                    speaking = false
                    status.text = "error: ${i.getStringExtra("message")}"
                    pump()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        DynamicColors.applyToActivityIfAvailable(this)

        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(12))
        }
        val head = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        head.addView(TextView(this).apply {
            text = "Talk"
            textSize = 24f; setTypeface(typeface, Typeface.BOLD)
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        head.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_share)
            contentDescription = "Conversation actions"
            background = null
            setOnClickListener { conversationMenu(this) }
        })
        col.addView(head)
        status = TextView(this).apply {
            textSize = 12f; alpha = 0.7f; text = "ready"
            setPadding(0, 0, 0, dp(8))
        }
        col.addView(status)

        transcript = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        scroll = ScrollView(this).apply { addView(transcript) }
        col.addView(scroll, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

        // Supertonic reads bracketed expression tags inline, so a row of one-tap
        // inserts sits right above the keyboard where they are actually used.
        val tags = android.widget.HorizontalScrollView(this).apply { isHorizontalScrollBarEnabled = false }
        val tagRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for ((label, tag) in EXPRESSIONS) {
            tagRow.addView(Button(this).apply {
                text = label
                textSize = 12f
                minWidth = 0; minimumWidth = 0
                setPadding(dp(12), 0, dp(12), 0)
                setOnClickListener { insertTag(tag) }
                // repeats stack: measured +0.2 s for one <laugh> and +0.9 s for
                // three, so a long press is the "really laugh" gesture
                setOnLongClickListener { insertTag(tag.repeat(3)); true }
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, dp(38)
                ).apply { marginEnd = dp(6) }
            })
        }
        tags.addView(tagRow)
        col.addView(tags)
        col.addView(TextView(this).apply {
            text = "a tag adds about a fifth of a second — hold one for a long version"
            textSize = 11f; alpha = 0.6f; setPadding(0, dp(2), 0, dp(2))
        })

        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        input = EditText(this).apply {
            hint = "Type and press enter to speak"
            // single line + IME_ACTION_SEND is what makes enter mean "say it"
            setSingleLine()
            imeOptions = EditorInfo.IME_ACTION_SEND
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_SEND) { say(); true } else false
            }
        }
        row.addView(input, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        row.addView(Button(this).apply {
            text = "Say"
            setOnClickListener { say() }
        })
        col.addView(row)

        setContentView(col)

        val model = ModelManager.selectedModel(this)
        if (model?.engine != "supertonic") {
            Toast.makeText(this,
                "Talk needs Supertonic (Settings → Model) — the 1.7B model is far too slow for live speech",
                Toast.LENGTH_LONG).show()
        }
        voice = intent.getStringExtra(EXTRA_VOICE)
            ?: VoiceStore.defaultFor(this, true)?.name
        if (voice == null) {
            Toast.makeText(this, "No style voice — download Supertonic in Settings", Toast.LENGTH_LONG).show()
        } else {
            status.text = "ready · $voice"
        }
    }

    override fun onResume() {
        super.onResume()
        registerReceiver(statusReceiver, IntentFilter(TtsService.STATUS_BROADCAST), RECEIVER_NOT_EXPORTED)
    }

    override fun onPause() {
        super.onPause()
        runCatching { unregisterReceiver(statusReceiver) }
    }

    /** Drops a tag at the cursor (or over the selection) and keeps focus in the
     *  field, so tapping one never costs the keyboard or the caret position. */
    private fun insertTag(tag: String) {
        val start = input.selectionStart.coerceAtLeast(0)
        val end = input.selectionEnd.coerceAtLeast(0)
        val text = input.text
        val pad = if (start > 0 && text.getOrNull(start - 1)?.isWhitespace() == false) " $tag" else tag
        text.replace(minOf(start, end), maxOf(start, end), "$pad ")
        input.requestFocus()
    }

    private fun say() {
        val text = input.text.toString().trim()
        if (text.isEmpty()) return
        input.setText("")
        said.add(text)
        addBubble(text)
        queued.addLast(text)
        pump()
    }

    /** One line at a time: the engine takes over on a new request, so queuing
     *  here is what keeps a fast typist from cutting off their own sentence. */
    private fun pump() {
        if (speaking) return
        val next = queued.removeFirstOrNull() ?: return
        speaking = true
        status.text = "speaking…"
        speak(next, save = false)
    }

    private fun speak(text: String, save: Boolean) {
        val v = voice ?: VoiceStore.styleList(this).firstOrNull()?.name ?: return
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, text)
            .putExtra(TtsService.EXTRA_TITLE, if (save) text.take(40) else "Talk")
            .putExtra(TtsService.EXTRA_VOICE, v)
            .putExtra(TtsService.EXTRA_BACKEND, prefs().getString("backend", "cpu"))
            .putExtra(TtsService.EXTRA_EPHEMERAL, !save)
            .putExtra(TtsService.EXTRA_SAVE, save))
    }

    // ---- keeping what was said ---------------------------------------------

    private fun copy(text: String, what: String) {
        (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager)
            .setPrimaryClip(ClipData.newPlainText("TTS Runner", text))
        Toast.makeText(this, "$what copied", Toast.LENGTH_SHORT).show()
    }

    private fun shareText(text: String) {
        startActivity(Intent.createChooser(
            Intent(Intent.ACTION_SEND).setType("text/plain")
                .putExtra(Intent.EXTRA_TEXT, text), "Share text"))
    }

    /** Per-line menu. "Save audio" re-speaks the line as a normal saved job,
     *  which is what puts a file in Music/TTS Runner — live lines are
     *  deliberately not written to disk as they play. */
    private fun lineMenu(anchor: android.view.View, text: String) {
        PopupMenu(this, anchor).apply {
            menu.add("Copy")
            menu.add("Share")
            menu.add("Save audio")
            menu.add("Say again")
            setOnMenuItemClickListener {
                when (it.title) {
                    "Copy" -> copy(text, "Line")
                    "Share" -> shareText(text)
                    "Save audio" -> {
                        speak(text, save = true)
                        Toast.makeText(this@TalkActivity,
                            "Saving to Music/TTS Runner…", Toast.LENGTH_SHORT).show()
                    }
                    "Say again" -> { queued.addLast(text); pump() }
                }
                true
            }
            show()
        }
    }

    private fun conversationMenu(anchor: android.view.View) {
        PopupMenu(this, anchor).apply {
            menu.add("Copy conversation")
            menu.add("Share conversation")
            menu.add("Save conversation to file")
            menu.add("Clear")
            setOnMenuItemClickListener {
                val all = said.joinToString("\n")
                when (it.title) {
                    "Copy conversation" -> if (all.isBlank()) toast("Nothing said yet") else copy(all, "Conversation")
                    "Share conversation" -> if (all.isBlank()) toast("Nothing said yet") else shareText(all)
                    "Save conversation to file" ->
                        if (all.isBlank()) toast("Nothing said yet") else
                            startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT)
                                .addCategory(Intent.CATEGORY_OPENABLE)
                                .setType("text/plain")
                                .putExtra(Intent.EXTRA_TITLE, "talk.txt"), REQ_SAVE_TEXT)
                    "Clear" -> { said.clear(); transcript.removeAllViews() }
                }
                true
            }
            show()
        }
    }

    private fun toast(t: String) = Toast.makeText(this, t, Toast.LENGTH_SHORT).show()

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_SAVE_TEXT && resultCode == RESULT_OK && data?.data != null) {
            runCatching {
                contentResolver.openOutputStream(data.data!!)!!.use {
                    it.write(said.joinToString("\n").toByteArray())
                }
            }.onSuccess { toast("Conversation saved") }
                .onFailure { toast("Save failed: ${it.message}") }
        }
    }

    companion object {
        const val EXTRA_VOICE = "voice"
        private const val REQ_SAVE_TEXT = 21

        /** Supertonic 3's expression tags. The model card says ten exist but
         *  does not list them, so they were found by probe: synthesise
         *  "I see. <x> Well then." and transcribe it — a real tag is consumed
         *  as a vocalisation, anything else is read out as a word
         *  (`<chuckle>` and `<gasp>` are spoken, so they are not tags).
         *  `<laughter>` and `<breathe>` are accepted spellings of two of them.
         *  One tag adds ~0.2 s; repeats stack (three `<laugh>` ran +0.9 s),
         *  which is why tapping a chip twice is worth suggesting. */
        private val EXPRESSIONS = listOf(
            "😄 laugh" to "<laugh>",
            "😔 sigh" to "<sigh>",
            "😮‍💨 breath" to "<breath>",
            "😢 cry" to "<cry>",
            "🥱 yawn" to "<yawn>",
            "🤧 cough" to "<cough>",
            "🤔 hmm" to "<hmm>",
            "😐 um" to "<um>",
            "😒 tsk" to "<tsk>",
            "😘 kiss" to "<kiss>",
            "… pause" to "...",
            "— dash" to " —",
        )
    }

    private fun addBubble(text: String) {
        transcript.addView(MaterialCardView(this).apply {
            radius = dp(16).toFloat(); cardElevation = 0f; strokeWidth = 1
            setCardBackgroundColor(themeColor(com.google.android.material.R.attr.colorSecondaryContainer))
            isClickable = true
            setOnClickListener { lineMenu(this, text) }
            addView(TextView(this@TalkActivity).apply {
                this.text = text
                textSize = 16f
                setPadding(dp(14), dp(10), dp(14), dp(10))
            })
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(8) }
        })
        scroll.post { scroll.fullScroll(ScrollView.FOCUS_DOWN) }
    }
}
