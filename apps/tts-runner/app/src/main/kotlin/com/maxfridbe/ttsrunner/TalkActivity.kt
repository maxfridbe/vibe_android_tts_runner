package com.maxfridbe.ttsrunner

import android.content.BroadcastReceiver
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
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.DynamicColors
import java.io.File

/** Live conversation mode: type a line, press enter, hear it immediately.
 *
 *  Only worthwhile with Supertonic — at RTF ~0.5 a sentence starts speaking
 *  in about a second, where the 1.7B model would take half a minute. Lines
 *  are spoken through the same engine service (so the model stays warm
 *  between lines) but marked ephemeral, keeping them out of the job history:
 *  a chat is not something you re-run or save. */
class TalkActivity : AppCompatActivity() {

    private lateinit var transcript: LinearLayout
    private lateinit var scroll: ScrollView
    private lateinit var input: EditText
    private lateinit var status: TextView
    private var voice: String? = null
    private val queued = ArrayDeque<String>()
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
        col.addView(TextView(this).apply {
            text = "Talk"
            textSize = 24f; setTypeface(typeface, Typeface.BOLD)
        })
        status = TextView(this).apply {
            textSize = 12f; alpha = 0.7f; text = "ready"
            setPadding(0, 0, 0, dp(8))
        }
        col.addView(status)

        transcript = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        scroll = ScrollView(this).apply { addView(transcript) }
        col.addView(scroll, LinearLayout.LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))

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
        voice = VoiceStore.styleList(this).firstOrNull()?.name
        if (voice == null) {
            Toast.makeText(this, "No style voice — download Supertonic in Settings", Toast.LENGTH_LONG).show()
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

    private fun say() {
        val text = input.text.toString().trim()
        if (text.isEmpty()) return
        input.setText("")
        addBubble(text)
        queued.addLast(text)
        pump()
    }

    /** One line at a time: the engine takes over on a new request, so queuing
     *  here is what keeps a fast typist from cutting off their own sentence. */
    private fun pump() {
        if (speaking) return
        val next = queued.removeFirstOrNull() ?: return
        val v = voice ?: VoiceStore.styleList(this).firstOrNull()?.name ?: return
        speaking = true
        status.text = "speaking…"
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, next)
            .putExtra(TtsService.EXTRA_TITLE, "Talk")
            .putExtra(TtsService.EXTRA_VOICE, v)
            .putExtra(TtsService.EXTRA_BACKEND, prefs().getString("backend", "cpu"))
            .putExtra(TtsService.EXTRA_EPHEMERAL, true)
            .putExtra(TtsService.EXTRA_SAVE, false))
    }

    private fun addBubble(text: String) {
        transcript.addView(MaterialCardView(this).apply {
            radius = dp(16).toFloat(); cardElevation = 0f; strokeWidth = 1
            setCardBackgroundColor(themeColor(com.google.android.material.R.attr.colorSecondaryContainer))
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
