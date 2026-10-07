package com.techhurts.ttsrunner

import android.content.Context
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import android.widget.Toast
import com.google.android.material.dialog.MaterialAlertDialogBuilder

/** Shared composition controls. The selected voice determines the visible controls. */
class SpeechControls(context: Context, private val editor: EditText) : LinearLayout(context) {
    private var engine = ""
    private var refreshSpeed: (() -> Unit)? = null
    init { orientation = VERTICAL; visibility = GONE }

    override fun onWindowFocusChanged(hasWindowFocus: Boolean) {
        super.onWindowFocusChanged(hasWindowFocus)
        // Another editor can change the shared speed while this screen is paused.
        if (hasWindowFocus) refreshSpeed?.invoke()
    }

    fun setEngine(value: String) {
        if (engine == value) return
        engine = value
        refreshSpeed = null
        removeAllViews()
        visibility = if (engine in listOf("kitten", "supertonic")) VISIBLE else GONE
        if (visibility == GONE) return
        addView(TextView(context).apply {
            text = context.getString(R.string.delivery_for_engine, ModelManager.engineLabel(engine))
            textSize = 14f
        })
        val row = LinearLayout(context).apply { orientation = HORIZONTAL }
        fun action(label: Int, block: () -> Unit) {
            row.addView(Button(context).apply {
                text = context.getString(label); contentDescription = text
                setOnClickListener { block() }
            })
        }
        if (engine == "kitten") action(R.string.delivery_emotion) {
            val names = listOf(context.getString(R.string.delivery_neutral)) + SpeechMarkup.kittenEmotions
            MaterialAlertDialogBuilder(context).setTitle(R.string.delivery_emotion)
                .setItems(names.toTypedArray()) { _, index ->
                    editor.setText(SpeechMarkup.withEmotion(editor.text.toString(),
                        SpeechMarkup.kittenEmotions.getOrNull(index - 1)))
                    editor.setSelection(editor.length()); editor.requestFocus()
                }.show()
        }
        action(R.string.delivery_event) {
            val tags = if (engine == "kitten") SpeechMarkup.kittenEvents else SpeechMarkup.supertonicEvents
            MaterialAlertDialogBuilder(context).setTitle(R.string.delivery_event)
                .setItems(tags.map { "<$it>" }.toTypedArray()) { _, index -> insert("<${tags[index]}>") }.show()
        }
        if (engine == "kitten") action(R.string.delivery_emphasis) {
            val a = editor.selectionStart.coerceAtLeast(0)
            val b = editor.selectionEnd.coerceAtLeast(0)
            if (a != b) emphasize(editor.text.substring(minOf(a, b), maxOf(a, b)))
            else {
                val phrase = EditText(context).apply { hint = context.getString(R.string.delivery_phrase); contentDescription = hint }
                MaterialAlertDialogBuilder(context).setTitle(R.string.delivery_emphasis).setView(phrase)
                    .setPositiveButton(R.string.delivery_insert) { _, _ -> emphasize(phrase.text.toString()) }
                    .setNegativeButton(android.R.string.cancel, null).show()
            }
        }
        addView(HorizontalScrollView(context).apply { isHorizontalScrollBarEnabled = false; addView(row) })
        addView(TextView(context).apply {
            text = context.getString(if (engine == "kitten") R.string.delivery_kitten_help else R.string.delivery_supertonic_help)
            textSize = 12f
        })
        if (engine == "supertonic") addSpeed()
    }

    private fun emphasize(phrase: String) {
        if (phrase.isBlank() || phrase.length > 80 || phrase.any { it == '(' || it == ')' || it == '\n' }) {
            Toast.makeText(context, R.string.delivery_phrase_invalid, Toast.LENGTH_LONG).show()
            return
        }
        insert("((($phrase)))")
    }

    private fun insert(markup: String) {
        val a = editor.selectionStart.coerceAtLeast(0)
        val b = editor.selectionEnd.coerceAtLeast(0)
        val start = minOf(a, b); val end = maxOf(a, b)
        val text = editor.text
        val before = if (start > 0 && !text[start - 1].isWhitespace()) " " else ""
        val after = if (end == text.length || !text[end].isWhitespace()) " " else ""
        text.replace(start, end, before + markup + after)
        editor.setSelection(start + before.length + markup.length + after.length)
        editor.requestFocus()
    }

    private fun addSpeed() {
        val prefs = context.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
        val label = TextView(context)
        fun show(pct: Int) { label.text = context.getString(R.string.delivery_speed, pct / 100f) }
        addView(label)
        val slider = SeekBar(context).apply {
            contentDescription = context.getString(R.string.delivery_speed_control)
            max = 20
            progress = (prefs.getInt("speech_speed_pct", 100) - 60) / 5
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(bar: SeekBar?, value: Int, fromUser: Boolean) {
                    val pct = 60 + value * 5
                    if (fromUser) prefs.edit().putInt("speech_speed_pct", pct).apply()
                    show(pct)
                }
                override fun onStartTrackingTouch(bar: SeekBar?) {}
                override fun onStopTrackingTouch(bar: SeekBar?) {}
            })
        }
        addView(slider)
        refreshSpeed = {
            val pct = prefs.getInt("speech_speed_pct", 100)
            slider.progress = (pct - 60) / 5
            show(pct)
        }
        refreshSpeed?.invoke()
    }
}
