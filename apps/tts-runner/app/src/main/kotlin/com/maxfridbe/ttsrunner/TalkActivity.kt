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
import android.view.View
import android.view.ViewGroup
import android.view.inputmethod.EditorInfo
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.DynamicColors
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import java.io.File
import kotlin.concurrent.thread

/** Live conversation mode: type a line, press enter, hear it immediately.
 *
 *  Only worthwhile with Supertonic — at RTF ~0.5 a sentence starts speaking in
 *  about a second, where the 1.7B model would take half a minute.
 *
 *  The transcript is a timeline, not a log: every line keeps the voice that
 *  said it and its own audio file, lines are dragged into the order you want
 *  (long press and move), and the whole thing replays or exports as one track
 *  in that order. Lines are spoken through the same engine service, so the
 *  model stays warm, but marked ephemeral so a chat never lands in the job
 *  history. */
class TalkActivity : AppCompatActivity() {

    /** One turn of the conversation. [audio] arrives when the engine finishes.
     *  [fast] records which engine spoke it: a Supertonic style answers in
     *  about a second, a Qwen reference voice takes tens of them, and that is
     *  the single most useful thing to know about a speaker before you pick it. */
    private class Line(var text: String, var voice: String, val fast: Boolean) {
        var audio: File? = null
        var secs: Double = 0.0
        var speaking = false
    }

    private lateinit var chat: ChatStore.Chat
    private val lines = mutableListOf<Line>()
    private lateinit var list: androidx.recyclerview.widget.RecyclerView
    private lateinit var adapter: LineAdapter
    private lateinit var titleView: TextView
    private lateinit var input: EditText
    private lateinit var status: TextView
    private lateinit var totals: TextView
    private lateinit var voiceSpinner: Spinner
    private var voice: String? = null
    private val queue = ArrayDeque<Line>()
    @Volatile private var busy: Line? = null
    private var player: android.media.MediaPlayer? = null

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()
    private fun prefs() = getSharedPreferences("ttsrunner", MODE_PRIVATE)
    private fun themeColor(attr: Int): Int {
        val tv = android.util.TypedValue()
        theme.resolveAttribute(attr, tv, true)
        return if (tv.resourceId != 0) getColor(tv.resourceId) else tv.data
    }

    private fun clipDir() = ChatStore.dir(this, chat.id)

    /** Writes the conversation back after anything that changes it. Cheap —
     *  one small JSON — and it means the chat list is never stale and a chat
     *  survives the activity being torn down mid-sentence. */
    private fun persist() {
        chat.lines.clear()
        lines.mapTo(chat.lines) { ChatStore.Line(it.text, it.voice, it.fast, it.audio, it.secs) }
        ChatStore.save(this, chat)
    }

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, i: Intent?) {
            val line = busy
            when (i?.getStringExtra("state")) {
                "loading" -> status.text = "warming up the model…"
                "frames", "generating" -> status.text = "speaking…"
                "done", "saved" -> {
                    if (line != null) {
                        val out = File(i.getStringExtra(TtsService.EXTRA_OUT) ?: "")
                        if (out.exists()) { line.audio = out; line.secs = Wav.seconds(out) }
                        line.speaking = false
                    }
                    busy = null
                    status.text = "ready"
                    persist()
                    render()
                    pump()
                }
                "error" -> {
                    line?.speaking = false
                    busy = null
                    status.text = "error: ${i.getStringExtra("message")}"
                    render()
                    pump()
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        DynamicColors.applyToActivityIfAvailable(this)
        val id = intent.getStringExtra(EXTRA_CHAT_ID)
        chat = (id?.let { ChatStore.load(this, it) })
            ?: ChatStore.create(this, ChatStore.defaultName(this))
        chat.lines.mapTo(lines) { Line(it.text, it.voice, it.fast).apply {
            audio = it.audio; secs = it.secs
        } }

        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(12))
        }
        val head = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        titleView = TextView(this).apply {
            text = chat.name
            textSize = 24f; setTypeface(typeface, Typeface.BOLD)
            maxLines = 1
            ellipsize = android.text.TextUtils.TruncateAt.END
            setOnClickListener { renameDialog() }
        }
        head.addView(titleView, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        head.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_play)
            contentDescription = "Replay the whole conversation"
            background = null
            setOnClickListener { replayAll() }
        })
        head.addView(ImageButton(this).apply {
            setImageResource(R.drawable.ic_more)
            contentDescription = "Conversation actions"
            background = null
            setOnClickListener { conversationMenu(this) }
        })
        col.addView(head)

        // Who speaks the next line. Changing it mid-conversation is the point:
        // that is what turns the transcript into a two-hander.
        val voiceRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
        }
        voiceRow.addView(TextView(this).apply {
            text = "Speaker"; textSize = 12f; alpha = 0.7f
            setPadding(0, 0, dp(10), 0)
        })
        voiceSpinner = Spinner(this)
        voiceRow.addView(voiceSpinner, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        col.addView(voiceRow)

        status = TextView(this).apply { textSize = 12f; alpha = 0.7f; text = "ready" }
        col.addView(status)
        totals = TextView(this).apply { textSize = 12f; alpha = 0.7f; setPadding(0, 0, 0, dp(8)) }
        col.addView(totals)

        adapter = LineAdapter()
        list = androidx.recyclerview.widget.RecyclerView(this).apply {
            layoutManager = androidx.recyclerview.widget.LinearLayoutManager(this@TalkActivity)
            this.adapter = this@TalkActivity.adapter
            // dragging past the edge should carry on scrolling, which is the
            // whole point of moving a line several places
            isNestedScrollingEnabled = true
        }
        dragger.attachToRecyclerView(list)
        col.addView(list, LinearLayout.LayoutParams(
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

        if (ModelManager.selectedModel(this)?.engine != "supertonic") {
            Toast.makeText(this,
                "Talk needs Supertonic (Settings → Model) — the 1.7B model is far too slow for live speech",
                Toast.LENGTH_LONG).show()
        }
        setupVoices(intent.getStringExtra(EXTRA_VOICE))
        render()
    }

    private fun supertonicSelected() = ModelManager.selectedModel(this)?.engine == "supertonic"

    /** Speakers the loaded model can actually use. */
    private fun voicesForModel(): List<String> =
        if (supertonicSelected()) VoiceStore.styleList(this).map { it.name }
        else VoiceStore.list(this).map { it.name }

    private fun setupVoices(preferred: String?) {
        val fast = supertonicSelected()
        val names = voicesForModel()
        if (names.isEmpty()) {
            Toast.makeText(this, "No style voice — download Supertonic in Settings", Toast.LENGTH_LONG).show()
            return
        }
        voice = preferred?.takeIf { it in names }
            ?: VoiceStore.defaultFor(this, fast)?.name?.takeIf { it in names }
            ?: names.first()
        // ⚡ answers in about a second, 🐢 takes tens of them — the one thing
        // worth knowing before choosing who speaks next
        val labels = names.map { VoiceStore.label(this, it, fast) }
        voiceSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        voiceSpinner.setSelection(names.indexOf(voice).coerceAtLeast(0))
        voiceSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                voice = names.getOrNull(pos)
                voice?.let {
                    VoiceStore.setDefaultFor(this@TalkActivity, VoiceStore.Voice(it, File("")), fast)
                }
            }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }
    }

    override fun onResume() {
        super.onResume()
        registerReceiver(statusReceiver, IntentFilter(TtsService.STATUS_BROADCAST), RECEIVER_NOT_EXPORTED)
    }

    override fun onDestroy() {
        stopPlayback()
        super.onDestroy()
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
        add(text)
    }

    /** Adds a line at the end and queues it. Order is a separate act: say it,
     *  then drag it where it belongs. */
    private fun add(text: String) {
        val v = voice ?: voicesForModel().firstOrNull() ?: return
        val line = Line(text, v, fast = supertonicSelected())
        lines.add(line)
        persist()
        render()
        list.post { list.scrollToPosition(lines.size - 1) }
        queue.addLast(line)
        pump()
    }

    /** One line at a time: the engine takes over on a new request, so queuing
     *  here is what keeps a fast typist from cutting off their own sentence. */
    private fun pump() {
        if (busy != null) return
        val next = queue.removeFirstOrNull() ?: return
        busy = next
        next.speaking = true
        status.text = "speaking…"
        render()
        speak(next, save = false)
    }

    private fun speak(line: Line, save: Boolean) {
        val out = File(clipDir(), "line_${System.nanoTime()}.wav")
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, line.text)
            .putExtra(TtsService.EXTRA_TITLE, if (save) line.text.take(40) else "Talk")
            .putExtra(TtsService.EXTRA_VOICE, line.voice)
            .putExtra(TtsService.EXTRA_BACKEND, Backends.current(this))
            .putExtra(TtsService.EXTRA_EPHEMERAL, !save)
            .putExtra(TtsService.EXTRA_OUT, out.absolutePath)
            .putExtra(TtsService.EXTRA_SAVE, save))
    }

    // ---- the transcript ----------------------------------------------------

    private fun render() {
        adapter.notifyDataSetChanged()
        val secs = lines.sumOf { it.secs }
        val pending = lines.count { it.audio == null }
        totals.text = when {
            lines.isEmpty() -> "nothing said yet · hold a line to drag it"
            else -> "${lines.size} line${if (lines.size == 1) "" else "s"} · ${Wav.fmt(secs)}" +
                if (pending > 0) " · $pending still generating" else ""
        }
    }

    /** Long press and move. Reordering is how a chat becomes a script: say the
     *  lines in any order, then put them where they belong — the audio travels
     *  with the line, so replay and export follow the new order without
     *  regenerating anything. */
    private val dragger = androidx.recyclerview.widget.ItemTouchHelper(
        object : androidx.recyclerview.widget.ItemTouchHelper.SimpleCallback(
            androidx.recyclerview.widget.ItemTouchHelper.UP or
                androidx.recyclerview.widget.ItemTouchHelper.DOWN, 0) {

            override fun onMove(
                rv: androidx.recyclerview.widget.RecyclerView,
                from: androidx.recyclerview.widget.RecyclerView.ViewHolder,
                to: androidx.recyclerview.widget.RecyclerView.ViewHolder,
            ): Boolean {
                val a = from.adapterPosition
                val b = to.adapterPosition
                if (a !in lines.indices || b !in lines.indices) return false
                lines.add(b, lines.removeAt(a))
                adapter.notifyItemMoved(a, b)
                persist()
                return true
            }

            override fun onSwiped(vh: androidx.recyclerview.widget.RecyclerView.ViewHolder, dir: Int) {}

            override fun isLongPressDragEnabled() = true

            // lift the card while it travels, so it reads as picked up
            override fun onSelectedChanged(
                vh: androidx.recyclerview.widget.RecyclerView.ViewHolder?, state: Int,
            ) {
                super.onSelectedChanged(vh, state)
                if (state == androidx.recyclerview.widget.ItemTouchHelper.ACTION_STATE_DRAG) {
                    (vh?.itemView as? MaterialCardView)?.cardElevation = dp(8).toFloat()
                }
            }

            override fun clearView(
                rv: androidx.recyclerview.widget.RecyclerView,
                vh: androidx.recyclerview.widget.RecyclerView.ViewHolder,
            ) {
                super.clearView(rv, vh)
                (vh.itemView as? MaterialCardView)?.cardElevation = 0f
                // positions changed, so the numbering in the menus follows
                adapter.notifyDataSetChanged()
            }
        })

    private inner class Holder(val card: MaterialCardView) :
        androidx.recyclerview.widget.RecyclerView.ViewHolder(card) {
        val who: TextView = TextView(this@TalkActivity).apply {
            textSize = 12f; alpha = 0.8f; setTypeface(typeface, Typeface.BOLD)
        }
        val meta: TextView = TextView(this@TalkActivity).apply { textSize = 11f; alpha = 0.6f }
        val play: ImageButton = ImageButton(this@TalkActivity).apply {
            setImageResource(R.drawable.ic_play)
            contentDescription = "Play this line"
            background = null
            minimumWidth = dp(36)
            setPadding(dp(8), dp(4), dp(4), dp(4))
        }
        val body: TextView = TextView(this@TalkActivity).apply {
            textSize = 16f
            setPadding(0, dp(4), 0, 0)
        }

        init {
            val colum = LinearLayout(this@TalkActivity).apply {
                orientation = LinearLayout.VERTICAL
                setPadding(dp(14), dp(10), dp(10), dp(10))
            }
            val top = LinearLayout(this@TalkActivity).apply {
                orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            }
            top.addView(who, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            top.addView(meta)
            top.addView(play)
            // the handle is the affordance; the whole card drags on long press
            top.addView(TextView(this@TalkActivity).apply {
                text = "≡"; textSize = 16f; alpha = 0.45f
                setPadding(dp(6), 0, dp(2), 0)
            })
            colum.addView(top)
            colum.addView(body)
            card.addView(colum)
        }
    }

    private inner class LineAdapter :
        androidx.recyclerview.widget.RecyclerView.Adapter<Holder>() {

        override fun getItemCount() = lines.size

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): Holder {
            val card = MaterialCardView(this@TalkActivity).apply {
                radius = dp(16).toFloat(); cardElevation = 0f; strokeWidth = 1
                isClickable = true
                layoutParams = androidx.recyclerview.widget.RecyclerView.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = dp(8) }
            }
            return Holder(card)
        }

        override fun onBindViewHolder(h: Holder, position: Int) {
            val line = lines[position]
            h.card.setCardBackgroundColor(themeColor(
                if (position % 2 == 0) com.google.android.material.R.attr.colorSecondaryContainer
                else com.google.android.material.R.attr.colorTertiaryContainer))
            h.who.text = VoiceStore.label(this@TalkActivity, line.voice, line.fast)
            h.meta.text = when {
                line.speaking -> "speaking…"
                line.audio != null -> Wav.fmt(line.secs)
                else -> "queued"
            }
            h.body.text = line.text
            h.play.visibility = if (line.audio != null) View.VISIBLE else View.GONE
            h.play.setOnClickListener { line.audio?.let { f -> playFiles(listOf(f)) } }
            h.card.setOnClickListener { lineMenu(h.card, h.adapterPosition) }
        }
    }

    // ---- playback ----------------------------------------------------------

    private fun stopPlayback() {
        player?.let { runCatching { it.stop() }; it.release() }
        player = null
    }

    /** Plays a list of clips back to back. Used for one line and for the
     *  whole conversation, which is the same thing with a longer list. */
    private fun playFiles(files: List<File>) {
        stopPlayback()
        val queueF = ArrayDeque(files.filter { it.exists() })
        if (queueF.isEmpty()) { toast("No audio for that yet"); return }
        fun next() {
            val f = queueF.removeFirstOrNull() ?: run { stopPlayback(); status.text = "ready"; return }
            player = android.media.MediaPlayer().apply {
                setDataSource(f.absolutePath)
                setOnCompletionListener { next() }
                setOnErrorListener { _, _, _ -> next(); true }
                prepare(); start()
            }
        }
        status.text = "playing…"
        next()
    }

    private fun replayAll() {
        val ready = lines.mapNotNull { it.audio }
        if (ready.isEmpty()) { toast("Nothing generated yet"); return }
        toast("Replaying ${ready.size} line${if (ready.size == 1) "" else "s"} · ${Wav.fmt(lines.sumOf { it.secs })}")
        playFiles(ready)
    }

    // ---- keeping what was said ---------------------------------------------

    private fun copy(text: String, what: String) {
        (getSystemService(CLIPBOARD_SERVICE) as ClipboardManager)
            .setPrimaryClip(ClipData.newPlainText("TTS Runner", text))
        toast("$what copied")
    }

    private fun shareText(text: String) {
        startActivity(Intent.createChooser(
            Intent(Intent.ACTION_SEND).setType("text/plain")
                .putExtra(Intent.EXTRA_TEXT, text), "Share text"))
    }

    /** Plain reading order, one speaker per line. */
    private fun asText() = lines.joinToString("\n") { it.text }

    /** The same conversation with its speakers, which is what makes it a chat
     *  rather than a paragraph. */
    private fun asChat() = lines.joinToString("\n") { "${it.voice}: ${it.text}" }

    private fun shareAudio(files: List<File>, title: String) {
        val ready = files.filter { it.exists() }
        if (ready.isEmpty()) { toast("No audio to share yet"); return }
        toast("Preparing audio…")
        thread {
            runCatching {
                val joined = File(cacheDir, "talk-share.wav")
                Wav.join(joined, ready) ?: error("nothing readable")
                AudioShare.shareWavAsM4a(this, joined, title)
            }.onFailure { e ->
                runOnUiThread { toast("Share failed: ${e.message}") }
            }
        }
    }

    private fun saveAudio(files: List<File>, title: String) {
        val ready = files.filter { it.exists() }
        if (ready.isEmpty()) { toast("No audio to save yet"); return }
        toast("Saving…")
        thread {
            val result = runCatching {
                val joined = File(cacheDir, "talk-save.wav")
                val clip = Wav.join(joined, ready)?.let { Wav.read(it) } ?: error("nothing readable")
                val saver = AudioSaver(this, title, clip.rate)
                saver.write(clip.pcm)
                saver.finish()
            }
            runOnUiThread {
                result.onSuccess { toast("Saved to $it") }
                    .onFailure { toast("Save failed: ${it.message}") }
            }
        }
    }

    private var pendingSave: Pair<String, String>? = null   // filename, content

    private fun saveTextToFile(name: String, content: String) {
        if (content.isBlank()) { toast("Nothing said yet"); return }
        pendingSave = name to content
        startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT)
            .addCategory(Intent.CATEGORY_OPENABLE)
            .setType("text/plain")
            .putExtra(Intent.EXTRA_TITLE, name), REQ_SAVE_TEXT)
    }

    private fun lineMenu(anchor: View, index: Int) {
        val line = lines.getOrNull(index) ?: return
        PopupMenu(this, anchor).apply {
            menu.add("Play")
            menu.add("Say again")
            menu.add("Copy text")
            menu.add("Share text")
            menu.add("Save text")
            menu.add("Share audio")
            menu.add("Save audio")
            menu.add("Delete")
            setOnMenuItemClickListener {
                when (it.title) {
                    "Play" -> line.audio?.let { f -> playFiles(listOf(f)) } ?: toast("Not generated yet")
                    "Say again" -> { queue.addLast(line); line.speaking = true; render(); pump() }
                    "Copy text" -> copy(line.text, "Line")
                    "Share text" -> shareText(line.text)
                    "Save text" -> saveTextToFile("line.txt", line.text)
                    "Share audio" -> shareAudio(listOfNotNull(line.audio), line.text.take(40))
                    "Save audio" -> saveAudio(listOfNotNull(line.audio), line.text.take(40))
                    "Delete" -> { lines.removeAt(index); persist(); render() }
                }
                true
            }
            show()
        }
    }

    private fun conversationMenu(anchor: View) {
        PopupMenu(this, anchor).apply {
            menu.add("Rename chat")
            menu.add("Close chat")
            menu.add("Replay all")
            menu.add("Share as one track")
            menu.add("Save as one track")
            menu.add("Copy chat")
            menu.add("Share chat")
            menu.add("Save chat")
            menu.add("Save text only")
            menu.add("Clear")
            setOnMenuItemClickListener {
                val audio = lines.mapNotNull { l -> l.audio }
                when (it.title) {
                    "Rename chat" -> renameDialog()
                    "Close chat" -> { persist(); finish() }
                    "Replay all" -> replayAll()
                    "Share as one track" -> shareAudio(audio, "Talk")
                    "Save as one track" -> saveAudio(audio, "Talk")
                    "Copy chat" -> if (lines.isEmpty()) toast("Nothing said yet") else copy(asChat(), "Conversation")
                    "Share chat" -> if (lines.isEmpty()) toast("Nothing said yet") else shareText(asChat())
                    "Save chat" -> saveTextToFile("talk-chat.txt", asChat())
                    "Save text only" -> saveTextToFile("talk.txt", asText())
                    "Clear" -> {
                        stopPlayback()
                        lines.clear(); queue.clear()
                        clipDir().listFiles()?.filter { f -> f.extension == "wav" }?.forEach { f -> f.delete() }
                        persist(); render()
                    }
                }
                true
            }
            show()
        }
    }

    private fun renameDialog() {
        val edit = EditText(this).apply {
            setText(chat.name); setSingleLine()
            setPadding(dp(20), dp(12), dp(20), dp(12))
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("Chat name")
            .setView(edit)
            .setPositiveButton("Rename") { _, _ ->
                chat.name = edit.text.toString().trim().ifBlank { chat.name }
                titleView.text = chat.name
                persist()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    override fun onPause() {
        super.onPause()
        runCatching { unregisterReceiver(statusReceiver) }
        persist()
    }

    private fun toast(t: String) = Toast.makeText(this, t, Toast.LENGTH_SHORT).show()

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_SAVE_TEXT && resultCode == RESULT_OK && data?.data != null) {
            val content = pendingSave?.second ?: return
            runCatching {
                contentResolver.openOutputStream(data.data!!)!!.use { it.write(content.toByteArray()) }
            }.onSuccess { toast("Saved") }.onFailure { toast("Save failed: ${it.message}") }
            pendingSave = null
        }
    }

    companion object {
        const val EXTRA_VOICE = "voice"
        const val EXTRA_CHAT_ID = "chat_id"
        private const val REQ_SAVE_TEXT = 21

        /** Supertonic 3's expression tags. The model card says ten exist but
         *  does not list them, so they were found by probe: synthesise
         *  "I see. <x> Well then." and transcribe it — a real tag is consumed
         *  as a vocalisation, anything else is read out as a word
         *  (`<chuckle>` and `<gasp>` are spoken, so they are not tags).
         *  `<laughter>` and `<breathe>` are accepted spellings of two of them.
         *  One tag adds ~0.2 s; repeats stack (three `<laugh>` ran +0.9 s). */
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
}
