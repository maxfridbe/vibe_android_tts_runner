package com.maxfridbe.ttsrunner

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.view.Gravity
import android.view.Menu
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.DynamicColors
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.navigationrail.NavigationRailView
import java.io.File
import kotlin.concurrent.thread
import kotlin.random.Random

/** Tabbed UI in the audiobook-maker mold: New job / Voices / Jobs / Settings.
 *  Bottom navigation on phones, navigation rail on tablets and the unfolded
 *  Fold (width >= 600dp). A live memory meter sits top-right. */
class MainActivity : AppCompatActivity() {

    private lateinit var content: FrameLayout
    private lateinit var memMeter: TextView
    private lateinit var statusPane: View
    private var navSelect: ((Int) -> Unit)? = null
    private val tabs = mutableMapOf<Int, View>()
    private var currentTab = TAB_JOBS
    private val tabOrder = listOf(TAB_JOBS, TAB_VOICES, TAB_CHATS, TAB_HOST, TAB_SETTINGS)

    // New job tab widgets (referenced from the status receiver)
    private lateinit var runStatus: TextView
    private lateinit var runProgress: ProgressBar
    private lateinit var etaText: TextView
    private lateinit var jobText: EditText
    private lateinit var stopBtn: Button
    private lateinit var voicePickBtn: Button
    private var pickedVoice: String? = null
    private var pickedSave = false

    private var player: android.media.MediaPlayer? = null
    private val ui = android.os.Handler(android.os.Looper.getMainLooper())

    private val engineConn = object : android.content.ServiceConnection {
        override fun onServiceConnected(n: android.content.ComponentName?, b: android.os.IBinder?) {}
        override fun onServiceDisconnected(n: android.content.ComponentName?) {}
    }

    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    // ---- status broadcasts -------------------------------------------------

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            intent ?: return
            val state = intent.getStringExtra("state")
            // Engine-side bookkeeping: it restarted and found a dead job. That
            // job's last broadcast was a busy one, so the pane is still showing
            // its progress — a ghost of work that stopped. Clear it and reload
            // the list.
            if (state == "jobs") {
                ui.removeCallbacks(hideStatusPane)
                statusPane.visibility = View.GONE
                runStatus.text = "Idle"
                runProgress.progress = 0
                etaText.text = ""
                stopBtn.visibility = View.GONE
                refreshJobsIfChanged(force = true)
                return
            }
            val chunk = intent.getIntExtra("chunk", 0)
            val total = intent.getIntExtra("total", 0)
            val message = intent.getStringExtra("message") ?: ""
            val permille = intent.getIntExtra("permille", -1)
            val eta = intent.getIntExtra("eta", -1)
            if (permille >= 0) runProgress.progress = permille
            val busy = state in setOf("loading", "generating", "frames")
            stopBtn.visibility = if (busy) View.VISIBLE else View.GONE
            etaText.text = if (eta >= 0) "≈ ${if (eta >= 60) "${eta / 60}m ${eta % 60}s" else "${eta}s"} remaining" else ""
            showStatusPane(busy)
            refreshJobsIfChanged()
            runStatus.text = when (state) {
                "loading" -> { runProgress.progress = 0; "Loading model…" }
                "generating" -> {
                    if (total > 0) runProgress.progress = chunk * 1000 / total
                    "Chunk ${chunk + 1}/$total: “$message…”"
                }
                "frames" ->
                    if (chunk == 0) "Processing text & voice…"
                    else "Generating… ${"%.1f".format(chunk / 12.5)} s of audio"
                "done" -> { finishUi(); "Done — $message" }
                "saved" -> { finishUi(); "Saved — $message" }
                "designed" -> { finishUi(); offerKeepDesignedVoice(); "Voice rolled" }
                "stopped" -> { runProgress.progress = 0; etaText.text = ""; "Stopped" }
                "note" -> message
                "error" -> { runProgress.progress = 0; etaText.text = ""; "Error: $message" }
                else -> state ?: ""
            }
            // the running job's card mirrors the pane, so the Jobs tab is a
            // complete view of the work on its own
            runningLine?.text = runStatus.text
            runningBar?.let { if (permille >= 0) it.progress = permille }
        }
    }

    private fun finishUi() {
        runProgress.progress = 1000
        etaText.text = ""
        previewing = null      // whatever was spinning has landed, one way or another
        if (tabs.containsKey(TAB_JOBS)) rebuildJobs()
        if (tabs.containsKey(TAB_VOICES)) rebuildVoices()
    }

    /** The pane is noise when the engine is idle: show it while a job runs,
     *  keep the outcome on screen briefly, then get out of the way. */
    private val hideStatusPane = Runnable { statusPane.visibility = View.GONE }

    private fun showStatusPane(busy: Boolean) {
        ui.removeCallbacks(hideStatusPane)
        statusPane.visibility = View.VISIBLE
        if (!busy) ui.postDelayed(hideStatusPane, 6000)
    }

    private fun themeColor(attr: Int): Int {
        val tv = android.util.TypedValue()
        theme.resolveAttribute(attr, tv, true)
        return if (tv.resourceId != 0) getColor(tv.resourceId) else tv.data
    }

    // ---- activity ----------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        DynamicColors.applyToActivityIfAvailable(this)
        currentTab = savedInstanceState?.getInt("tab") ?: TAB_JOBS

        if (android.os.Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) !=
                android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 12)
        }

        content = FrameLayout(this)
        memMeter = TextView(this).apply {
            textSize = 11f; alpha = 0.6f
            setPadding(dp(8), dp(6), dp(12), dp(4))
        }
        // persistent status pane: what the engine is doing right now, visible
        // from every tab (a job started from the share sheet or a re-run is
        // just as interesting as one started on the New job tab)
        statusPane = buildStatusPane().noStateSave()
        statusPane.visibility = View.GONE
        val contentWithMeter = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(FrameLayout(context).apply {
                addView(statusPane)
                addView(memMeter, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT,
                    Gravity.TOP or Gravity.END))
            }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(content, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        }

        val wide = resources.configuration.screenWidthDp >= 600
        val navItems = { menu: Menu ->
            menu.add(0, TAB_JOBS, 0, "Jobs").setIcon(R.drawable.ic_jobs)
            menu.add(0, TAB_VOICES, 1, "Speakers").setIcon(R.drawable.ic_mic)
            menu.add(0, TAB_CHATS, 2, "Chats").setIcon(R.drawable.ic_chat)
            menu.add(0, TAB_HOST, 3, "Hosting").setIcon(R.drawable.ic_host)
            menu.add(0, TAB_SETTINGS, 4, "Settings").setIcon(R.drawable.ic_settings)
        }

        val root: View = if (wide) {
            val rail = NavigationRailView(this)
            navItems(rail.menu)
            rail.setOnItemSelectedListener { showTab(it.itemId); true }
            rail.selectedItemId = currentTab
            navSelect = { id -> rail.selectedItemId = id }
            LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(rail, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT))
                addView(contentWithMeter, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f))
            }
        } else {
            val bottom = BottomNavigationView(this)
            navItems(bottom.menu)
            bottom.setOnItemSelectedListener { showTab(it.itemId); true }
            bottom.selectedItemId = currentTab
            navSelect = { id -> bottom.selectedItemId = id }
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                addView(contentWithMeter, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
                addView(bottom, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            }
        }
        setContentView(root)
        showTab(currentTab)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putInt("tab", currentTab)
    }

    /** These views are built in code with hand-assigned ids that repeat across
     *  tabs, so Android's view-state save/restore can hand a RadioButton's
     *  state to a different widget with the same id after a configuration
     *  change (ClassCastException in CompoundButton.onRestoreInstanceState —
     *  seen when unfolding). Every tab is rebuilt from app state anyway. */
    private fun View.noStateSave(): View {
        isSaveEnabled = false
        isSaveFromParentEnabled = false
        if (this is ViewGroup) for (i in 0 until childCount) getChildAt(i).noStateSave()
        return this
    }

    private fun showTab(id: Int) {
        currentTab = id
        // rebuilt each visit: Settings reads live permission state, Hosting
        // reads whether the server is up right now
        if (id == TAB_SETTINGS || id == TAB_HOST || id == TAB_CHATS) tabs.remove(id)
        val v = tabs.getOrPut(id) {
            when (id) {
                TAB_VOICES -> buildVoicesTab()
                TAB_CHATS -> buildChatsTab()
                TAB_JOBS -> buildJobsTab()
                TAB_HOST -> buildHostTab()
                else -> buildSettingsTab()
            }.noStateSave()
        }
        when (id) {
            // opening the tab is the natural moment to reconcile with the
            // backup folder: new speakers go out, a fresh install pulls back in
            TAB_VOICES -> { rebuildVoices(); syncSpeakerFolder(loud = false) }
            TAB_JOBS -> { rebuildJobs(); refreshVoiceLabel() }
        }
        content.removeAllViews()
        content.addView(v)
        v.noStateSave()
    }

    /** Switch tab and move the navigation highlight with it. */
    private fun selectTab(id: Int) {
        val nav = navSelect
        if (nav != null) nav(id) else showTab(id)
    }

    /** Keep the :engine process (and its loaded model) alive while visible. */
    override fun onStart() {
        super.onStart()
        bindService(Intent(this, TtsService::class.java), engineConn, Context.BIND_AUTO_CREATE)
    }

    override fun onStop() {
        super.onStop()
        runCatching { unbindService(engineConn) }
    }

    private val memTicker = object : Runnable {
        override fun run() {
            thread {
                val am = getSystemService(android.app.ActivityManager::class.java)
                val free = runCatching {
                    val mi = android.app.ActivityManager.MemoryInfo()
                    am.getMemoryInfo(mi)
                    "free ${"%.1f".format(mi.availMem / 1073741824.0)} GB"
                }.getOrDefault("")
                val engine = runCatching {
                    val pid = am.runningAppProcesses?.find { it.processName.endsWith(":engine") }?.pid
                    if (pid == null) "engine idle"
                    else {
                        // getProcessMemoryInfo works for our own processes;
                        // /proc/<pid> of a sibling is blocked by hidepid
                        val pss = am.getProcessMemoryInfo(intArrayOf(pid)).firstOrNull()?.totalPss ?: 0
                        if (pss > 0) "engine ${"%.1f".format(pss / 1048576.0)} GB" else "engine loaded"
                    }
                }.getOrDefault("")
                val text = listOf(engine, free).filter { it.isNotBlank() }.joinToString(" · ")
                runOnUiThread { memMeter.text = text }
            }
            ui.postDelayed(this, 2500)
        }
    }

    override fun onResume() {
        super.onResume()
        registerReceiver(statusReceiver, IntentFilter(TtsService.STATUS_BROADCAST), RECEIVER_NOT_EXPORTED)
        ui.post(memTicker)
        // a voice may have been recorded while we were paused (RecordActivity)
        if (currentTab == TAB_VOICES && tabs.containsKey(TAB_VOICES)) rebuildVoices()
        // and a chat has almost certainly changed: coming back from Talk is the
        // normal way to arrive here, so the list is re-read rather than cached
        if (currentTab == TAB_CHATS) selectTab(TAB_CHATS)
    }

    override fun onPause() {
        super.onPause()
        unregisterReceiver(statusReceiver)
        ui.removeCallbacks(memTicker)
    }

    override fun onDestroy() {
        stopPlayback()
        runningPulse?.cancel()
        ui.removeCallbacks(hideStatusPane)
        super.onDestroy()
    }

    // ---- shared view helpers ----------------------------------------------

    private fun page(): Pair<ScrollView, LinearLayout> {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(16), dp(20), dp(24))
        }
        val scroll = ScrollView(this).apply { isFillViewport = true; addView(col) }
        return scroll to col
    }

    private fun LinearLayout.title(t: String) = addView(TextView(context).apply {
        text = t; textSize = 24f; setTypeface(typeface, Typeface.BOLD)
        setPadding(0, 0, 0, dp(4))
    })

    private fun LinearLayout.caption(t: String) = addView(TextView(context).apply {
        text = t; textSize = 13f; alpha = 0.7f; setPadding(0, 0, 0, dp(12))
    })

    private fun card(build: LinearLayout.() -> Unit): MaterialCardView {
        val inner = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(12))
            build()
        }
        return MaterialCardView(this).apply {
            radius = dp(16).toFloat()
            cardElevation = 0f
            strokeWidth = 1
            addView(inner)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { bottomMargin = dp(10) }
        }
    }

    /** Compact icon button: rows carry up to five of these next to a name, so
     *  the default ImageButton padding would starve the label. */
    private fun iconBtn(icon: Int, desc: String, action: () -> Unit) = ImageButton(this).apply {
        setImageResource(icon)
        contentDescription = desc
        background = null
        minimumWidth = dp(40)
        setPadding(dp(9), dp(9), dp(9), dp(9))
        setOnClickListener { action() }
    }

    private fun toast(t: String) = Toast.makeText(this, t, Toast.LENGTH_LONG).show()

    private fun currentModelId() = ModelManager.selectedModel(this)?.id ?: "none"
    private fun prefs() = getSharedPreferences("ttsrunner", MODE_PRIVATE)

    // ---- New job ------------------------------------------------------------

    /** Composing a job lives at the top of the Jobs tab: it is the thing that
     *  creates the list below it, and Material's bottom navigation stops at
     *  five destinations — a tab of its own would have cost Chats or Hosting. */
    private fun newJobCard(): View {
        return card {
            jobText = EditText(context).apply {
                setText("The quick brown fox jumps over the lazy dog, then reads the entire internet aloud.")
                minLines = 3
                background = null
                hint = "Text to read"
            }
            addView(jobText)

            // voice and mode on separate lines: side by side, a long speaker
            // name squeezed "Save file" into one letter per line on a phone
            voicePickBtn = Button(context).apply { setOnClickListener { pickVoiceDialog() } }
            addView(voicePickBtn, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT))
            val modeGroup = RadioGroup(context).apply { orientation = LinearLayout.VERTICAL }
            modeGroup.addView(RadioButton(context).apply { text = "Listen"; id = 1; isChecked = true })
            modeGroup.addView(RadioButton(context).apply { text = "Save file"; id = 2 })
            modeGroup.setOnCheckedChangeListener { _, id -> pickedSave = id == 2 }
            addView(modeGroup)

            val actions = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL }
            // live mode lives on the Speakers tab now: you start it as a
            // particular speaker, which is a choice about the voice, not the job
            actions.addView(Button(context).apply {
                text = "Play live"
                setOnClickListener {
                    val text = jobText.text.toString()
                    if (text.isBlank()) { toast("Nothing to read"); return@setOnClickListener }
                    val name = pickedVoice ?: voicesForCurrentModel().firstOrNull()
                    if (startTtsJob(text, "Reading", save = false, voiceName = name)) {
                        startActivity(Intent(this@MainActivity, PlayerActivity::class.java)
                            .putExtra(PlayerActivity.EXTRA_TITLE, "Reading")
                            .putExtra(PlayerActivity.EXTRA_TEXT, text)
                            .putExtra(PlayerActivity.EXTRA_VOICE, name ?: ""))
                    }
                }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            actions.addView(Button(context).apply {
                text = "Add job"
                setOnClickListener {
                    val text = jobText.text.toString()
                    if (text.isBlank()) { toast("Nothing to read"); return@setOnClickListener }
                    val started = startTtsJob(text, if (pickedSave) "TTS " + text.take(24) else "TTS job",
                        save = pickedSave, voiceName = pickedVoice)
                    if (started) refreshJobsIfChanged()
                }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(actions)
        }
    }

    /** Current Status pane, pinned above the tabs. */
    private fun buildStatusPane(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(10), dp(20), dp(8))
        }
        row.addView(TextView(this).apply {
            text = "Current status"; textSize = 11f; alpha = 0.6f
        })
        val line = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        runStatus = TextView(this).apply { text = "Idle"; textSize = 15f }
        line.addView(runStatus, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        etaText = TextView(this).apply { textSize = 12f; alpha = 0.7f }
        line.addView(etaText)
        stopBtn = Button(this).apply {
            text = "Stop"
            visibility = View.GONE
            setOnClickListener {
                startService(Intent(this@MainActivity, TtsService::class.java).setAction(TtsService.ACTION_STOP))
            }
        }
        line.addView(stopBtn)
        row.addView(line)
        runProgress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 1000 }
        row.addView(runProgress)
        return row
    }

    private fun refreshVoiceLabel() {
        if (!::voicePickBtn.isInitialized) return
        val supertonic = ModelManager.selectedModel(this)?.engine == "supertonic"
        val name = pickedVoice?.takeIf { it in voicesForCurrentModel() }
            ?: VoiceStore.defaultFor(this, supertonic)?.name
            ?: voicesForCurrentModel().firstOrNull()
        voicePickBtn.text =
            if (name == null) "Voice: none ▾" else "${VoiceStore.label(this, name)} ▾"
    }

    /** "⚡ 🦊 Dale" labels for the picker dialogs, in list order. */
    private fun voiceLabels(names: List<String>) =
        names.map { VoiceStore.label(this, it) }.toTypedArray()

    /** Voices the selected model can actually speak with: Supertonic uses
     *  style files, Qwen uses reference recordings. */
    private fun voicesForCurrentModel(): List<String> {
        val supertonic = ModelManager.selectedModel(this)?.engine == "supertonic"
        return if (supertonic) VoiceStore.styleList(this).map { it.name }
               else VoiceStore.list(this).map { it.name }
    }

    private fun pickVoiceDialog() {
        val names = voicesForCurrentModel()
        if (names.isEmpty()) {
            toast(if (ModelManager.selectedModel(this)?.engine == "supertonic")
                      "No style voices — re-download Supertonic in Settings"
                  else "Import or design a voice first (Voices tab)")
            return
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("Voice for this job")
            .setItems(voiceLabels(names)) { _, which ->
                pickedVoice = names[which]
                refreshVoiceLabel()
            }
            .show()
    }

    private fun startTtsJob(text: String, title: String, save: Boolean, voiceName: String? = null): Boolean {
        val model = ModelManager.selectedModel(this)
        if (model == null) { toast("Download a model first (Settings tab)"); return false }
        if (model.engine == "supertonic") {
            val style = voiceName?.takeIf { VoiceStore.styleFile(this, it) != null }
                ?: VoiceStore.defaultFor(this, true)?.name
            if (style == null) { toast("No style voices — re-download Supertonic in Settings"); return false }
            startForegroundService(Intent(this, TtsService::class.java)
                .setAction(TtsService.ACTION_SPEAK)
                .putExtra(TtsService.EXTRA_TEXT, text)
                .putExtra(TtsService.EXTRA_TITLE, title)
                .putExtra(TtsService.EXTRA_VOICE, style)
                .putExtra(TtsService.EXTRA_BACKEND, Backends.current(this))
                .putExtra(TtsService.EXTRA_SAVE, save))
            ui.postDelayed({ refreshJobsIfChanged() }, 500)
            return true
        }
        val v = voiceName?.let { n -> VoiceStore.list(this).find { it.name == n } } ?: VoiceStore.defaultVoice(this)
        if (v == null) { toast("Import or design a voice first (Voices tab)"); return false }
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, text)
            .putExtra(TtsService.EXTRA_TITLE, title)
            .putExtra(TtsService.EXTRA_VOICE, v.name)
            .putExtra(TtsService.EXTRA_BACKEND, Backends.current(this))
            .putExtra(TtsService.EXTRA_SAVE, save))
        // the engine writes the "running" row from its own process; give it a
        // beat, then show it (broadcasts refresh the list from then on)
        ui.postDelayed({ refreshJobsIfChanged() }, 500)
        return true
    }

    // ---- audio playback helpers -------------------------------------------

    private fun lastAudio() = File(filesDir, "last_audio.wav")

    private var playingBtn: ImageButton? = null
    private var playingKey: String? = null

    private fun playFile(path: String, btn: ImageButton) = playSource(path, btn) { it.setDataSource(path) }
    private fun playUri(uri: Uri, btn: ImageButton) = playSource(uri.toString(), btn) { it.setDataSource(this, uri) }

    /** One player at a time; the button that started it shows a pause icon
     *  and toggles, every other play button resets to the play icon. */
    private fun playSource(key: String, btn: ImageButton, set: (android.media.MediaPlayer) -> Unit) {
        val p = player
        if (p != null && playingKey == key) {
            if (p.isPlaying) { p.pause(); btn.setImageResource(R.drawable.ic_play) }
            else { p.start(); btn.setImageResource(R.drawable.ic_pause) }
            return
        }
        stopPlayback()
        try {
            player = android.media.MediaPlayer().apply {
                set(this)
                prepare()
                setOnCompletionListener { stopPlayback() }
                start()
            }
            playingKey = key
            playingBtn = btn
            btn.setImageResource(R.drawable.ic_pause)
        } catch (e: Exception) {
            toast("Playback failed: ${e.message}")
            stopPlayback()
        }
    }

    private fun stopPlayback() {
        playingBtn?.setImageResource(R.drawable.ic_play)
        playingBtn = null
        playingKey = null
        player?.let { runCatching { it.stop() }; it.release() }
        player = null
    }

    // ---- Voices tab --------------------------------------------------------

    private lateinit var voicesList: LinearLayout

    /** Speaker awaiting a folder pick, or null when exporting the whole library. */
    private var exportOne: VoiceStore.Voice? = null

    private fun buildVoicesTab(): View {
        val (scroll, col) = page()
        col.title("Speakers")
        col.caption("Two kinds live here: Supertonic style files (⚡ about a second a sentence) and " +
            "reference recordings for the Qwen models (🐢 tens of seconds). Each section holds the " +
            "speakers one model can use.")
        val topRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        topRow.addView(Button(this).apply {
            text = "Clone ▾"
            setOnClickListener { cloneMenu(this) }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        topRow.addView(Button(this).apply {
            text = "Design"
            setOnClickListener { designVoiceDialog() }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6) })
        topRow.addView(Button(this).apply {
            text = "Backup / Restore ▾"
            setOnClickListener { filesMenu(this) }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6) })
        col.addView(topRow)

        voicesList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(10), 0, 0) }
        col.addView(voicesList)
        return scroll
    }

    // ---- Chats tab ---------------------------------------------------------

    /** Conversations are things you keep, so they get a list of their own
     *  rather than a button that always dropped you into the same unnamed one. */
    private fun buildChatsTab(): View {
        val (scroll, col) = page()
        col.title("Chats")
        col.caption("Each chat is a conversation you can reopen: the lines, who said them, and " +
            "the audio already generated for each one.")

        col.addView(Button(this).apply {
            text = "New chat"
            setOnClickListener { openChat(null) }
        }, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.WRAP_CONTENT))

        val chats = ChatStore.list(this)
        if (chats.isEmpty()) {
            col.addView(TextView(this).apply {
                text = "No chats yet."
                alpha = 0.6f; setPadding(0, dp(14), 0, 0)
            })
            return scroll
        }
        for (c in chats) {
            col.addView(card {
                val row = LinearLayout(context).apply {
                    orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
                    isClickable = true
                    setOnClickListener { openChat(c.id) }
                }
                row.addView(LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    addView(TextView(context).apply {
                        text = c.name
                        textSize = 17f; setTypeface(typeface, Typeface.BOLD)
                        maxLines = 1
                        ellipsize = android.text.TextUtils.TruncateAt.END
                    })
                    addView(TextView(context).apply {
                        text = listOf(
                            "${c.lines} line${if (c.lines == 1) "" else "s"}",
                            Wav.fmt(c.seconds),
                            android.text.format.DateUtils.getRelativeTimeSpanString(c.updated),
                        ).joinToString(" · ")
                        textSize = 12f; alpha = 0.6f
                    })
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                lateinit var moreBtn: ImageButton
                moreBtn = iconBtn(R.drawable.ic_more, "Chat actions") { chatMenu(moreBtn, c) }
                row.addView(moreBtn)
                addView(row)
            })
        }
        return scroll
    }

    private fun openChat(id: String?) {
        if (voicesForCurrentModel().isEmpty()) {
            toast("No speakers for the selected model — add one on the Speakers tab")
            return
        }
        startActivity(Intent(this, TalkActivity::class.java)
            .putExtra(TalkActivity.EXTRA_CHAT_ID, id))
    }

    private fun chatMenu(anchor: View, c: ChatStore.Meta) {
        android.widget.PopupMenu(this, anchor).apply {
            menu.add("Open")
            menu.add("Rename")
            menu.add("Delete")
            setOnMenuItemClickListener {
                when (it.title) {
                    "Open" -> openChat(c.id)
                    "Rename" -> {
                        val edit = EditText(this@MainActivity).apply {
                            setText(c.name); setSingleLine()
                            setPadding(dp(20), dp(12), dp(20), dp(12))
                        }
                        MaterialAlertDialogBuilder(this@MainActivity)
                            .setTitle("Chat name")
                            .setView(edit)
                            .setPositiveButton("Rename") { _, _ ->
                                ChatStore.rename(this@MainActivity, c.id, edit.text.toString())
                                selectTab(TAB_CHATS)
                            }
                            .setNegativeButton("Cancel", null).show()
                    }
                    "Delete" -> MaterialAlertDialogBuilder(this@MainActivity)
                        .setMessage("Delete “${c.name}” and its audio?")
                        .setPositiveButton("Delete") { _, _ ->
                            ChatStore.delete(this@MainActivity, c.id)
                            selectTab(TAB_CHATS)
                        }
                        .setNegativeButton("Cancel", null).show()
                }
                true
            }
            show()
        }
    }

    /** Both ways to turn a person into a speaker: a recording you already have,
     *  or one you make now. Either way it lands as a reference voice and is
     *  cloned into a Supertonic style if the encoder is installed. */
    private fun cloneMenu(anchor: View) {
        android.widget.PopupMenu(this, anchor).apply {
            menu.add("From a sound file…")
            menu.add("Record now…")
            setOnMenuItemClickListener {
                when (it.title) {
                    "From a sound file…" -> startActivityForResult(
                        Intent(Intent.ACTION_OPEN_DOCUMENT)
                            .addCategory(Intent.CATEGORY_OPENABLE)
                            .setType("audio/*"), REQ_CLONE_AUDIO)
                    "Record now…" -> startActivity(Intent(this@MainActivity, RecordActivity::class.java))
                }
                true
            }
            show()
        }
    }

    /** Import and export in one menu: a single file, a whole folder, or the
     *  library back out to a folder — the round trip that moves speakers
     *  between phones and off the desktop cloning tool. */
    private fun filesMenu(anchor: View) {
        val folder = SpeakerFolder.label(this)
        android.widget.PopupMenu(this, anchor).apply {
            menu.add("Import a file…")
            menu.add("Import a folder…")
            menu.add("Export all speakers…")
            menu.add(if (folder == null) "Keep a backup folder…" else "Backup folder: $folder")
            if (folder != null) menu.add("Sync backup folder now")
            setOnMenuItemClickListener {
                when (it.title) {
                    "Import a file…" -> startActivityForResult(
                        Intent(Intent.ACTION_OPEN_DOCUMENT)
                            .addCategory(Intent.CATEGORY_OPENABLE)
                            .setType("*/*")
                            .putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true)
                            .putExtra(Intent.EXTRA_MIME_TYPES,
                                arrayOf("audio/*", "application/json", "text/json")), REQ_IMPORT)
                    "Import a folder…" ->
                        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), REQ_IMPORT_DIR)
                    "Export all speakers…" -> {
                        exportOne = null
                        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), REQ_EXPORT_DIR)
                    }
                    "Sync backup folder now" -> syncSpeakerFolder(loud = true)
                    else -> startActivityForResult(
                        Intent(Intent.ACTION_OPEN_DOCUMENT_TREE)
                            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or
                                      Intent.FLAG_GRANT_WRITE_URI_PERMISSION or
                                      Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION), REQ_BACKUP_DIR)
                }
                true
            }
            show()
        }
    }

    /** Mirrors the library into the backup folder and pulls back anything only
     *  the folder has. Runs off-main: it is a pile of content-provider calls. */
    private fun syncSpeakerFolder(loud: Boolean) {
        if (SpeakerFolder.uri(this) == null) return
        thread {
            val r = SpeakerFolder.sync(this)
            runOnUiThread {
                if (r.imported > 0) rebuildVoices()
                if (loud) toast(when {
                    r.error != null -> "Backup folder: ${r.error}"
                    r.exported == 0 && r.imported == 0 -> "Backup folder already up to date"
                    else -> "Backup folder: ${r.exported} saved, ${r.imported} restored"
                })
            }
        }
    }

    /** Supertonic ships style voices with the model; surface them in the
     *  library the first time they appear so they are pickable like any voice. */
    private fun adoptSupertonicStyles() {
        val have = VoiceStore.styleList(this).map { it.name }.toSet()
        for (f in ModelManager.supertonicStyles(this)) {
            if (f.nameWithoutExtension !in have) {
                runCatching { VoiceStore.importStyle(this, f, f.nameWithoutExtension) }
            }
        }
    }

    /** A collapsible group. Which one is open is remembered per section, but
     *  the first time round the model you are actually using decides — that is
     *  the section whose speakers you can pick right now. */
    private fun section(key: String, title: String, subtitle: String, count: Int,
                        defaultOpen: Boolean, body: LinearLayout.() -> Unit) {
        val open = prefs().getBoolean("sec_$key", defaultOpen)
        val header = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(10), 0, dp(6))
            isClickable = true
            setOnClickListener {
                prefs().edit().putBoolean("sec_$key", !open).apply()
                rebuildVoices()
            }
        }
        header.addView(TextView(this).apply {
            text = if (open) "▾" else "▸"
            textSize = 14f; alpha = 0.7f; setPadding(0, 0, dp(8), 0)
        })
        header.addView(LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            addView(TextView(this@MainActivity).apply {
                text = "$title  ($count)"
                textSize = 14f; setTypeface(typeface, Typeface.BOLD)
            })
            addView(TextView(this@MainActivity).apply {
                text = subtitle; textSize = 11f; alpha = 0.6f
            })
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        voicesList.addView(header)
        if (!open) return
        val holder = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; body() }
        voicesList.addView(holder)
    }

    private fun rebuildVoices() {
        if (!::voicesList.isInitialized) return
        adoptSupertonicStyles()
        voicesList.removeAllViews()
        val def = VoiceStore.defaultVoice(this)
        val modelId = currentModelId()
        val supertonicModel = ModelManager.selectedModel(this)?.engine == "supertonic"
        val styleVoices = VoiceStore.styleList(this)
        val styleDef = VoiceStore.defaultFor(this, true)

        section("styles", "⚡ Supertonic 3 speakers", "style files · about a second a sentence",
                styleVoices.size, defaultOpen = supertonicModel) {
            if (styleVoices.isEmpty()) {
                addView(TextView(context).apply {
                    text = "None yet — import a style file or folder."
                    textSize = 12f; alpha = 0.6f; setPadding(0, 0, 0, dp(8))
                })
            }
            for (v in styleVoices) {
                addView(card {
                    val row = LinearLayout(context).apply {
                        orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
                    }
                    row.addView(RadioButton(context).apply {
                        isChecked = v.name == styleDef?.name
                        setOnClickListener {
                            VoiceStore.setDefaultFor(this@MainActivity, v, true)
                            rebuildVoices(); refreshVoiceLabel()
                        }
                    })
                    row.addView(TextView(context).apply {
                        text = "⚡"; textSize = 11f
                        setPadding(dp(2), 0, dp(2), 0)
                    })
                    row.addView(TextView(context).apply {
                        text = VoiceStore.icon(this@MainActivity, v.name)
                        textSize = 20f; setPadding(0, 0, dp(8), 0)
                    })
                    row.addView(TextView(context).apply {
                        text = v.name; textSize = 17f; setTypeface(typeface, Typeface.BOLD)
                        maxLines = 1
                        ellipsize = android.text.TextUtils.TruncateAt.END
                    }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                    val hasPreview = VoiceStore.previewFile(this@MainActivity, v.name, modelId).exists()
                    row.addView(previewControl(v, hasPreview))
                    lateinit var moreBtn: ImageButton
                    moreBtn = iconBtn(R.drawable.ic_more, "More") { speakerMenu(moreBtn, v) }
                    row.addView(moreBtn)
                    addView(row)
                    addView(TextView(context).apply {
                        text = styleSubtitle(v)
                        textSize = 12f; alpha = 0.6f
                    })
                })
            }
        }

        val refVoices = VoiceStore.list(this)
        section("refs", "🐢 Qwen speakers", "reference recordings · tens of seconds a sentence",
                refVoices.size, defaultOpen = !supertonicModel) {
        if (refVoices.isEmpty()) {
            addView(TextView(context).apply {
                text = "None yet — record, design or import a recording."
                textSize = 12f; alpha = 0.6f; setPadding(0, 0, 0, dp(8))
            })
        }
        for (v in refVoices) {
            addView(card {
                val head = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
                head.addView(RadioButton(context).apply {
                    isChecked = v.name == def?.name
                    setOnClickListener {
                        VoiceStore.setDefault(this@MainActivity, v)
                        rebuildVoices(); refreshVoiceLabel()
                    }
                })
                head.addView(TextView(context).apply {
                    text = "🐢"; textSize = 11f
                    setPadding(dp(2), 0, dp(2), 0)
                })
                head.addView(TextView(context).apply {
                    text = VoiceStore.icon(this@MainActivity, v.name)
                    textSize = 20f
                    setPadding(0, 0, dp(8), 0)
                })
                head.addView(TextView(context).apply {
                    text = v.name; textSize = 17f; setTypeface(typeface, Typeface.BOLD)
                    maxLines = 1
                    ellipsize = android.text.TextUtils.TruncateAt.END
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                val hasPreview = VoiceStore.previewFile(this@MainActivity, v.name, modelId).exists()
                head.addView(previewControl(v, hasPreview))
                if (VoiceCloner.available(this@MainActivity)) {
                    head.addView(iconBtn(R.drawable.ic_add, "Clone to a Supertonic voice") {
                        cloneToSupertonic(v)
                    })
                }
                lateinit var moreBtn: ImageButton
                moreBtn = iconBtn(R.drawable.ic_more, "More") { speakerMenu(moreBtn, v) }
                head.addView(moreBtn)
                addView(head)
                val bits = mutableListOf("${v.file.length() / 1024} KB")
                if (!hasPreview) bits.add("no preview for $modelId yet")
                if (VoiceStore.transcriptFile(this@MainActivity, v.name).exists()) bits.add("transcript ✓")
                addView(TextView(context).apply { text = bits.joinToString(" · "); textSize = 12f; alpha = 0.6f })
            })
        }
        }
    }

    /** What a style file can tell about itself: the cloner writes where it came
     *  from and how close it scored, and that is worth showing. */
    private fun styleSubtitle(v: VoiceStore.Voice): String {
        val meta = runCatching {
            org.json.JSONObject(v.file.readText()).optJSONObject("metadata")
        }.getOrNull() ?: return "style file · ${v.file.length() / 1024} KB"
        val bits = mutableListOf<String>()
        meta.optString("source").takeIf { it.isNotBlank() }?.let { bits.add(it) }
        meta.optDouble("held_out_cos", Double.NaN).takeIf { !it.isNaN() }
            ?.let { bits.add("similarity ${"%.2f".format(it)}") }
        if (bits.isEmpty()) bits.add("style file")
        return bits.joinToString(" · ")
    }

    /** The rest of a speaker's actions. They live behind one button because a
     *  phone-width row cannot hold five icons and still show a name. */
    private fun speakerMenu(anchor: View, v: VoiceStore.Voice) {
        android.widget.PopupMenu(this, anchor).apply {
            menu.add("Share as a file")
            menu.add("Share preview audio")
            menu.add("Save to folder…")
            menu.add("Edit…")
            menu.add("Delete")
            setOnMenuItemClickListener {
                when (it.title) {
                    "Share as a file" -> shareSpeaker(v)
                    // the file is what another phone imports; the preview is
                    // what you send someone to let them hear the voice
                    "Share preview audio" -> {
                        val p = VoiceStore.previewFile(this@MainActivity, v.name, currentModelId())
                        if (!p.exists()) toast("Generate a preview first (▶)")
                        else thread {
                            runCatching { AudioShare.shareWavAsM4a(this@MainActivity, p, "Voice ${v.name}") }
                                .onFailure { e -> runOnUiThread { toast("Share failed: ${e.message}") } }
                        }
                    }
                    "Save to folder…" -> {
                        exportOne = v
                        startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), REQ_EXPORT_DIR)
                    }
                    "Edit…" -> editSpeakerDialog(v)
                    "Delete" -> MaterialAlertDialogBuilder(this@MainActivity)
                        .setMessage("Delete speaker “${v.name}”?")
                        .setPositiveButton("Delete") { _, _ ->
                            val gone = v.file.name
                            VoiceStore.delete(this@MainActivity, v); rebuildVoices()
                            thread { SpeakerFolder.remove(this@MainActivity, gone) }
                        }
                        .setNegativeButton("Cancel", null).show()
                }
                true
            }
            show()
        }
    }

    /** Hands the speaker out as the file it actually is — a style JSON or the
     *  reference recording — so the other end can import it unchanged. */
    private fun shareSpeaker(v: VoiceStore.Voice) {
        try {
            val uri = androidx.core.content.FileProvider.getUriForFile(
                this, "$packageName.files", v.file)
            val mime = if (v.file.extension.equals("json", true)) "application/json" else "audio/*"
            startActivity(Intent.createChooser(
                Intent(Intent.ACTION_SEND)
                    .setType(mime)
                    .putExtra(Intent.EXTRA_STREAM, uri)
                    .putExtra(Intent.EXTRA_SUBJECT, "Speaker: ${v.name}")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),
                "Share speaker"))
        } catch (e: Exception) {
            toast("Share failed: ${e.message}")
        }
    }

    /** Predicts a Supertonic style from this reference recording, on the phone.
     *  Experimental: the desktop cloner is still the quality reference, and the
     *  dialog says so rather than pretending otherwise. */
    private fun cloneToSupertonic(v: VoiceStore.Voice) {
        toast("Cloning “${v.name}” on device…")
        thread {
            val cloner = VoiceCloner(this)
            val out = cloner.cloneToStyle(v.file, "${v.name} (device clone)")
            cloner.close()
            runOnUiThread {
                if (out == null) {
                    toast("Cloning failed — see Settings → Copy log")
                } else {
                    rebuildVoices()
                    MaterialAlertDialogBuilder(this)
                        .setTitle("Cloned: ${out.name}")
                        .setMessage("Predicted a Supertonic style from ${v.name}. " +
                            "Switch the model to Supertonic 3 and pick this voice to hear it.\n\n" +
                            "This is the experimental on-device encoder — the desktop " +
                            "cloning tool still produces closer voices.")
                        .setPositiveButton("OK", null)
                        .show()
                }
            }
        }
    }

    /** Voice whose preview is being generated right now, so its row can show a
     *  spinner instead of a play button that looks like it did nothing. */
    private var previewing: String? = null

    /** Play button, or a spinner while this voice is being generated. The
     *  first tap on a voice costs a model load and a synthesis pass — several
     *  seconds — and an unchanged play icon reads as "nothing happened". */
    private fun previewControl(v: VoiceStore.Voice, hasPreview: Boolean): View {
        if (previewing == v.name) {
            return ProgressBar(this).apply {
                isIndeterminate = true
                val s = dp(22)
                layoutParams = LinearLayout.LayoutParams(s, s).apply {
                    marginStart = dp(9); marginEnd = dp(9)
                }
            }
        }
        lateinit var btn: ImageButton
        btn = iconBtn(R.drawable.ic_play,
            if (hasPreview) "Play preview" else "Generate preview") { previewVoice(v, btn) }
        return btn
    }

    private fun previewVoice(v: VoiceStore.Voice, btn: ImageButton) {
        val cached = VoiceStore.previewFile(this, v.name, currentModelId())
        if (cached.exists()) { playFile(cached.absolutePath, btn); return }
        if (ModelManager.selectedModel(this) == null) { toast("Download a model first (Settings tab)"); return }
        previewing = v.name
        rebuildVoices()
        toast("Generating preview with ${currentModelId()} — plays when ready, then cached")
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, VoiceStore.PREVIEW_TEXT)
            .putExtra(TtsService.EXTRA_TITLE, "Preview: ${v.name}")
            .putExtra(TtsService.EXTRA_VOICE, v.name)
            .putExtra(TtsService.EXTRA_BACKEND, Backends.current(this))
            .putExtra(TtsService.EXTRA_PREVIEW, true))
    }

    /** Name, icon and reference transcript in one sheet. The transcript is
     *  manual for now — the slot a future on-device whisper would pre-fill for
     *  the user to validate. */
    private fun editSpeakerDialog(v: VoiceStore.Voice) {
        val transcript = VoiceStore.transcriptFile(this, v.name)
        var chosenIcon = VoiceStore.icon(this, v.name)

        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(20), dp(8), dp(20), dp(4))
        }
        col.addView(TextView(this).apply { text = "Name"; textSize = 12f; alpha = 0.7f })
        val nameEdit = EditText(this).apply { setText(v.name); setSingleLine() }
        col.addView(nameEdit)

        col.addView(TextView(this).apply {
            text = "Icon — pick one, or type your own"
            textSize = 12f; alpha = 0.7f; setPadding(0, dp(14), 0, dp(4))
        })
        // an emoji field beats any list I could ship: the keyboard already has
        // every glyph, and someone's voice is theirs to label
        val customIcon = EditText(this).apply {
            hint = "any emoji"
            setText(chosenIcon)
            setSingleLine()
            textSize = 20f
            filters = arrayOf(android.text.InputFilter.LengthFilter(4))
        }
        col.addView(customIcon)
        val perRow = 8
        val cells = mutableListOf<TextView>()
        fun paintCells() = cells.forEach {
            it.setBackgroundColor(if (it.text == chosenIcon)
                themeColor(com.google.android.material.R.attr.colorSecondaryContainer)
            else android.graphics.Color.TRANSPARENT)
        }
        VoiceStore.ICONS.chunked(perRow).forEach { row ->
            col.addView(LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                row.forEach { emoji ->
                    val cell = TextView(context).apply {
                        text = emoji
                        textSize = 22f
                        gravity = Gravity.CENTER
                        setPadding(0, dp(6), 0, dp(6))
                        setOnClickListener {
                            chosenIcon = emoji
                            customIcon.setText(emoji)
                            paintCells()
                        }
                    }
                    cells.add(cell)
                    addView(cell, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                }
            })
        }
        paintCells()

        col.addView(TextView(this).apply {
            text = "Transcript of the reference recording (optional)"
            textSize = 12f; alpha = 0.7f; setPadding(0, dp(14), 0, dp(4))
        })
        val transcriptEdit = EditText(this).apply {
            setText(if (transcript.exists()) transcript.readText() else "")
            hint = "What is said in the recording"
            minLines = 2
        }
        col.addView(transcriptEdit)

        MaterialAlertDialogBuilder(this)
            .setTitle("Edit speaker")
            .setView(ScrollView(this).apply { addView(col) })
            .setNeutralButton("Save to folder…") { _, _ ->
                exportOne = v
                startActivityForResult(Intent(Intent.ACTION_OPEN_DOCUMENT_TREE), REQ_EXPORT_DIR)
            }
            .setPositiveButton("Save") { _, _ ->
                // whatever is in the field wins: it holds the tapped choice
                // unless the user typed over it
                chosenIcon = customIcon.text.toString().trim().ifBlank { chosenIcon }
                val t = transcriptEdit.text.toString().trim()
                if (t.isBlank()) transcript.delete() else transcript.writeText(t)
                val renamed = try {
                    VoiceStore.rename(this, v, nameEdit.text.toString())
                } catch (e: Exception) {
                    toast("Rename failed: ${e.message}"); v
                }
                VoiceStore.setIcon(this, renamed.name, chosenIcon)
                if (pickedVoice == v.name) pickedVoice = renamed.name
                rebuildVoices()
                refreshVoiceLabel()
                if (renamed.file.name != v.file.name) thread {
                    // the backup folder tracks the library, so the old name goes
                    SpeakerFolder.remove(this, v.file.name)
                    SpeakerFolder.mirror(this, renamed)
                }
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    // ---- voice designer ----------------------------------------------------

    private var designing = false
    private var designInstruct = ""

    private fun designVoiceDialog() {
        if (ModelManager.selectedModel(this) == null) { toast("Download a model first (Settings tab)"); return }
        val hasVd = ModelManager.designModel(this) != null
        val edit = EditText(this).apply {
            hint = if (hasVd) "Describe the voice, e.g. “a deep, gravelly old male voice, slow delivery” — or leave empty to roll a random one"
                   else "VoiceDesign model not installed — leave empty to roll a random voice"
            setText(designInstruct)
            minLines = 2
            isEnabled = true
            setPadding(dp(20), dp(12), dp(20), dp(12))
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("Design a voice")
            .setMessage("Listen to the result, keep it if you like it, or roll again. Kept voices become " +
                "normal reference audio usable with any model.")
            .setView(edit)
            .setPositiveButton("Roll") { _, _ ->
                designInstruct = edit.text.toString().trim()
                if (designInstruct.isNotBlank() && !hasVd) {
                    toast("Description needs the VoiceDesign model (Settings) — rolling random instead")
                }
                rollDesignVoice()
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun rollDesignVoice() {
        designing = true
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, VoiceStore.DESIGN_TEXT)
            .putExtra(TtsService.EXTRA_TITLE, "Designing voice")
            .putExtra(TtsService.EXTRA_VOICE, "")
            .putExtra(TtsService.EXTRA_DESIGN, true)
            .putExtra(TtsService.EXTRA_INSTRUCT, designInstruct)
            .putExtra(TtsService.EXTRA_SEED, Random.nextInt(1, 1 shl 30))
            .putExtra(TtsService.EXTRA_BACKEND, Backends.current(this)))
        toast(if (designInstruct.isBlank()) "Rolling a random voice…" else "Designing: “${designInstruct.take(48)}…”")
    }

    private fun offerKeepDesignedVoice() {
        if (!designing) return
        designing = false
        val edit = EditText(this).apply {
            hint = "Voice name"
            setPadding(dp(20), dp(12), dp(20), dp(12))
        }
        MaterialAlertDialogBuilder(this)
            .setTitle("Keep this voice?")
            .setView(edit)
            .setPositiveButton("Keep") { _, _ ->
                val name = edit.text.toString().ifBlank { "designed" }
                try {
                    val v = VoiceStore.adopt(this, lastAudio(), name)
                    lastAudio().copyTo(VoiceStore.previewFile(this, v.name, currentModelId()), overwrite = true)
                    rebuildVoices()
                    toast("Voice “${v.name}” saved")
                } catch (e: Exception) { toast("Keep failed: ${e.message}") }
            }
            .setNeutralButton("Roll again") { _, _ -> rollDesignVoice() }
            .setNegativeButton("Discard", null)
            .show()
    }

    private fun displayName(uri: Uri, fallback: String = "voice.wav"): String =
        contentResolver.query(uri, null, null, null, null)?.use { c ->
            val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
            if (c.moveToFirst() && idx >= 0) c.getString(idx) else null
        } ?: fallback

    /** Imports every speaker file in a picked folder. Uses DocumentsContract
     *  directly rather than pulling in documentfile for one listing. */
    private fun importFolder(tree: Uri) {
        val children = android.provider.DocumentsContract.buildChildDocumentsUriUsingTree(
            tree, android.provider.DocumentsContract.getTreeDocumentId(tree))
        var ok = 0
        var skipped = 0
        val failures = mutableListOf<String>()
        contentResolver.query(children, arrayOf(
            android.provider.DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            android.provider.DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null)
            ?.use { c ->
                while (c.moveToNext()) {
                    val name = c.getString(1) ?: continue
                    if (!VoiceStore.isSpeakerFile(name)) { skipped++; continue }
                    val doc = android.provider.DocumentsContract.buildDocumentUriUsingTree(tree, c.getString(0))
                    try {
                        VoiceStore.importAny(this, doc, name); ok++
                    } catch (e: Exception) {
                        failures.add("$name: ${e.message}")
                    }
                }
            }
        rebuildVoices()
        DebugLog.log(this, "MainActivity",
            "folder import: $ok imported, $skipped skipped, ${failures.size} failed ${failures.take(3)}")
        toast(when {
            ok == 0 && failures.isEmpty() -> "No speaker files in that folder"
            failures.isEmpty() -> "Imported $ok speaker${if (ok == 1) "" else "s"}"
            else -> "Imported $ok, ${failures.size} failed — see Settings → Copy log"
        })
    }

    /** Writes [what] (or the whole library) into a picked folder. */
    private fun exportFolder(tree: Uri, what: VoiceStore.Voice?) {
        val parent = android.provider.DocumentsContract.buildDocumentUriUsingTree(
            tree, android.provider.DocumentsContract.getTreeDocumentId(tree))
        val voices = what?.let { listOf(it) } ?: (VoiceStore.styleList(this) + VoiceStore.list(this))
        var ok = 0
        val failures = mutableListOf<String>()
        for (v in voices) {
            try {
                val mime = if (v.file.extension.equals("json", true)) "application/json" else "audio/*"
                val dest = android.provider.DocumentsContract.createDocument(
                    contentResolver, parent, mime, v.file.name)
                    ?: throw java.io.IOException("could not create ${v.file.name}")
                contentResolver.openOutputStream(dest)!!.use { out -> v.file.inputStream().use { it.copyTo(out) } }
                ok++
            } catch (e: Exception) {
                failures.add("${v.file.name}: ${e.message}")
            }
        }
        DebugLog.log(this, "MainActivity", "folder export: $ok written, ${failures.size} failed ${failures.take(3)}")
        toast(if (failures.isEmpty()) "Exported $ok file${if (ok == 1) "" else "s"}"
              else "Exported $ok, ${failures.size} failed — see Settings → Copy log")
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode != RESULT_OK || data == null) return
        when (requestCode) {
            REQ_IMPORT -> {
                // the picker returns one uri in data, or several in a ClipData
                val uris = data.clipData?.let { clip ->
                    (0 until clip.itemCount).map { clip.getItemAt(it).uri }
                } ?: listOfNotNull(data.data)
                var ok = 0
                for (uri in uris) {
                    try {
                        VoiceStore.importAny(this, uri, displayName(uri)); ok++
                    } catch (e: Exception) {
                        toast("Import failed: ${e.message}")
                    }
                }
                if (ok > 0) toast("Imported $ok speaker${if (ok == 1) "" else "s"}")
                rebuildVoices()
            }
            REQ_CLONE_AUDIO -> data.data?.let { uri ->
                val v = try {
                    VoiceStore.import(this, uri, displayName(uri, "recording.wav"))
                } catch (e: Exception) {
                    toast("Could not read that file: ${e.message}"); return
                }
                rebuildVoices()
                if (VoiceCloner.available(this)) cloneToSupertonic(v)
                else MaterialAlertDialogBuilder(this)
                    .setTitle("Imported “${v.name}”")
                    .setMessage("It is usable with the Qwen models as it is. To turn it into a " +
                        "fast Supertonic speaker, download the cloning encoder in Settings — or " +
                        "run the desktop cloning tool, which is still much closer to the original.")
                    .setPositiveButton("OK", null)
                    .show()
            }
            REQ_IMPORT_DIR -> data.data?.let { importFolder(it) }
            REQ_EXPORT_DIR -> data.data?.let { exportFolder(it, exportOne); exportOne = null }
            REQ_SAVE_SPEC -> data.data?.let { uri ->
                val port = prefs().getInt("host_port", HostingService.DEFAULT_PORT)
                val base = "http://${HttpServer.lanAddress() ?: "127.0.0.1"}:$port"
                runCatching {
                    contentResolver.openOutputStream(uri)!!.use { it.write(specText(base).toByteArray()) }
                }.onSuccess { toast("Spec saved") }.onFailure { toast("Save failed: ${it.message}") }
            }
            REQ_BACKUP_DIR -> data.data?.let {
                SpeakerFolder.remember(this, it)
                toast("Speakers will be kept in ${SpeakerFolder.label(this)}")
                syncSpeakerFolder(loud = true)
            }
        }
    }

    // ---- Jobs tab ----------------------------------------------------------

    private lateinit var jobsList: LinearLayout
    private lateinit var jobsSummary: TextView
    private lateinit var waveform: WaveformView

    // widgets of the currently running job's card, driven by the status
    // broadcasts; null whenever no job is running or the tab is elsewhere
    private var runningLine: TextView? = null
    private var runningBar: ProgressBar? = null
    private var runningPulse: android.animation.ObjectAnimator? = null

    private fun buildJobsTab(): View {
        val (scroll, col) = page()
        col.title("Jobs")
        col.caption("Share a web page — or any selected text — to TTS Runner from another app's " +
            "share sheet and it fetches the article, strips the navigation and reads it aloud. " +
            "You get the extracted text to check and edit before a word is spoken.")
        col.addView(newJobCard())
        jobsSummary = TextView(this).apply { textSize = 13f; alpha = 0.7f; setPadding(0, 0, 0, dp(12)) }
        col.addView(jobsSummary)

        col.addView(card {
            addView(TextView(context).apply { text = "Last generated audio"; textSize = 14f; setTypeface(typeface, Typeface.BOLD) })
            val row = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
            lateinit var lastBtn: ImageButton
            lastBtn = iconBtn(R.drawable.ic_play, "Play last audio") {
                val f = lastAudio()
                if (f.exists() && f.length() > 44) playFile(f.absolutePath, lastBtn) else toast("No audio yet")
            }
            row.addView(lastBtn)
            waveform = WaveformView(context)
            row.addView(waveform, LinearLayout.LayoutParams(0, dp(48), 1f).apply { marginStart = dp(8) })
            row.addView(iconBtn(R.drawable.ic_share, "Share last audio") {
                val f = lastAudio()
                if (!f.exists() || f.length() <= 44) { toast("No audio yet"); return@iconBtn }
                thread {
                    runCatching { AudioShare.shareWavAsM4a(this@MainActivity, f, "TTS audio") }
                        .onFailure { e -> runOnUiThread { toast("Share failed: ${e.message}") } }
                }
            })
            addView(row)
        })

        jobsList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        col.addView(jobsList)
        return scroll
    }

    /** Cheap poll off the status broadcasts: rebuild only when a job appears or
     *  changes state, so a running job's per-frame updates don't re-lay-out the
     *  whole list (they go straight to runningLine/runningBar instead). */
    private var jobsSig = ""
    private var lastJobsCheck = 0L

    private fun refreshJobsIfChanged(force: Boolean = false) {
        if (currentTab != TAB_JOBS || !::jobsList.isInitialized) return
        val now = android.os.SystemClock.uptimeMillis()
        if (!force && now - lastJobsCheck < 1500) return
        lastJobsCheck = now
        if (jobsSignature() != jobsSig) rebuildJobs()
    }

    private fun jobsSignature() = JobStore.list(this).joinToString(",") { "${it.id}:${it.status}" }

    private fun rebuildJobs() {
        if (!::jobsList.isInitialized) return
        runningPulse?.cancel()
        runningPulse = null
        runningLine = null
        runningBar = null
        val engineAlive = runCatching {
            getSystemService(android.app.ActivityManager::class.java)
                .runningAppProcesses?.any { it.processName.endsWith(":engine") } == true
        }.getOrDefault(false)
        JobStore.reconcile(this, engineAlive)
        JobStore.sweepCaches(this)
        val lf = lastAudio()
        if (lf.exists() && lf.length() > 44) thread { waveform.loadWav(lf) }
        jobsList.removeAllViews()
        val jobs = JobStore.list(this)
        jobsSig = jobs.joinToString(",") { "${it.id}:${it.status}" }
        val totalAudio = jobs.sumOf { it.audioSecs }
        jobsSummary.text = "${jobs.size} jobs · ${"%.0f".format(totalAudio / 60)} min of audio generated"
        val fmt = java.text.SimpleDateFormat("MMM d HH:mm", java.util.Locale.US)
        for (j in jobs) {
            val running = j.status == "running"
            val jobCard = card {
                val head = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
                if (running) {
                    // pulsing dot: the one moving thing in the list
                    head.addView(TextView(context).apply {
                        text = "●"; textSize = 14f
                        setTextColor(themeColor(com.google.android.material.R.attr.colorPrimary))
                        setPadding(0, 0, dp(8), 0)
                        runningPulse = android.animation.ObjectAnimator.ofFloat(this, "alpha", 1f, 0.25f).apply {
                            duration = 700
                            repeatMode = android.animation.ValueAnimator.REVERSE
                            repeatCount = android.animation.ValueAnimator.INFINITE
                            start()
                        }
                    })
                }
                head.addView(TextView(context).apply {
                    text = j.title.ifBlank { j.text.take(40) }
                    textSize = 16f; setTypeface(typeface, Typeface.BOLD)
                    if (running) setTextColor(themeColor(com.google.android.material.R.attr.colorPrimary))
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                head.addView(iconBtn(R.drawable.ic_copy, "Copy text") {
                    getSystemService(android.content.ClipboardManager::class.java)
                        .setPrimaryClip(android.content.ClipData.newPlainText("tts-text", j.text))
                    toast("Text copied (${j.text.length} chars)")
                })
                if (j.outputUri.isNotBlank()) {
                    lateinit var jobBtn: ImageButton
                    jobBtn = iconBtn(R.drawable.ic_play, "Play") { playUri(Uri.parse(j.outputUri), jobBtn) }
                    head.addView(jobBtn)
                    head.addView(iconBtn(R.drawable.ic_share, "Share") {
                        AudioShare.shareUri(this@MainActivity, Uri.parse(j.outputUri), j.title)
                    })
                }
                head.addView(iconBtn(R.drawable.ic_delete, "Delete") {
                    JobStore.delete(this@MainActivity, j.id); rebuildJobs()
                })
                addView(head)
                val statusIcon = when (j.status) {
                    "done" -> "✓"; "running" -> "⏳"; "stopped" -> "⏹"; else -> "✗"
                }
                val stats = if (j.audioSecs > 0)
                    " · ${"%.0f".format(j.audioSecs)}s audio · ${j.genMs / 1000}s gen · RTF ${"%.1f".format(j.genMs / 1000.0 / j.audioSecs)}"
                else ""
                addView(TextView(context).apply {
                    textSize = 12f; alpha = 0.7f
                    text = "$statusIcon ${j.status} · ${fmt.format(java.util.Date(j.id))} · " +
                        // speed comes from the job's own model, not from today's library:
                        // the voice may have been renamed or deleted since
                        "${j.text.length} chars · " +
                        VoiceStore.label(this@MainActivity, j.voice,
                            ModelManager.CATALOG.find { m -> m.id == j.model }?.engine == "supertonic") +
                        " · ${j.model}/${j.backend}$stats" +
                        (if (j.output.isNotBlank()) "\n→ ${j.output}" else "") +
                        (if (j.error.isNotBlank()) "\n${j.error}" else "")
                })
                if (running) {
                    // live mirror of the status pane, updated by the receiver
                    runningLine = TextView(context).apply {
                        textSize = 13f
                        text = runStatus.text.ifBlank { "Starting…" }
                        setPadding(0, dp(4), 0, dp(2))
                    }
                    addView(runningLine)
                    runningBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
                        max = 1000
                        progress = runProgress.progress
                        isIndeterminate = runProgress.progress <= 0
                    }
                    addView(runningBar)
                }
                val cachedChunks = if (running) 0 else JobStore.cachedChunks(this@MainActivity, j.id)
                if (cachedChunks > 0) {
                    addView(Button(context).apply {
                        text = "Resume — $cachedChunks${if (j.chunksTotal > 0) "/${j.chunksTotal}" else ""} chunks kept"
                        setOnClickListener { resumeDialog(j, cachedChunks) }
                        layoutParams = LinearLayout.LayoutParams(
                            ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
                    })
                }
                addView(LinearLayout(context).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(Button(context).apply {
                        text = if (cachedChunks > 0) "Start over" else "Re-run"
                        setOnClickListener {
                            if (cachedChunks > 0) JobStore.jobDir(this@MainActivity, j.id).deleteRecursively()
                            rerunJob(j, j.voice)
                        }
                    }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                    addView(Button(context).apply {
                        text = "Other voice…"
                        setOnClickListener {
                            val names = VoiceStore.list(this@MainActivity).map { it.name }
                            if (names.isEmpty()) toast("No voices")
                            else MaterialAlertDialogBuilder(this@MainActivity)
                                .setTitle("Re-run with voice")
                                .setItems(voiceLabels(names)) { _, which -> rerunJob(j, names[which]) }
                                .show()
                        }
                    }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6) })
                })
            }
            if (running) {
                jobCard.strokeWidth = dp(2)
                jobCard.strokeColor = themeColor(com.google.android.material.R.attr.colorPrimary)
                jobCard.setCardBackgroundColor(
                    themeColor(com.google.android.material.R.attr.colorSecondaryContainer))
            }
            jobsList.addView(jobCard)
        }
        if (jobsList.childCount == 0) {
            jobsList.addView(TextView(this).apply { text = "No jobs yet."; alpha = 0.6f })
        }
    }

    /** A failed or stopped job keeps its generated chunks, so resuming only
     *  pays for what is missing. The engine is picked here, not guessed: a job
     *  that died on the GPU is exactly when CPU is worth choosing. */
    private fun resumeDialog(j: JobStore.Job, cachedChunks: Int) {
        val options = listOf(
            "CPU" to "cpu",
            "GPU · Vulkan" to "vulkan",
            "GPU · OpenCL" to "opencl",
        )
        val left = if (j.chunksTotal > 0) "${j.chunksTotal - cachedChunks} of ${j.chunksTotal} chunks left"
                   else "$cachedChunks chunks kept"
        // a custom view, not setItems: an AlertDialog shows either a message or
        // a list, never both, and the chunk count is the reason to read the list
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(24), dp(4), dp(24), dp(4))
            addView(TextView(context).apply {
                text = "$left. Generated audio is reused as-is."
                textSize = 14f; alpha = 0.8f
                setPadding(0, 0, 0, dp(8))
            })
        }
        val dialog = MaterialAlertDialogBuilder(this)
            .setTitle("Resume with which engine?")
            .setView(ScrollView(this).apply { addView(col) })
            .setNegativeButton("Cancel", null)
            .create()
        for ((label, id) in options) {
            col.addView(Button(this).apply {
                text = if (id == j.backend) "$label — same as the failed run" else label
                setOnClickListener { dialog.dismiss(); resumeJob(j, id) }
                layoutParams = LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT
                ).apply { bottomMargin = dp(6) }
            })
        }
        dialog.show()
    }

    private fun resumeJob(j: JobStore.Job, backend: String) {
        val v = VoiceStore.list(this).find { it.name == j.voice }
        if (v == null) { toast("Voice “${j.voice}” is gone — use Start over with another voice"); return }
        if (ModelManager.selectedModel(this) == null) { toast("Download a model first (Settings tab)"); return }
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, j.text)
            .putExtra(TtsService.EXTRA_TITLE, j.title)
            .putExtra(TtsService.EXTRA_VOICE, v.name)
            .putExtra(TtsService.EXTRA_BACKEND, backend)
            .putExtra(TtsService.EXTRA_SAVE, j.save)
            .putExtra(TtsService.EXTRA_JOB_ID, j.id))
        toast("Resuming on $backend")
        selectTab(TAB_JOBS)
        ui.postDelayed({ refreshJobsIfChanged() }, 500)
    }

    private fun rerunJob(j: JobStore.Job, voiceName: String) {
        if (!startTtsJob(j.text, j.title, j.save, voiceName)) return
        toast("Re-running “${j.title.ifBlank { j.text.take(24) }}” with $voiceName")
        selectTab(TAB_JOBS)
    }

    // ---- Settings tab ------------------------------------------------------

    // ---- Hosting tab -------------------------------------------------------

    /** Turns the phone into a speech server: an OpenAI-compatible endpoint and
     *  a browser client for it, both on the same port. Everything stays on the
     *  device — this is about reaching it from a laptop, not about the cloud. */
    private fun buildHostTab(): View {
        val (scroll, col) = page()
        col.title("Hosting")
        col.caption("Serve this phone's voices as an OpenAI-compatible speech API, and a web page " +
            "that uses it. Anything on your network can call it; nothing leaves the phone.")

        val port = prefs().getInt("host_port", HostingService.DEFAULT_PORT)
        val addresses = HttpServer.addresses()
        val lan = addresses.firstOrNull()
        val base = "http://${lan ?: "127.0.0.1"}:$port"
        val up = HostingService.running

        col.addView(card {
            val head = LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            }
            // a running server should be obvious from across the room
            head.addView(TextView(context).apply {
                text = if (up) "●" else "●"
                textSize = 20f
                setTextColor(if (up) 0xFF4CAF50.toInt() else 0xFF9E9E9E.toInt())
                setPadding(0, 0, dp(8), 0)
            })
            head.addView(TextView(context).apply {
                text = if (up) "Running" else "Stopped"
                textSize = 17f; setTypeface(typeface, Typeface.BOLD)
                setTextColor(if (up) 0xFF4CAF50.toInt() else themeColor(
                    com.google.android.material.R.attr.colorOnSurface))
            })
            addView(head)
            addView(TextView(context).apply {
                text = if (up) base
                       else HostingService.lastBindError?.let { "Last attempt failed: $it" }
                           ?: "Start it to get an address"
                textSize = 14f; alpha = 0.8f
                setPadding(0, dp(2), 0, dp(8))
            })
            if (up && addresses.size > 1) {
                addView(TextView(context).apply {
                    // a 100.x address is Tailscale's, and it is the one that
                    // still works when the phone leaves the house
                    text = "Also reachable on " + addresses.drop(1).joinToString(", ") { "$it:$port" }
                    textSize = 12f; alpha = 0.7f
                    setPadding(0, 0, 0, dp(6))
                })
            }
            if (up) {
                addView(TextView(context).apply {
                    val s = HostingService.server
                    text = "${s?.requests ?: 0} requests · " +
                        (s?.lastActivity ?: "nothing asked yet")
                    textSize = 12f; alpha = 0.7f
                    setPadding(0, 0, 0, dp(6))
                })
            }
            if (lan == null) {
                addView(TextView(context).apply {
                    text = "No network address — on mobile data without Tailscale, only the phone " +
                        "itself can reach it."
                    textSize = 12f; alpha = 0.7f
                })
            }

            val portRow = LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL
            }
            portRow.addView(TextView(context).apply {
                text = "Port"; textSize = 13f; alpha = 0.7f
                setPadding(0, 0, dp(10), 0)
            })
            val portEdit = EditText(context).apply {
                setText(port.toString())
                inputType = android.text.InputType.TYPE_CLASS_NUMBER
                setSingleLine()
                isEnabled = !HostingService.running
            }
            portRow.addView(portEdit, LinearLayout.LayoutParams(dp(110), ViewGroup.LayoutParams.WRAP_CONTENT))
            addView(portRow)

            addView(Button(context).apply {
                text = if (up) "Stop server" else "Start server"
                setOnClickListener {
                    if (HostingService.running) {
                        HostingService.stop(this@MainActivity)
                    } else {
                        val p = portEdit.text.toString().toIntOrNull()
                        if (p == null || p !in 1024..65535) {
                            toast("Pick a port between 1024 and 65535"); return@setOnClickListener
                        }
                        prefs().edit().putInt("host_port", p).apply()
                        HostingService.start(this@MainActivity, p)
                    }
                    // the service broadcasts, but rebuilding after a beat is
                    // what makes the button flip without a second visit
                    ui.postDelayed({ if (currentTab == TAB_HOST) selectTab(TAB_HOST) }, 700)
                }
            })
        })

        col.addView(card {
            addView(TextView(context).apply {
                text = "Browser client"; textSize = 15f; setTypeface(typeface, Typeface.BOLD)
            })
            addView(TextView(context).apply {
                text = "Open $base on any device here: type text, pick a speaker, hear it back. " +
                    "The page calls the same endpoints as any other client."
                textSize = 13f; alpha = 0.75f
                setPadding(0, dp(2), 0, dp(8))
            })
            val row = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL }
            row.addView(Button(context).apply {
                text = "Open"
                setOnClickListener {
                    if (!HostingService.running) { toast("Start the server first"); return@setOnClickListener }
                    runCatching {
                        startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(base)))
                    }.onFailure { toast("No browser to open it with") }
                }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            row.addView(Button(context).apply {
                text = "Copy address"
                setOnClickListener {
                    (getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager)
                        .setPrimaryClip(android.content.ClipData.newPlainText("TTS Runner", base))
                    toast("Copied $base")
                }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(row)
        })

        col.addView(card {
            addView(TextView(context).apply {
                text = "API specification"; textSize = 15f; setTypeface(typeface, Typeface.BOLD)
            })
            addView(TextView(context).apply {
                text = "OpenAPI 3.1 for POST /v1/audio/speech, /v1/models and /v1/audio/voices. " +
                    "Hand it to a generator and you have a client in any language."
                textSize = 13f; alpha = 0.75f
                setPadding(0, dp(2), 0, dp(8))
            })
            val row = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL }
            row.addView(Button(context).apply {
                text = "Share"
                setOnClickListener { shareSpec(base) }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            row.addView(Button(context).apply {
                text = "Save"
                setOnClickListener {
                    startActivityForResult(Intent(Intent.ACTION_CREATE_DOCUMENT)
                        .addCategory(Intent.CATEGORY_OPENABLE)
                        .setType("application/yaml")
                        .putExtra(Intent.EXTRA_TITLE, "openapi.yaml"), REQ_SAVE_SPEC)
                }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(row)
        })

        col.addView(card {
            addView(TextView(context).apply {
                text = "How requests are handled"; textSize = 15f; setTypeface(typeface, Typeface.BOLD)
            })
            addView(TextView(context).apply {
                text = "Callers can arrive together: each request becomes a job in the queue and " +
                    "they are generated in order, since there is one model loaded. Each shows up " +
                    "in Jobs while it runs and disappears when it is delivered; a failed one stays " +
                    "so you can see why. There is no authentication, so keep this to networks you trust."
                textSize = 13f; alpha = 0.75f
            })
        })

        return scroll
    }

    /** The spec with this server's address baked into it, so whoever receives
     *  it can point a generated client straight at the phone. */
    private fun specText(base: String): String =
        assets.open("openapi.yaml").use { it.readBytes() }.decodeToString()
            .replace("{{SERVER}}", base)

    private fun shareSpec(base: String) {
        runCatching {
            val f = File(File(cacheDir, "share").apply { mkdirs() }, "openapi.yaml")
            f.writeText(specText(base))
            val uri = androidx.core.content.FileProvider.getUriForFile(this, "$packageName.files", f)
            startActivity(Intent.createChooser(
                Intent(Intent.ACTION_SEND)
                    .setType("application/yaml")
                    .putExtra(Intent.EXTRA_STREAM, uri)
                    .putExtra(Intent.EXTRA_SUBJECT, "TTS Runner speech API")
                    .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),
                "Share the API spec"))
        }.onFailure { toast("Share failed: ${it.message}") }
    }

    private lateinit var modelStatus: TextView
    private lateinit var modelBtn: Button
    private lateinit var modelProgress: ProgressBar
    private lateinit var modelGroup: RadioGroup
    private var downloading = false

    private fun buildSettingsTab(): View {
        val (scroll, col) = page()
        col.title("Settings")
        col.caption("v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")

        col.addView(card {
            addView(TextView(context).apply { text = "Model"; textSize = 17f; setTypeface(typeface, Typeface.BOLD) })
            modelGroup = RadioGroup(context)
            for ((i, m) in ModelManager.CATALOG.withIndex()) {
                modelGroup.addView(RadioButton(context).apply { text = m.label; id = i })
            }
            addView(modelGroup)
            modelStatus = TextView(context)
            addView(modelStatus)
            modelProgress = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal)
            addView(modelProgress)
            modelBtn = Button(context).apply { setOnClickListener { onModelButton() } }
            addView(modelBtn)
            modelGroup.setOnCheckedChangeListener { _, id ->
                ModelManager.CATALOG.getOrNull(id)?.let { ModelManager.selectModel(this@MainActivity, it.id) }
                refreshModelUi()
                // the Engine card below belongs to the model: rebuild the tab so
                // it stops offering llama.cpp backends to Supertonic and back
                ui.post { if (currentTab == TAB_SETTINGS) selectTab(TAB_SETTINGS) }
            }
        })

        col.addView(card {
            // The engines have different backends, so this card follows the
            // selected model: offering "Vulkan" to Supertonic meant nothing.
            val model = ModelManager.selectedModel(this@MainActivity)
            val engine = model?.engine ?: "llama"
            addView(TextView(context).apply {
                text = "Engine for ${model?.label ?: "the selected model"}"
                textSize = 17f; setTypeface(typeface, Typeface.BOLD)
            })
            val options = Backends.options(engine)
            val backendGroup = RadioGroup(context)
            val buttons = options.mapIndexed { i, o ->
                RadioButton(context).apply {
                    text = "${o.label} — ${o.why}"
                    id = 100 + i
                }.also { backendGroup.addView(it) }
            }
            addView(backendGroup)
            val detectNote = TextView(context).apply { textSize = 12f; alpha = 0.7f; text = "Detecting GPU…" }
            addView(detectNote)
            val stored = Backends.current(this@MainActivity, engine)
            backendGroup.check(100 + options.indexOfFirst { it.id == stored }.coerceAtLeast(0))
            backendGroup.setOnCheckedChangeListener { _, id ->
                options.getOrNull(id - 100)?.let { Backends.set(this@MainActivity, engine, it.id) }
            }
            thread {
                val info = runCatching { TtsEngine.nDeviceInfo() }.getOrDefault("")
                val (rec, why) = Backends.recommend(engine, info, model?.gpuCapable ?: false)
                val gpuName = Backends.gpuName(info)
                runOnUiThread {
                    val idx = options.indexOfFirst { it.id == rec }
                    if (idx >= 0) buttons[idx].text = "${buttons[idx].text}  ★ recommended"
                    detectNote.text = "Detected GPU: $gpuName. Recommended here because $why. " +
                        "Your choice is kept: a job that fails on one engine can be resumed on " +
                        "another from the Jobs tab."
                }
            }
            val pm = getSystemService(android.os.PowerManager::class.java)
            if (!pm.isIgnoringBatteryOptimizations(packageName)) {
                addView(Button(context).apply {
                    text = "Allow background generation (battery)"
                    setOnClickListener {
                        startActivity(Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                            Uri.parse("package:$packageName")))
                    }
                })
            }
        })

        col.addView(card {
            addView(TextView(context).apply {
                text = "On-device voice cloning"
                textSize = 17f; setTypeface(typeface, Typeface.BOLD)
            })
            val state = TextView(context).apply {
                textSize = 13f; alpha = 0.75f
                setPadding(0, dp(2), 0, dp(8))
            }
            val bar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 1000
                visibility = View.GONE
            }
            lateinit var btn: Button
            fun paint() {
                val have = ClonerModel.installed(this@MainActivity)
                state.text = if (have)
                    "Installed. Clone a recording into a Supertonic speaker from the Speakers tab."
                else "89 MB. Predicts a speaker style from a recording in a second, on the phone.\n" +
                    "Experimental: it scores about 0.22 speaker similarity against 0.82 for the " +
                    "desktop cloning tool, so expect a voice in the right family rather than the " +
                    "person themselves."
                btn.text = if (have) "Remove" else "Download encoder"
            }
            btn = Button(context).apply {
                setOnClickListener {
                    if (ClonerModel.installed(this@MainActivity)) {
                        ClonerModel.remove(this@MainActivity)
                        paint(); rebuildVoices()
                        return@setOnClickListener
                    }
                    isEnabled = false
                    bar.visibility = View.VISIBLE
                    thread {
                        val err = ClonerModel.download(this@MainActivity) { d, t ->
                            runOnUiThread { bar.progress = (d * 1000 / t.coerceAtLeast(1)).toInt() }
                        }
                        runOnUiThread {
                            bar.visibility = View.GONE
                            isEnabled = true
                            if (err != null) toast("Download failed: $err")
                            paint(); rebuildVoices()
                        }
                    }
                }
            }
            addView(state)
            addView(bar)
            addView(btn)
            paint()
        })

        col.addView(card {
            addView(TextView(context).apply { text = "Debug"; textSize = 17f; setTypeface(typeface, Typeface.BOLD) })
            val statusText = TextView(context).apply { textSize = 12f; setTypeface(Typeface.MONOSPACE) }
            addView(statusText)
            addView(LinearLayout(context).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(Button(context).apply {
                    text = "Status"
                    setOnClickListener {
                        thread {
                            val s = runCatching { TtsEngine.nDeviceInfo() }.getOrElse { "device query failed: $it" }
                            runOnUiThread { statusText.text = s }
                        }
                    }
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                addView(Button(context).apply {
                    text = "Copy log"
                    setOnClickListener {
                        thread {
                            val report = DebugLog.buildReport(this@MainActivity)
                            runOnUiThread {
                                val cm = getSystemService(android.content.ClipboardManager::class.java)
                                cm.setPrimaryClip(android.content.ClipData.newPlainText("ttsrunner-debug", report))
                                toast("Debug report copied (${report.length / 1024} KB)")
                            }
                        }
                    }
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6) })
                addView(Button(context).apply {
                    text = "Clear log"
                    setOnClickListener { DebugLog.clear(this@MainActivity); toast("Log cleared") }
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6) })
            })
        })
        refreshModelUi()
        return scroll
    }

    private fun currentModel(): ModelManager.CatalogModel =
        ModelManager.CATALOG.getOrNull(modelGroup.checkedRadioButtonId) ?: ModelManager.CATALOG[0]

    private fun refreshModelUi() {
        if (!::modelGroup.isInitialized) return
        if (modelGroup.checkedRadioButtonId == -1) {
            val sel = ModelManager.selectedModel(this)
            modelGroup.check(ModelManager.CATALOG.indexOfFirst { it.id == sel?.id }.takeIf { it >= 0 } ?: 0)
            return // listener re-enters
        }
        val m = currentModel()
        val have = ModelManager.isDownloaded(this, m)
        modelStatus.text = if (have) "Downloaded ✓" else "Not downloaded (${m.totalBytes / 1_000_000} MB)"
        modelProgress.progress = if (have) 100 else 0
        modelBtn.text = if (downloading) "Cancel" else if (have) "Re-download" else
            if (m.quantizeFrom != null) "Get (download + convert)" else "Download"
    }

    private fun onModelButton() {
        if (downloading) { ModelManager.cancelDownload(); return }
        val m = currentModel()
        downloading = true
        refreshModelUi()
        thread {
            ModelManager.download(this, m, object : ModelManager.DownloadListener {
                override fun onProgress(file: String, doneBytes: Long, totalBytes: Long) {
                    runOnUiThread {
                        modelStatus.text = if (totalBytes > 0)
                            "$file — ${doneBytes / 1_000_000} / ${totalBytes / 1_000_000} MB" else file
                        if (totalBytes > 0) modelProgress.progress = (doneBytes * 100 / totalBytes).toInt()
                    }
                }
                override fun onDone() {
                    runOnUiThread { downloading = false; refreshModelUi(); toast("Model ready") }
                }
                override fun onError(message: String) {
                    runOnUiThread { downloading = false; refreshModelUi(); toast("Download: $message") }
                }
            })
        }
    }

    companion object {
        private const val REQ_IMPORT = 11
        private const val REQ_IMPORT_DIR = 12
        private const val REQ_EXPORT_DIR = 13
        private const val REQ_BACKUP_DIR = 14
        private const val REQ_SAVE_SPEC = 15
        private const val REQ_CLONE_AUDIO = 16
        private const val TAB_VOICES = 2
        private const val TAB_JOBS = 3
        private const val TAB_SETTINGS = 4
        private const val TAB_HOST = 5
        private const val TAB_CHATS = 6
    }
}
