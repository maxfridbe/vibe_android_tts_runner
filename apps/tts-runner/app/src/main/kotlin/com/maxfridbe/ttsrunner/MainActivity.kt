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
    private lateinit var swipeTrack: FrameLayout
    private lateinit var swipeBar: View
    private var navSelect: ((Int) -> Unit)? = null
    private val tabs = mutableMapOf<Int, View>()
    private var currentTab = TAB_NEWJOB
    private val tabOrder = listOf(TAB_NEWJOB, TAB_VOICES, TAB_JOBS, TAB_SETTINGS)

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
        currentTab = savedInstanceState?.getInt("tab") ?: TAB_NEWJOB

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
            menu.add(0, TAB_NEWJOB, 0, "New job").setIcon(R.drawable.ic_add)
            menu.add(0, TAB_VOICES, 1, "Voices").setIcon(R.drawable.ic_mic)
            menu.add(0, TAB_JOBS, 2, "Jobs").setIcon(R.drawable.ic_jobs)
            menu.add(0, TAB_SETTINGS, 3, "Settings").setIcon(R.drawable.ic_settings)
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
            // Swipe indicator: one slot per tab, riding the top edge of the
            // navigation bar so the line sits directly over the tab it points
            // at. Bottom nav divides its width evenly, so slot = width / tabs.
            swipeEnabled = true
            swipeBar = View(this).apply {
                setBackgroundColor(themeColor(com.google.android.material.R.attr.colorPrimary))
                alpha = 0f
            }
            swipeTrack = FrameLayout(this).apply {
                // the nav bar carries a Material elevation, so the indicator
                // needs a higher one to draw over its top edge
                elevation = dp(16).toFloat()
                addView(swipeBar, FrameLayout.LayoutParams(0, dp(3)))
                addOnLayoutChangeListener { _, _, _, _, _, _, _, _, _ ->
                    val slot = width / tabOrder.size
                    if (slot > 0 && swipeBar.layoutParams.width != slot) {
                        swipeBar.layoutParams = swipeBar.layoutParams.also { it.width = slot }
                        swipeBar.translationX = slotIndex() * slot.toFloat()
                    }
                }
            }
            val navHolder = FrameLayout(this).apply {
                addView(bottom, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT))
                addView(swipeTrack, FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, dp(3), Gravity.TOP))
            }
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                addView(contentWithMeter, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
                addView(navHolder, LinearLayout.LayoutParams(
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
        if (id == TAB_SETTINGS) tabs.remove(id)  // rebuilt each visit (battery button state etc.)
        val v = tabs.getOrPut(id) {
            when (id) {
                TAB_NEWJOB -> buildNewJobTab()
                TAB_VOICES -> buildVoicesTab()
                TAB_JOBS -> buildJobsTab()
                else -> buildSettingsTab()
            }.noStateSave()
        }
        when (id) {
            TAB_VOICES -> rebuildVoices()
            TAB_JOBS -> rebuildJobs()
            TAB_NEWJOB -> refreshVoiceLabel()
        }
        content.removeAllViews()
        content.addView(v)
        v.noStateSave()
        // keep the swipe indicator in sync with taps on the nav bar too
        if (swipeEnabled && !dragging && swipeBar.width > 0) {
            swipeBar.translationX = slotIndex() * swipeBar.width.toFloat()
        }
    }

    // ---- tab swipe ---------------------------------------------------------

    private var downX = 0f
    private var downY = 0f
    private var dragging = false
    /** Only with the bottom navigation bar. On a tablet or an unfolded Fold the
     *  tabs are a rail on the left, there is nothing for a horizontal indicator
     *  to line up with, and wide layouts have side-by-side content that a
     *  swipe-to-switch gesture would fight with. */
    private var swipeEnabled = false

    private fun slotIndex() = tabOrder.indexOf(currentTab).coerceAtLeast(0)

    /** Every touch is observed here — children still handle their own gestures,
     *  the content just rides along horizontally and the indicator shows which
     *  tab a release would land on. */
    override fun dispatchTouchEvent(ev: android.view.MotionEvent): Boolean {
        if (!swipeEnabled) return super.dispatchTouchEvent(ev)
        when (ev.actionMasked) {
            android.view.MotionEvent.ACTION_DOWN -> {
                downX = ev.x; downY = ev.y; dragging = false
            }
            android.view.MotionEvent.ACTION_MOVE -> {
                val dx = ev.x - downX
                val dy = ev.y - downY
                if (!dragging && kotlin.math.abs(dx) > dp(24) &&
                    kotlin.math.abs(dx) > kotlin.math.abs(dy) * 1.5f) {
                    dragging = true
                    swipeBar.animate().alpha(1f).setStartDelay(0).setDuration(90).start()
                }
                if (dragging) dragSwipe(dx)
            }
            android.view.MotionEvent.ACTION_UP, android.view.MotionEvent.ACTION_CANCEL ->
                if (dragging) { dragging = false; releaseSwipe(ev.x - downX) }
        }
        return super.dispatchTouchEvent(ev)
    }

    private fun dragSwipe(dx: Float) {
        val slot = swipeBar.width.toFloat()
        if (slot <= 0f) return
        val idx = slotIndex()
        // the indicator leads toward the incoming tab, so it moves opposite the
        // finger — the same convention as a pager underline
        var travel = (-dx * 0.5f).coerceIn(-slot, slot)
        if (idx == 0 && travel < 0f) travel = 0f
        if (idx == tabOrder.lastIndex && travel > 0f) travel = 0f
        swipeBar.translationX = idx * slot + travel
        content.translationX = if (travel == 0f) 0f else (dx * 0.12f).coerceIn(-slot, slot)
    }

    private fun releaseSwipe(dx: Float) {
        val slot = swipeBar.width.toFloat()
        content.animate().translationX(0f).setDuration(150).start()
        val next = slotIndex() + if (dx < 0) 1 else -1
        if (kotlin.math.abs(dx) > dp(72) && next in tabOrder.indices) selectTab(tabOrder[next])
        swipeBar.animate().translationX(slotIndex() * slot).setDuration(180)
            .withEndAction { swipeBar.animate().alpha(0f).setStartDelay(140).setDuration(220).start() }
            .start()
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

    // ---- New job tab -------------------------------------------------------

    private fun buildNewJobTab(): View {
        val (scroll, col) = page()
        col.title("New TTS job")
        col.caption("Or share text / an article link to TTS Runner from any app.")

        col.addView(card {
            jobText = EditText(context).apply {
                setText("The quick brown fox jumps over the lazy dog, then reads the entire internet aloud.")
                minLines = 3
                background = null
                hint = "Text to read"
            }
            addView(jobText)

            val optionsRow = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
            voicePickBtn = Button(context).apply { setOnClickListener { pickVoiceDialog() } }
            optionsRow.addView(voicePickBtn)
            val modeGroup = RadioGroup(context).apply { orientation = LinearLayout.HORIZONTAL }
            modeGroup.addView(RadioButton(context).apply { text = "Listen"; id = 1; isChecked = true })
            modeGroup.addView(RadioButton(context).apply { text = "Save file"; id = 2 })
            modeGroup.setOnCheckedChangeListener { _, id -> pickedSave = id == 2 }
            optionsRow.addView(modeGroup, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT).apply { marginStart = dp(12) })
            addView(optionsRow)

            val actions = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL }
            actions.addView(Button(context).apply {
                text = "Add job"
                setOnClickListener {
                    val text = jobText.text.toString()
                    if (text.isBlank()) { toast("Nothing to read"); return@setOnClickListener }
                    val started = startTtsJob(text, if (pickedSave) "TTS " + text.take(24) else "TTS job",
                        save = pickedSave, voiceName = pickedVoice)
                    // a queued job belongs to the Jobs tab — go watch it there
                    if (started) selectTab(TAB_JOBS)
                }
            }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            addView(actions)
        })

        return scroll
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
        val name = pickedVoice ?: VoiceStore.defaultVoice(this)?.name
        voicePickBtn.text =
            if (name == null) "Voice: none ▾" else "${VoiceStore.icon(this, name)} $name ▾"
    }

    /** "🦊 dale" labels for the picker dialogs, in list order. */
    private fun voiceLabels(names: List<String>) =
        names.map { "${VoiceStore.icon(this, it)}  $it" }.toTypedArray()

    private fun pickVoiceDialog() {
        val names = VoiceStore.list(this).map { it.name }
        if (names.isEmpty()) { toast("Import or design a voice first (Voices tab)"); return }
        MaterialAlertDialogBuilder(this)
            .setTitle("Voice for this job")
            .setItems(voiceLabels(names)) { _, which ->
                pickedVoice = names[which]
                refreshVoiceLabel()
            }
            .show()
    }

    private fun startTtsJob(text: String, title: String, save: Boolean, voiceName: String? = null): Boolean {
        val v = voiceName?.let { n -> VoiceStore.list(this).find { it.name == n } } ?: VoiceStore.defaultVoice(this)
        if (v == null) { toast("Import or design a voice first (Voices tab)"); return false }
        if (ModelManager.selectedModel(this) == null) { toast("Download a model first (Settings tab)"); return false }
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, text)
            .putExtra(TtsService.EXTRA_TITLE, title)
            .putExtra(TtsService.EXTRA_VOICE, v.name)
            .putExtra(TtsService.EXTRA_BACKEND, prefs().getString("backend", "cpu"))
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

    private fun buildVoicesTab(): View {
        val (scroll, col) = page()
        col.title("Voices")
        col.caption("A voice is 10–20 s of clean speech. Previews are generated once per model and cached.")
        val topRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        topRow.addView(Button(this).apply {
            text = "Import"
            setOnClickListener {
                startActivityForResult(
                    Intent(Intent.ACTION_OPEN_DOCUMENT)
                        .addCategory(Intent.CATEGORY_OPENABLE)
                        .setType("audio/*"), REQ_IMPORT)
            }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
        topRow.addView(Button(this).apply {
            text = "Design new voice"
            setOnClickListener { designVoiceDialog() }
        }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply { marginStart = dp(6) })
        col.addView(topRow)
        voicesList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(10), 0, 0) }
        col.addView(voicesList)
        return scroll
    }

    private fun rebuildVoices() {
        if (!::voicesList.isInitialized) return
        voicesList.removeAllViews()
        val def = VoiceStore.defaultVoice(this)
        val modelId = currentModelId()
        for (v in VoiceStore.list(this)) {
            voicesList.addView(card {
                val head = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
                head.addView(RadioButton(context).apply {
                    isChecked = v.name == def?.name
                    setOnClickListener {
                        VoiceStore.setDefault(this@MainActivity, v)
                        rebuildVoices(); refreshVoiceLabel()
                    }
                })
                head.addView(TextView(context).apply {
                    text = VoiceStore.icon(this@MainActivity, v.name)
                    textSize = 20f
                    setPadding(dp(2), 0, dp(8), 0)
                })
                head.addView(TextView(context).apply {
                    text = v.name; textSize = 17f; setTypeface(typeface, Typeface.BOLD)
                    maxLines = 1
                    ellipsize = android.text.TextUtils.TruncateAt.END
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                val hasPreview = VoiceStore.previewFile(this@MainActivity, v.name, modelId).exists()
                lateinit var prevBtn: ImageButton
                prevBtn = iconBtn(R.drawable.ic_play,
                    if (hasPreview) "Play preview" else "Generate preview") { previewVoice(v, prevBtn) }
                head.addView(prevBtn)
                head.addView(iconBtn(R.drawable.ic_share, "Share preview") {
                    val p = VoiceStore.previewFile(this@MainActivity, v.name, modelId)
                    if (!p.exists()) toast("Generate a preview first (▶)")
                    else thread {
                        runCatching { AudioShare.shareWavAsM4a(this@MainActivity, p, "Voice ${v.name}") }
                            .onFailure { e -> runOnUiThread { toast("Share failed: ${e.message}") } }
                    }
                })
                head.addView(iconBtn(R.drawable.ic_edit, "Edit speaker") { editSpeakerDialog(v) })
                head.addView(iconBtn(R.drawable.ic_delete, "Delete") {
                    MaterialAlertDialogBuilder(this@MainActivity)
                        .setMessage("Delete voice “${v.name}”?")
                        .setPositiveButton("Delete") { _, _ ->
                            VoiceStore.delete(this@MainActivity, v); rebuildVoices()
                        }
                        .setNegativeButton("Cancel", null).show()
                })
                addView(head)
                val bits = mutableListOf("${v.file.length() / 1024} KB")
                if (!hasPreview) bits.add("no preview for $modelId yet")
                if (VoiceStore.transcriptFile(this@MainActivity, v.name).exists()) bits.add("transcript ✓")
                addView(TextView(context).apply { text = bits.joinToString(" · "); textSize = 12f; alpha = 0.6f })
            })
        }
        if (voicesList.childCount == 0) {
            voicesList.addView(TextView(this).apply {
                text = "No voices yet — import a recording or design one."; alpha = 0.6f
            })
        }
    }

    private fun previewVoice(v: VoiceStore.Voice, btn: ImageButton) {
        val cached = VoiceStore.previewFile(this, v.name, currentModelId())
        if (cached.exists()) { playFile(cached.absolutePath, btn); return }
        if (ModelManager.selectedModel(this) == null) { toast("Download a model first (Settings tab)"); return }
        toast("Generating preview with ${currentModelId()} — plays when ready, then cached")
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, VoiceStore.PREVIEW_TEXT)
            .putExtra(TtsService.EXTRA_TITLE, "Preview: ${v.name}")
            .putExtra(TtsService.EXTRA_VOICE, v.name)
            .putExtra(TtsService.EXTRA_BACKEND, prefs().getString("backend", "cpu"))
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
            text = "Icon"; textSize = 12f; alpha = 0.7f; setPadding(0, dp(14), 0, dp(4))
        })
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
                        setOnClickListener { chosenIcon = emoji; paintCells() }
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
            .setPositiveButton("Save") { _, _ ->
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
            .putExtra(TtsService.EXTRA_BACKEND, prefs().getString("backend", "cpu")))
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

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQ_IMPORT && resultCode == RESULT_OK && data?.data != null) {
            val uri: Uri = data.data!!
            val name = contentResolver.query(uri, null, null, null, null)?.use { c ->
                val idx = c.getColumnIndex(OpenableColumns.DISPLAY_NAME)
                if (c.moveToFirst() && idx >= 0) c.getString(idx) else null
            } ?: "voice.wav"
            try {
                VoiceStore.import(this, uri, name)
                rebuildVoices()
            } catch (e: Exception) {
                toast("Import failed: ${e.message}")
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

    private fun refreshJobsIfChanged() {
        if (currentTab != TAB_JOBS || !::jobsList.isInitialized) return
        val now = android.os.SystemClock.uptimeMillis()
        if (now - lastJobsCheck < 1500) return
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
                        "${j.text.length} chars · ${j.voice} · ${j.model}/${j.backend}$stats" +
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
                addView(LinearLayout(context).apply {
                    orientation = LinearLayout.HORIZONTAL
                    addView(Button(context).apply {
                        text = "Re-run"
                        setOnClickListener { rerunJob(j, j.voice) }
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

    private fun rerunJob(j: JobStore.Job, voiceName: String) {
        if (!startTtsJob(j.text, j.title, j.save, voiceName)) return
        toast("Re-running “${j.title.ifBlank { j.text.take(24) }}” with $voiceName")
        selectTab(TAB_JOBS)
    }

    // ---- Settings tab ------------------------------------------------------

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
            }
        })

        col.addView(card {
            addView(TextView(context).apply { text = "Engine"; textSize = 17f; setTypeface(typeface, Typeface.BOLD) })
            val backendGroup = RadioGroup(context)
            val cpuBtn = RadioButton(context).apply { text = "CPU"; id = 100 }
            val clBtn = RadioButton(context).apply { text = "GPU · OpenCL (Adreno, Q4_0 model only)"; id = 101 }
            val vkBtn = RadioButton(context).apply { text = "GPU · Vulkan (any model)"; id = 102 }
            backendGroup.addView(cpuBtn); backendGroup.addView(clBtn); backendGroup.addView(vkBtn)
            addView(backendGroup)
            val detectNote = TextView(context).apply { textSize = 12f; alpha = 0.7f; text = "Detecting GPU…" }
            addView(detectNote)
            val stored = when (val b = prefs().getString("backend", "cpu")) { "gpu" -> "opencl"; else -> b }
            backendGroup.check(when (stored) { "opencl" -> 101; "vulkan" -> 102; else -> 100 })
            val gpuNote = TextView(context).apply {
                textSize = 12f
                text = "A GPU run was killed for memory, so jobs are on CPU. Pick a GPU " +
                    "option again to retry it."
                visibility = if (prefs().getBoolean("gpu_unstable", false)) View.VISIBLE else View.GONE
            }
            addView(gpuNote)
            backendGroup.setOnCheckedChangeListener { _, id ->
                val chosen = when (id) { 101 -> "opencl"; 102 -> "vulkan"; else -> "cpu" }
                // choosing a GPU again is the explicit retry that clears the
                // memory-kill lockout the engine set
                prefs().edit().putString("backend", chosen)
                    .putBoolean("gpu_unstable", false).apply()
                gpuNote.visibility = View.GONE
            }
            thread {
                val info = runCatching { TtsEngine.nDeviceInfo() }.getOrDefault("")
                val hasAdrenoCl = info.contains("OpenCL") && info.contains("Adreno")
                val vulkanLine = info.lines().find { it.contains("Vulkan") } ?: ""
                val rec = when {
                    hasAdrenoCl -> "opencl"
                    vulkanLine.isNotEmpty() && !vulkanLine.contains("Adreno") -> "vulkan"
                    else -> "cpu"
                }
                val gpuName = Regex("— ([^(]+)\\(").find(vulkanLine)?.groupValues?.get(1)?.trim()
                    ?: if (hasAdrenoCl) "Adreno" else "none"
                runOnUiThread {
                    when (rec) {
                        "opencl" -> clBtn.text = "${clBtn.text}  ★ recommended"
                        "vulkan" -> vkBtn.text = "${vkBtn.text}  ★ recommended"
                        else -> cpuBtn.text = "${cpuBtn.text}  ★ recommended"
                    }
                    detectNote.text = "Detected GPU: $gpuName. A GPU engine that fails to start falls back to CPU automatically."
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
        private const val TAB_NEWJOB = 1
        private const val TAB_VOICES = 2
        private const val TAB_JOBS = 3
        private const val TAB_SETTINGS = 4
    }
}
