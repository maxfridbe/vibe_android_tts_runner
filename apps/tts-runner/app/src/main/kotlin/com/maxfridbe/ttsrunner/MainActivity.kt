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
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.google.android.material.card.MaterialCardView
import com.google.android.material.color.DynamicColors
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.navigationrail.NavigationRailView
import kotlin.concurrent.thread

/** Tabbed UI in the audiobook-maker mold: Listen / Voices / Jobs / Settings.
 *  One programmatic layout, responsive: bottom navigation on phones, a
 *  navigation rail on tablets and the unfolded Fold (width >= 600dp). */
class MainActivity : AppCompatActivity() {

    private lateinit var content: FrameLayout
    private val tabs = mutableMapOf<Int, View>()
    private var currentTab = TAB_LISTEN

    // Listen tab widgets (referenced from the status receiver)
    private lateinit var runStatus: TextView
    private lateinit var runProgress: ProgressBar
    private lateinit var etaText: TextView
    private lateinit var statsText: TextView
    private lateinit var waveform: WaveformView
    private lateinit var playBtn: Button
    private lateinit var voiceLabel: TextView
    private var player: android.media.MediaPlayer? = null
    private var previewPlayer: android.media.MediaPlayer? = null
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
            etaText.text = if (eta >= 0) "≈ ${if (eta >= 60) "${eta / 60}m ${eta % 60}s" else "${eta}s"} remaining" else ""
            runStatus.text = when (state) {
                "loading" -> { runProgress.progress = 0; "Loading model…" }
                "generating" -> {
                    if (total > 0) runProgress.progress = chunk * 1000 / total
                    "Chunk ${chunk + 1}/$total: “$message…”"
                }
                "frames" ->
                    if (chunk == 0) "Processing text & voice…"
                    else "Generating… ${"%.1f".format(chunk / 12.5)} s of audio"
                "done" -> { finishUi(); statsText.text = message; "Done" }
                "saved" -> { finishUi(); statsText.text = message; "Saved" }
                "stopped" -> { runProgress.progress = 0; etaText.text = ""; "Stopped" }
                "note" -> message
                "error" -> { runProgress.progress = 0; etaText.text = ""; "Error: $message" }
                else -> state ?: ""
            }
        }
    }

    private fun finishUi() {
        runProgress.progress = 1000
        etaText.text = ""
        refreshPlayer()
        (tabs[TAB_JOBS] as? ViewGroup)?.let { rebuildJobs() }
    }

    // ---- activity ----------------------------------------------------------

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        DynamicColors.applyToActivityIfAvailable(this)
        currentTab = savedInstanceState?.getInt("tab") ?: TAB_LISTEN

        if (android.os.Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) !=
                android.content.pm.PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 12)
        }

        content = FrameLayout(this)
        val wide = resources.configuration.screenWidthDp >= 600

        val navItems = { menu: Menu ->
            menu.add(0, TAB_LISTEN, 0, "Listen").setIcon(android.R.drawable.ic_media_play)
            menu.add(0, TAB_VOICES, 1, "Voices").setIcon(android.R.drawable.ic_btn_speak_now)
            menu.add(0, TAB_JOBS, 2, "Jobs").setIcon(android.R.drawable.ic_menu_recent_history)
            menu.add(0, TAB_SETTINGS, 3, "Settings").setIcon(android.R.drawable.ic_menu_preferences)
        }

        val root: View = if (wide) {
            val rail = NavigationRailView(this)
            navItems(rail.menu)
            rail.setOnItemSelectedListener { showTab(it.itemId); true }
            rail.selectedItemId = currentTab
            LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                addView(rail, LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.MATCH_PARENT))
                addView(content, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.MATCH_PARENT, 1f))
            }
        } else {
            val bottom = BottomNavigationView(this)
            navItems(bottom.menu)
            bottom.setOnItemSelectedListener { showTab(it.itemId); true }
            bottom.selectedItemId = currentTab
            LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                addView(content, LinearLayout.LayoutParams(
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

    private fun showTab(id: Int) {
        currentTab = id
        val v = tabs.getOrPut(id) {
            when (id) {
                TAB_LISTEN -> buildListenTab()
                TAB_VOICES -> buildVoicesTab()
                TAB_JOBS -> buildJobsTab()
                else -> buildSettingsTab()
            }
        }
        when (id) {
            TAB_VOICES -> rebuildVoices()
            TAB_JOBS -> rebuildJobs()
            TAB_LISTEN -> { refreshVoiceLabel(); refreshPlayer() }
        }
        content.removeAllViews()
        content.addView(v)
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

    override fun onResume() {
        super.onResume()
        registerReceiver(statusReceiver, IntentFilter(TtsService.STATUS_BROADCAST), RECEIVER_NOT_EXPORTED)
        if (::playBtn.isInitialized) refreshPlayer()
    }

    override fun onPause() {
        super.onPause()
        unregisterReceiver(statusReceiver)
    }

    override fun onDestroy() {
        player?.release(); player = null
        previewPlayer?.release(); previewPlayer = null
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

    private fun buttonRow(vararg buttons: Pair<String, () -> Unit>): LinearLayout =
        LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            for ((label, action) in buttons) {
                addView(Button(context).apply {
                    text = label
                    setOnClickListener { action() }
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    .apply { marginEnd = dp(6) })
            }
        }

    private fun toast(t: String) = Toast.makeText(this, t, Toast.LENGTH_LONG).show()

    // ---- Listen tab --------------------------------------------------------

    private lateinit var tryText: EditText

    private fun buildListenTab(): View {
        val (scroll, col) = page()
        col.title("Listen")
        col.caption("Share text or a link from any app — or paste here. v${BuildConfig.VERSION_NAME}")

        col.addView(card {
            tryText = EditText(context).apply {
                setText("The quick brown fox jumps over the lazy dog, then reads the entire internet aloud.")
                minLines = 3
                background = null
                hint = "Text to read"
            }
            addView(tryText)
            voiceLabel = TextView(context).apply { textSize = 13f; alpha = 0.7f; setPadding(0, dp(4), 0, dp(8)) }
            addView(voiceLabel)
            addView(buttonRow(
                "Speak" to { startTtsJob(tryText.text.toString(), "Test", save = false) },
                "Save audio" to { startTtsJob(tryText.text.toString(), "TTS " + tryText.text.take(24), save = true) },
                "Stop" to { startService(Intent(this@MainActivity, TtsService::class.java).setAction(TtsService.ACTION_STOP)) },
            ))
        })

        col.addView(card {
            runStatus = TextView(context)
            addView(runStatus)
            runProgress = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply { max = 1000 }
            addView(runProgress)
            etaText = TextView(context).apply { textSize = 13f; alpha = 0.7f }
            addView(etaText)
            statsText = TextView(context).apply { textSize = 13f; alpha = 0.7f }
            addView(statsText)

            val playerRow = LinearLayout(context).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
            playBtn = Button(context).apply { text = "▶"; setOnClickListener { togglePlayback() } }
            playerRow.addView(playBtn)
            waveform = WaveformView(context).apply {
                onSeek = { frac -> player?.let { it.seekTo((it.duration * frac).toInt()) } }
            }
            playerRow.addView(waveform, LinearLayout.LayoutParams(0, dp(52), 1f).apply { marginStart = dp(8) })
            playerRow.addView(Button(context).apply {
                text = "Share"
                setOnClickListener {
                    val f = lastAudio()
                    if (!f.exists() || f.length() <= 44) { toast("No audio yet"); return@setOnClickListener }
                    thread {
                        runCatching { AudioShare.shareWavAsM4a(this@MainActivity, f, "TTS audio") }
                            .onFailure { e -> runOnUiThread { toast("Share failed: ${e.message}") } }
                    }
                }
            })
            addView(playerRow)
        })
        return scroll
    }

    private fun refreshVoiceLabel() {
        if (!::voiceLabel.isInitialized) return
        val v = VoiceStore.defaultVoice(this)
        val m = ModelManager.selectedModel(this)
        val backend = getSharedPreferences("ttsrunner", MODE_PRIVATE).getString("backend", "cpu")
        voiceLabel.text = "Voice: ${v?.name ?: "none"} · Model: ${m?.id ?: "none"} · ${backend?.uppercase()}"
    }

    private fun startTtsJob(text: String, title: String, save: Boolean, voiceName: String? = null) {
        val v = voiceName?.let { n -> VoiceStore.list(this).find { it.name == n } } ?: VoiceStore.defaultVoice(this)
        if (v == null) { toast("Import a voice first (Voices tab)"); return }
        if (ModelManager.selectedModel(this) == null) { toast("Download a model first (Settings tab)"); return }
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, text)
            .putExtra(TtsService.EXTRA_TITLE, title)
            .putExtra(TtsService.EXTRA_VOICE, v.name)
            .putExtra(TtsService.EXTRA_BACKEND, getSharedPreferences("ttsrunner", MODE_PRIVATE).getString("backend", "cpu"))
            .putExtra(TtsService.EXTRA_SAVE, save))
    }

    // ---- last-audio player -------------------------------------------------

    private fun lastAudio() = java.io.File(filesDir, "last_audio.wav")

    private fun refreshPlayer() {
        if (!::playBtn.isInitialized) return
        val f = lastAudio()
        val visible = f.exists() && f.length() > 44
        playBtn.isEnabled = visible
        if (visible) thread { waveform.loadWav(f) }
    }

    private val progressTicker = object : Runnable {
        override fun run() {
            val p = player ?: return
            if (p.isPlaying) {
                if (p.duration > 0) waveform.setProgress(p.currentPosition / p.duration.toFloat())
                ui.postDelayed(this, 200)
            }
        }
    }

    private fun togglePlayback() {
        val p = player
        if (p != null && p.isPlaying) { p.pause(); playBtn.text = "▶"; return }
        if (p != null) { p.start(); playBtn.text = "❚❚"; ui.post(progressTicker); return }
        try {
            player = android.media.MediaPlayer().apply {
                setDataSource(lastAudio().absolutePath)
                prepare()
                setOnCompletionListener {
                    playBtn.text = "▶"; waveform.setProgress(1f); it.release(); player = null
                }
                start()
            }
            playBtn.text = "❚❚"
            ui.post(progressTicker)
        } catch (e: Exception) {
            toast("Playback failed: ${e.message}")
            player?.release(); player = null
        }
    }

    // ---- Voices tab --------------------------------------------------------

    private lateinit var voicesList: LinearLayout

    private fun buildVoicesTab(): View {
        val (scroll, col) = page()
        col.title("Voices")
        col.caption("A voice is 10–20 s of clean speech from one speaker. Previews are generated once and cached.")
        col.addView(Button(this).apply {
            text = "Import voice"
            setOnClickListener {
                startActivityForResult(
                    Intent(Intent.ACTION_OPEN_DOCUMENT)
                        .addCategory(Intent.CATEGORY_OPENABLE)
                        .setType("audio/*"), REQ_IMPORT)
            }
        })
        voicesList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(10), 0, 0) }
        col.addView(voicesList)
        return scroll
    }

    private fun rebuildVoices() {
        if (!::voicesList.isInitialized) return
        voicesList.removeAllViews()
        val def = VoiceStore.defaultVoice(this)
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
                    text = v.name; textSize = 17f; setTypeface(typeface, Typeface.BOLD)
                }, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
                head.addView(TextView(context).apply {
                    text = "${v.file.length() / 1024} KB"; textSize = 12f; alpha = 0.6f
                })
                addView(head)
                val hasPreview = VoiceStore.previewFile(this@MainActivity, v.name).exists()
                addView(buttonRow(
                    (if (hasPreview) "▶ Preview" else "Make preview") to { previewVoice(v) },
                    "Share" to {
                        val p = VoiceStore.previewFile(this@MainActivity, v.name)
                        if (!p.exists()) toast("Make a preview first")
                        else thread {
                            runCatching { AudioShare.shareWavAsM4a(this@MainActivity, p, "Voice preview ${v.name}") }
                                .onFailure { e -> runOnUiThread { toast("Share failed: ${e.message}") } }
                        }
                    },
                    "Delete" to {
                        MaterialAlertDialogBuilder(this@MainActivity)
                            .setMessage("Delete voice “${v.name}”?")
                            .setPositiveButton("Delete") { _, _ ->
                                VoiceStore.delete(this@MainActivity, v); rebuildVoices()
                            }
                            .setNegativeButton("Cancel", null).show()
                    },
                ))
            })
        }
        if (voicesList.childCount == 0) {
            voicesList.addView(TextView(this).apply { text = "No voices yet — import one above."; alpha = 0.6f })
        }
    }

    private fun previewVoice(v: VoiceStore.Voice) {
        val cached = VoiceStore.previewFile(this, v.name)
        if (cached.exists()) {
            previewPlayer?.release()
            try {
                previewPlayer = android.media.MediaPlayer().apply {
                    setDataSource(cached.absolutePath)
                    prepare()
                    setOnCompletionListener { it.release(); previewPlayer = null }
                    start()
                }
            } catch (e: Exception) { toast("Preview failed: ${e.message}") }
            return
        }
        if (ModelManager.selectedModel(this) == null) { toast("Download a model first (Settings tab)"); return }
        toast("Generating preview — it will play when ready and be cached")
        startForegroundService(Intent(this, TtsService::class.java)
            .setAction(TtsService.ACTION_SPEAK)
            .putExtra(TtsService.EXTRA_TEXT, VoiceStore.PREVIEW_TEXT)
            .putExtra(TtsService.EXTRA_TITLE, "Preview: ${v.name}")
            .putExtra(TtsService.EXTRA_VOICE, v.name)
            .putExtra(TtsService.EXTRA_BACKEND, getSharedPreferences("ttsrunner", MODE_PRIVATE).getString("backend", "cpu"))
            .putExtra(TtsService.EXTRA_PREVIEW, true))
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

    private fun buildJobsTab(): View {
        val (scroll, col) = page()
        col.title("Jobs")
        jobsSummary = TextView(this).apply { textSize = 13f; alpha = 0.7f; setPadding(0, 0, 0, dp(12)) }
        col.addView(jobsSummary)
        jobsList = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL }
        col.addView(jobsList)
        return scroll
    }

    private fun rebuildJobs() {
        if (!::jobsList.isInitialized) return
        jobsList.removeAllViews()
        val jobs = JobStore.list(this)
        val totalAudio = jobs.sumOf { it.audioSecs }
        jobsSummary.text = "${jobs.size} jobs · ${"%.0f".format(totalAudio / 60)} min of audio generated"
        val fmt = java.text.SimpleDateFormat("MMM d HH:mm", java.util.Locale.US)
        for (j in jobs) {
            jobsList.addView(card {
                addView(TextView(context).apply {
                    text = j.title.ifBlank { j.text.take(40) }
                    textSize = 16f; setTypeface(typeface, Typeface.BOLD)
                })
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
                addView(buttonRow(
                    "Re-run" to { rerunJob(j, j.voice) },
                    "Share" to {
                        if (j.outputUri.isNotBlank())
                            AudioShare.shareUri(this@MainActivity, Uri.parse(j.outputUri), j.title)
                        else toast("No saved file for this job (use Save mode, or Share from the Listen player)")
                    },
                    "Other voice…" to {
                        val names = VoiceStore.list(this@MainActivity).map { it.name }.toTypedArray()
                        if (names.isEmpty()) toast("No voices")
                        else MaterialAlertDialogBuilder(this@MainActivity)
                            .setTitle("Re-run with voice")
                            .setItems(names) { _, which -> rerunJob(j, names[which]) }
                            .show()
                    },
                    "Delete" to { JobStore.delete(this@MainActivity, j.id); rebuildJobs() },
                ))
            })
        }
        if (jobsList.childCount == 0) {
            jobsList.addView(TextView(this).apply { text = "No jobs yet."; alpha = 0.6f })
        }
    }

    private fun rerunJob(j: JobStore.Job, voiceName: String) {
        startTtsJob(j.text, j.title, j.save, voiceName)
        toast("Re-running “${j.title.ifBlank { j.text.take(24) }}” with $voiceName")
        showTab(TAB_LISTEN)
    }

    // ---- Settings tab ------------------------------------------------------

    private lateinit var modelStatus: TextView
    private lateinit var modelBtn: Button
    private lateinit var modelProgress: ProgressBar
    private lateinit var modelGroup: android.widget.RadioGroup
    private var downloading = false

    private fun buildSettingsTab(): View {
        val (scroll, col) = page()
        col.title("Settings")

        col.addView(card {
            addView(TextView(context).apply { text = "Model"; textSize = 17f; setTypeface(typeface, Typeface.BOLD) })
            modelGroup = android.widget.RadioGroup(context)
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
                refreshModelUi(); refreshVoiceLabel()
            }
        })

        col.addView(card {
            addView(TextView(context).apply { text = "Engine"; textSize = 17f; setTypeface(typeface, Typeface.BOLD) })
            val prefs = getSharedPreferences("ttsrunner", MODE_PRIVATE)
            val backendGroup = android.widget.RadioGroup(context)
            val cpuBtn = RadioButton(context).apply { text = "CPU"; id = 100 }
            val clBtn = RadioButton(context).apply { text = "GPU · OpenCL (Adreno, Q4_0 model only)"; id = 101 }
            val vkBtn = RadioButton(context).apply { text = "GPU · Vulkan (any model)"; id = 102 }
            backendGroup.addView(cpuBtn); backendGroup.addView(clBtn); backendGroup.addView(vkBtn)
            addView(backendGroup)
            val detectNote = TextView(context).apply { textSize = 12f; alpha = 0.7f; text = "Detecting GPU…" }
            addView(detectNote)
            val stored = when (val b = prefs.getString("backend", "cpu")) { "gpu" -> "opencl"; else -> b }
            backendGroup.check(when (stored) { "opencl" -> 101; "vulkan" -> 102; else -> 100 })
            backendGroup.setOnCheckedChangeListener { _, id ->
                prefs.edit().putString("backend",
                    when (id) { 101 -> "opencl"; 102 -> "vulkan"; else -> "cpu" }).apply()
                refreshVoiceLabel()
            }
            // recommend the backend that fits this phone's GPU: Adreno ->
            // OpenCL (its Vulkan driver can't compile the shaders); any other
            // Vulkan GPU (Xclipse/Mali/RDNA) -> Vulkan; no GPU -> CPU
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
            addView(buttonRow(
                "Status" to {
                    thread {
                        val s = buildString {
                            val am = getSystemService(android.app.ActivityManager::class.java)
                            val mi = android.app.ActivityManager.MemoryInfo()
                            am.getMemoryInfo(mi)
                            appendLine("v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})")
                            appendLine("RAM ${mi.availMem / 1048576}/${mi.totalMem / 1048576} MB free")
                            append(runCatching { TtsEngine.nDeviceInfo() }.getOrElse { "device query failed: $it" })
                        }
                        runOnUiThread { statusText.text = s }
                    }
                },
                "Copy log" to {
                    thread {
                        val report = DebugLog.buildReport(this@MainActivity)
                        runOnUiThread {
                            val cm = getSystemService(android.content.ClipboardManager::class.java)
                            cm.setPrimaryClip(android.content.ClipData.newPlainText("ttsrunner-debug", report))
                            toast("Debug report copied (${report.length / 1024} KB)")
                        }
                    }
                },
                "Clear log" to { DebugLog.clear(this@MainActivity); toast("Log cleared") },
            ))
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
        private const val TAB_LISTEN = 1
        private const val TAB_VOICES = 2
        private const val TAB_JOBS = 3
        private const val TAB_SETTINGS = 4
    }
}
