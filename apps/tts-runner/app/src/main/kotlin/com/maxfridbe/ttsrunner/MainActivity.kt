package com.maxfridbe.ttsrunner

import android.app.Activity
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import kotlin.concurrent.thread

/** Setup + test screen: download a model, pick backend, manage voices, try a
 *  sentence. The real entry point day-to-day is the share sheet
 *  (ShareActivity). */
class MainActivity : Activity() {

    private lateinit var modelStatus: TextView
    private lateinit var modelBtn: Button
    private lateinit var modelProgress: ProgressBar
    private lateinit var modelGroup: RadioGroup
    private lateinit var voicesGroup: RadioGroup
    private lateinit var runStatus: TextView
    private lateinit var runProgress: ProgressBar
    private lateinit var waveform: WaveformView
    private lateinit var playBtn: Button
    private var player: android.media.MediaPlayer? = null
    private var downloading = false
    private val ui = android.os.Handler(android.os.Looper.getMainLooper())

    private val engineConn = object : android.content.ServiceConnection {
        override fun onServiceConnected(n: android.content.ComponentName?, b: android.os.IBinder?) {}
        override fun onServiceDisconnected(n: android.content.ComponentName?) {}
    }

    private val statusReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            intent ?: return
            val state = intent.getStringExtra("state")
            val chunk = intent.getIntExtra("chunk", 0)
            val total = intent.getIntExtra("total", 0)
            val message = intent.getStringExtra("message") ?: ""
            val permille = intent.getIntExtra("permille", -1)
            if (permille >= 0) runProgress.progress = permille
            runStatus.text = when (state) {
                "loading" -> { runProgress.progress = 0; "Loading model…" }
                "generating" -> {
                    if (total > 0) runProgress.progress = chunk * 1000 / total
                    "Generating chunk ${chunk + 1}/$total: “$message…”"
                }
                "frames" ->
                    if (chunk == 0) "Processing text & voice…"
                    else "Generating… ${"%.1f".format(chunk / 12.5)} s of audio"
                "done" -> { runProgress.progress = 1000; refreshPlayer(); "Done ($total chunks)" }
                "saved" -> { runProgress.progress = 1000; refreshPlayer(); "Saved to $message" }
                "stopped" -> { runProgress.progress = 0; "Stopped" }
                "note" -> message
                "error" -> { runProgress.progress = 0; "Error: $message" }
                else -> state ?: ""
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }

        fun header(t: String) = root.addView(TextView(this).apply {
            text = t; textSize = 17f; setTypeface(typeface, Typeface.BOLD)
            setPadding(0, pad, 0, pad / 4)
        })

        root.addView(TextView(this).apply {
            text = "Share text or a link to TTS Runner from any app, pick a voice, listen."
        })
        root.addView(TextView(this).apply {
            text = "v${BuildConfig.VERSION_NAME} (${BuildConfig.VERSION_CODE})"
            textSize = 12f
        })
        if (android.os.Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) !=
                android.content.pm.PackageManager.PERMISSION_GRANTED) {
            // without this the foreground-service notification (the only
            // progress surface outside the app) is silently suppressed
            requestPermissions(arrayOf(android.Manifest.permission.POST_NOTIFICATIONS), 12)
        }

        header("1 · Model")
        modelGroup = RadioGroup(this)
        for ((i, m) in ModelManager.CATALOG.withIndex()) {
            modelGroup.addView(RadioButton(this).apply { text = m.label; id = i })
        }
        root.addView(modelGroup)
        modelStatus = TextView(this)
        root.addView(modelStatus)
        modelProgress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal)
        root.addView(modelProgress)
        modelBtn = Button(this).apply { setOnClickListener { onModelButton() } }
        root.addView(modelBtn)
        modelGroup.setOnCheckedChangeListener { _, id ->
            ModelManager.CATALOG.getOrNull(id)?.let { ModelManager.selectModel(this, it.id) }
            refreshModelUi()
        }

        header("2 · Engine")
        val prefs = getSharedPreferences("ttsrunner", MODE_PRIVATE)
        val backendGroup = RadioGroup(this).apply { orientation = LinearLayout.HORIZONTAL }
        backendGroup.addView(RadioButton(this).apply { text = "CPU"; id = 100 })
        backendGroup.addView(RadioButton(this).apply { text = "GPU"; id = 101 })
        root.addView(backendGroup)
        root.addView(TextView(this).apply {
            textSize = 12f
            text = "GPU needs the Q4_0 model (the Adreno kernels are tuned for it; ~1.7x faster " +
                "speech generation). Other models always run on CPU."
        })
        backendGroup.check(if (prefs.getString("backend", "cpu") == "gpu") 101 else 100)
        backendGroup.setOnCheckedChangeListener { _, id ->
            prefs.edit().putString("backend", if (id == 101) "gpu" else "cpu").apply()
        }
        val pm = getSystemService(android.os.PowerManager::class.java)
        if (!pm.isIgnoringBatteryOptimizations(packageName)) {
            root.addView(Button(this).apply {
                text = "Allow background generation (battery)"
                setOnClickListener {
                    // Samsung throttles screen-off background CPU hard without
                    // this; a 30 s generation once took 24 minutes
                    startActivity(Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS,
                        Uri.parse("package:$packageName")))
                }
            })
        }

        header("3 · Voices")
        root.addView(TextView(this).apply {
            text = "A voice is 10–20 s of clean speech (wav/mp3/flac). The selected voice is the share-sheet default."
        })
        voicesGroup = RadioGroup(this)
        root.addView(voicesGroup)
        val voiceButtons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        voiceButtons.addView(Button(this).apply {
            text = "Import voice"
            setOnClickListener {
                startActivityForResult(
                    Intent(Intent.ACTION_OPEN_DOCUMENT)
                        .addCategory(Intent.CATEGORY_OPENABLE)
                        .setType("audio/*"), REQ_IMPORT)
            }
        })
        voiceButtons.addView(Button(this).apply {
            text = "Delete selected"
            setOnClickListener {
                selectedVoice()?.let { VoiceStore.delete(it); refreshVoices() }
            }
        })
        root.addView(voiceButtons)

        header("4 · Try it")
        val tryText = EditText(this).apply {
            setText("The quick brown fox jumps over the lazy dog, then reads the entire internet aloud.")
            minLines = 2
        }
        root.addView(tryText)
        fun startJob(save: Boolean) {
            val v = selectedVoice()
            if (v == null) { toast("Import a voice first"); return }
            if (ModelManager.selectedModel(this) == null) { toast("Download a model first"); return }
            VoiceStore.setDefault(this, v)
            startForegroundService(
                Intent(this, TtsService::class.java)
                    .setAction(TtsService.ACTION_SPEAK)
                    .putExtra(TtsService.EXTRA_TEXT, tryText.text.toString())
                    .putExtra(TtsService.EXTRA_TITLE, if (save) "TTS ${tryText.text.take(24)}" else "Test")
                    .putExtra(TtsService.EXTRA_VOICE, v.name)
                    .putExtra(TtsService.EXTRA_BACKEND, prefs.getString("backend", "cpu"))
                    .putExtra(TtsService.EXTRA_SAVE, save))
        }
        val runButtons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        runButtons.addView(Button(this).apply {
            text = "Speak"
            setOnClickListener { startJob(false) }
        })
        runButtons.addView(Button(this).apply {
            text = "Save audio"
            setOnClickListener { startJob(true) }
        })
        runButtons.addView(Button(this).apply {
            text = "Stop"
            setOnClickListener {
                startService(Intent(this@MainActivity, TtsService::class.java).setAction(TtsService.ACTION_STOP))
            }
        })
        root.addView(runButtons)
        runStatus = TextView(this)
        root.addView(runStatus)
        runProgress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 1000 }
        root.addView(runProgress)

        // last generated audio: play/pause + tappable waveform
        val playerRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        playBtn = Button(this).apply {
            text = "▶"
            setOnClickListener { togglePlayback() }
        }
        playerRow.addView(playBtn,
            LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT))
        waveform = WaveformView(this).apply {
            onSeek = { frac ->
                player?.let { it.seekTo((it.duration * frac).toInt()) }
            }
        }
        playerRow.addView(waveform,
            LinearLayout.LayoutParams(0, (48 * resources.displayMetrics.density).toInt(), 1f))
        root.addView(playerRow)

        header("5 · Debug")
        val statusText = TextView(this).apply { textSize = 12f; setTypeface(android.graphics.Typeface.MONOSPACE) }
        root.addView(statusText)
        val debugButtons = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        debugButtons.addView(Button(this).apply {
            text = "Refresh status"
            setOnClickListener {
                thread {
                    val s = buildString {
                        val am = getSystemService(android.app.ActivityManager::class.java)
                        val mi = android.app.ActivityManager.MemoryInfo()
                        am.getMemoryInfo(mi)
                        appendLine("${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}, android ${android.os.Build.VERSION.RELEASE}")
                        appendLine("RAM ${mi.availMem / 1048576}/${mi.totalMem / 1048576} MB free, low=${mi.lowMemory}")
                        append(runCatching { TtsEngine.nDeviceInfo() }.getOrElse { "device query failed: $it" })
                    }
                    runOnUiThread { statusText.text = s }
                }
            }
        })
        debugButtons.addView(Button(this).apply {
            text = "Copy debug log"
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
        })
        debugButtons.addView(Button(this).apply {
            text = "Clear log"
            setOnClickListener { DebugLog.clear(this@MainActivity); toast("Log cleared") }
        })
        root.addView(debugButtons)

        setContentView(ScrollView(this).apply { addView(root) })
        refreshModelUi()
        refreshVoices()
        refreshPlayer()
    }

    /** Keep the :engine process (and its loaded model) alive while the app is
     *  on screen. */
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
        refreshPlayer()
    }

    override fun onPause() {
        super.onPause()
        unregisterReceiver(statusReceiver)
    }

    override fun onDestroy() {
        player?.release(); player = null
        super.onDestroy()
    }

    // ---- last-audio player -------------------------------------------------

    private fun lastAudio() = java.io.File(filesDir, "last_audio.wav")

    private fun refreshPlayer() {
        val f = lastAudio()
        val visible = f.exists() && f.length() > 44
        playBtn.isEnabled = visible
        if (visible) thread {
            waveform.loadWav(f)
            runOnUiThread { waveform.setProgress(0f) }
        }
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
        if (p != null && p.isPlaying) {
            p.pause()
            playBtn.text = "▶"
            return
        }
        if (p != null) {
            p.start()
            playBtn.text = "❚❚"
            ui.post(progressTicker)
            return
        }
        try {
            player = android.media.MediaPlayer().apply {
                setDataSource(lastAudio().absolutePath)
                prepare()
                setOnCompletionListener {
                    playBtn.text = "▶"
                    waveform.setProgress(1f)
                    it.release(); player = null
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

    private fun currentModel(): ModelManager.CatalogModel =
        ModelManager.CATALOG.getOrNull(modelGroup.checkedRadioButtonId) ?: ModelManager.CATALOG[0]

    private fun refreshModelUi() {
        if (modelGroup.checkedRadioButtonId == -1) {
            val sel = ModelManager.selectedModel(this)
            modelGroup.check(ModelManager.CATALOG.indexOfFirst { it.id == sel?.id }.takeIf { it >= 0 } ?: 0)
            return // listener re-enters
        }
        val m = currentModel()
        val have = ModelManager.isDownloaded(this, m)
        modelStatus.text = if (have) "Downloaded ✓" else "Not downloaded (${m.totalBytes / 1_000_000} MB)"
        modelProgress.progress = if (have) 100 else 0
        modelBtn.text = if (downloading) "Cancel download" else if (have) "Re-download" else "Download"
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
                        modelStatus.text = "$file — ${doneBytes / 1_000_000} / ${totalBytes / 1_000_000} MB"
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

    private fun selectedVoice(): VoiceStore.Voice? =
        VoiceStore.list(this).getOrNull(voicesGroup.checkedRadioButtonId)

    private fun refreshVoices() {
        voicesGroup.removeAllViews()
        val def = VoiceStore.defaultVoice(this)
        val list = VoiceStore.list(this)
        for ((i, v) in list.withIndex()) {
            voicesGroup.addView(RadioButton(this).apply {
                text = "${v.name}  (${v.file.length() / 1024} KB)"
                id = i
                isChecked = v.name == def?.name
            })
        }
        if (list.isNotEmpty() && voicesGroup.checkedRadioButtonId == -1) voicesGroup.check(0)
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
                refreshVoices()
            } catch (e: Exception) {
                toast("Import failed: ${e.message}")
            }
        }
    }

    private fun toast(t: String) = Toast.makeText(this, t, Toast.LENGTH_LONG).show()

    companion object { private const val REQ_IMPORT = 11 }
}
