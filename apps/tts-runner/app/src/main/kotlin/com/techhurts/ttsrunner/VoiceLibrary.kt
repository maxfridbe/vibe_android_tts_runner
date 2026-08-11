package com.techhurts.ttsrunner

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.net.URL

/** The bundled example voices, fetched from this repo's `/voices` folder — a
 *  set of generated Supertonic speakers (no reference recordings, no real
 *  people). Same raw-file approach as the cloning models: a manifest lists each
 *  voice, and the styles download on demand, carrying their name and
 *  description. */
object VoiceLibrary {
    private const val BASE =
        "https://raw.githubusercontent.com/maxfridbe/vibe_android_tts_runner/main/voices/"

    data class Entry(
        val name: String, val file: String,
        val gender: String, val british: Boolean, val description: String,
    )

    /** Reads the manifest. Network — call off the UI thread. */
    fun manifest(): List<Entry> {
        val arr = JSONObject(URL(BASE + "library.json").readText()).getJSONArray("voices")
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            Entry(o.getString("name"), o.getString("file"),
                o.optString("gender"), o.optBoolean("british"), o.optString("description"))
        }
    }

    /** Downloads one voice's style JSON to the cache and returns the file. */
    fun download(ctx: Context, e: Entry): File {
        val f = File(ctx.cacheDir, e.file)
        URL(BASE + e.file).openStream().use { input -> f.outputStream().use { input.copyTo(it) } }
        return f
    }
}
