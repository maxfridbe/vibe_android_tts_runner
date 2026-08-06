package com.maxfridbe.ttsrunner

import android.content.Context
import android.net.Uri
import java.io.File

/** Voice library: each voice is one reference audio file (wav/mp3/flac,
 *  ideally 10-20 s of clean speech) in filesDir/voices. Cloning is
 *  speaker-embedding based, so no transcript is needed. */
object VoiceStore {

    data class Voice(val name: String, val file: File)

    private fun dir(ctx: Context): File = File(ctx.filesDir, "voices").apply { mkdirs() }

    fun list(ctx: Context): List<Voice> =
        dir(ctx).listFiles { f -> f.isFile && f.extension.lowercase() in setOf("wav", "mp3", "flac", "ogg", "m4a") }
            ?.sortedBy { it.name.lowercase() }
            ?.map { Voice(it.nameWithoutExtension, it) }
            ?: emptyList()

    fun import(ctx: Context, uri: Uri, displayName: String): Voice {
        val safe = displayName.replace(Regex("[^A-Za-z0-9 ._-]"), "_").ifBlank { "voice" }
        val ext = safe.substringAfterLast('.', "").lowercase().ifBlank { "wav" }
        val base = safe.substringBeforeLast('.').trim().ifBlank { "voice" }
        var dest = File(dir(ctx), "$base.$ext")
        var i = 2
        while (dest.exists()) { dest = File(dir(ctx), "$base ($i).$ext"); i++ }
        ctx.contentResolver.openInputStream(uri).use { input ->
            requireNotNull(input) { "cannot open $uri" }
            dest.outputStream().use { input.copyTo(it) }
        }
        return Voice(dest.nameWithoutExtension, dest)
    }

    fun delete(v: Voice) { v.file.delete() }

    fun defaultVoice(ctx: Context): Voice? {
        val prefs = ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
        val name = prefs.getString("default_voice", null)
        val all = list(ctx)
        return all.find { it.name == name } ?: all.firstOrNull()
    }

    fun setDefault(ctx: Context, v: Voice) {
        ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE).edit().putString("default_voice", v.name).apply()
    }
}
