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

    /** Supertonic style voices live beside the reference audio as .json.
     *  They are a different kind of voice: a style vector, not a recording,
     *  so only the Supertonic engine can use them. */
    fun styleList(ctx: Context): List<Voice> =
        dir(ctx).listFiles { f -> f.isFile && f.extension.lowercase() == "json" }
            ?.sortedBy { it.name.lowercase() }
            ?.map { Voice(it.nameWithoutExtension, it) }
            ?: emptyList()

    fun styleFile(ctx: Context, name: String): File? =
        File(dir(ctx), "$name.json").takeIf { it.exists() }

    /** Copies a style JSON into the library (from the model download or a
     *  file the user picked). */
    fun importStyle(ctx: Context, src: File, name: String): Voice {
        val safe = name.replace(Regex("[^A-Za-z0-9 ._-]"), "_").trim().ifBlank { "style" }
        var dest = File(dir(ctx), "$safe.json")
        var i = 2
        while (dest.exists()) { dest = File(dir(ctx), "$safe ($i).json"); i++ }
        src.copyTo(dest)
        return Voice(dest.nameWithoutExtension, dest)
    }

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

    fun delete(ctx: Context, v: Voice) {
        v.file.delete()
        transcriptFile(ctx, v.name).delete()
        previewsDir(ctx).listFiles { f -> f.name.startsWith("${v.name}__") }?.forEach { it.delete() }
        prefs(ctx).edit().remove(iconKey(v.name)).apply()
    }

    /** Renames the reference audio and everything keyed to the old name:
     *  transcript, cached previews, icon, and the default-voice pointer. */
    fun rename(ctx: Context, v: Voice, newName: String): Voice {
        val safe = newName.replace(Regex("[^A-Za-z0-9 ._-]"), "_").trim()
        if (safe.isBlank() || safe == v.name) return v
        var dest = File(dir(ctx), "$safe.${v.file.extension}")
        var i = 2
        while (dest.exists()) { dest = File(dir(ctx), "$safe ($i).${v.file.extension}"); i++ }
        if (!v.file.renameTo(dest)) throw java.io.IOException("could not rename ${v.file.name}")
        val name = dest.nameWithoutExtension
        transcriptFile(ctx, v.name).takeIf { it.exists() }?.renameTo(transcriptFile(ctx, name))
        previewsDir(ctx).listFiles { f -> f.name.startsWith("${v.name}__") }?.forEach {
            it.renameTo(File(previewsDir(ctx), name + it.name.removePrefix(v.name)))
        }
        val p = prefs(ctx)
        val icon = p.getString(iconKey(v.name), null)
        p.edit().apply {
            remove(iconKey(v.name))
            if (icon != null) putString(iconKey(name), icon)
            if (p.getString("default_voice", null) == v.name) putString("default_voice", name)
        }.apply()
        return Voice(name, dest)
    }

    /** Emoji a voice is shown with. Unset voices get a stable one derived from
     *  the name, so the library is never a wall of identical rows. */
    val ICONS = listOf(
        "🎙️", "🦊", "🐻", "🦉", "🐧", "🐙", "🦁", "🐉",
        "🤖", "👽", "🧙", "🧛", "🤠", "👑", "🎩", "🦄",
        "🐳", "🦖", "🐺", "🌚", "🎭", "📻", "🧜", "🕵️",
    )

    private fun prefs(ctx: Context) = ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)
    private fun iconKey(name: String) = "voice_icon_$name"

    fun icon(ctx: Context, name: String): String =
        prefs(ctx).getString(iconKey(name), null)
            ?: ICONS[Math.floorMod(name.hashCode(), ICONS.size)]

    fun setIcon(ctx: Context, name: String, icon: String) {
        prefs(ctx).edit().putString(iconKey(name), icon).apply()
    }

    private fun previewsDir(ctx: Context) = File(ctx.filesDir, "previews").apply { mkdirs() }

    /** Cached generated sample for a voice. Previews are model-specific (the
     *  same reference sounds different per model/quant), so the cache key is
     *  (voice, model). */
    fun previewFile(ctx: Context, name: String, modelId: String): File =
        File(previewsDir(ctx), "${name}__$modelId.wav")

    /** Optional transcript of the reference audio, user-editable. Not used by
     *  the current x-vector cloning; reserved for ICL cloning (and a future
     *  on-device whisper auto-fill the user can validate). */
    fun transcriptFile(ctx: Context, name: String): File = File(dir(ctx), "$name.txt")

    /** Register generated audio (e.g. a designed-voice roll) as a new voice. */
    fun adopt(ctx: Context, wav: File, name: String): Voice {
        val safe = name.replace(Regex("[^A-Za-z0-9 ._-]"), "_").trim().ifBlank { "designed" }
        var dest = File(dir(ctx), "$safe.wav")
        var i = 2
        while (dest.exists()) { dest = File(dir(ctx), "$safe ($i).wav"); i++ }
        wav.copyTo(dest)
        return Voice(dest.nameWithoutExtension, dest)
    }

    const val PREVIEW_TEXT = "Hello! This is a short preview of my voice. The quick brown fox jumps over the lazy dog."
    const val DESIGN_TEXT = "Hello there! This is how I sound. I can read articles, stories, and anything else you like."

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
