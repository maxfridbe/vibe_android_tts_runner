package com.techhurts.ttsrunner

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.DocumentsContract

/** A folder outside the app that holds a copy of every speaker.
 *
 *  The library itself lives in filesDir, which Android deletes with the app —
 *  so a voice you recorded or a style someone sent you is gone on uninstall.
 *  Point this at a real folder (Documents, an SD card, a synced drive) and the
 *  app mirrors every speaker into it and imports back anything it finds there,
 *  which makes a reinstall a restore rather than a fresh start.
 *
 *  Generated audio is not part of this: saved jobs already go to
 *  Music/TTS Runner through MediaStore, which uninstalling does not touch. */
object SpeakerFolder {

    private const val KEY = "speaker_folder"

    private fun prefs(ctx: Context) = ctx.getSharedPreferences("ttsrunner", Context.MODE_PRIVATE)

    fun uri(ctx: Context): Uri? = prefs(ctx).getString(KEY, null)?.let(Uri::parse)

    /** Human-readable tail of the tree id, e.g. "Documents/Speakers". */
    fun label(ctx: Context): String? = uri(ctx)?.let {
        runCatching { DocumentsContract.getTreeDocumentId(it).substringAfter(':') }.getOrNull()
    }

    /** Keeps the grant across reboots and app restarts; without this the uri
     *  is only good for the current task. */
    fun remember(ctx: Context, tree: Uri) {
        runCatching {
            ctx.contentResolver.takePersistableUriPermission(tree,
                Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        }
        prefs(ctx).edit().putString(KEY, tree.toString()).apply()
    }

    fun forget(ctx: Context) = prefs(ctx).edit().remove(KEY).apply()

    private fun parentDoc(tree: Uri) =
        DocumentsContract.buildDocumentUriUsingTree(tree, DocumentsContract.getTreeDocumentId(tree))

    /** File names currently in the folder. */
    private fun listNames(ctx: Context, tree: Uri): Map<String, Uri> {
        val out = mutableMapOf<String, Uri>()
        val children = DocumentsContract.buildChildDocumentsUriUsingTree(
            tree, DocumentsContract.getTreeDocumentId(tree))
        ctx.contentResolver.query(children, arrayOf(
            DocumentsContract.Document.COLUMN_DOCUMENT_ID,
            DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null)?.use { c ->
            while (c.moveToNext()) {
                val name = c.getString(1) ?: continue
                out[name] = DocumentsContract.buildDocumentUriUsingTree(tree, c.getString(0))
            }
        }
        return out
    }

    /** Copies one speaker out, replacing an older copy of the same name. */
    fun mirror(ctx: Context, v: VoiceStore.Voice): Boolean {
        val tree = uri(ctx) ?: return false
        return try {
            val existing = listNames(ctx, tree)[v.file.name]
            val dest = existing ?: DocumentsContract.createDocument(
                ctx.contentResolver, parentDoc(tree),
                if (v.file.extension.equals("json", true)) "application/json" else "audio/*",
                v.file.name) ?: return false
            ctx.contentResolver.openOutputStream(dest, "wt")!!.use { out ->
                v.file.inputStream().use { it.copyTo(out) }
            }
            true
        } catch (e: Exception) {
            DebugLog.log(ctx, "SpeakerFolder", "mirror ${v.file.name} failed", e)
            false
        }
    }

    /** Drops a name from the folder. Deleting or renaming a speaker has to do
     *  this, or the next sync would faithfully restore what you just removed. */
    fun remove(ctx: Context, fileName: String) {
        val tree = uri(ctx) ?: return
        runCatching {
            listNames(ctx, tree)[fileName]?.let {
                DocumentsContract.deleteDocument(ctx.contentResolver, it)
            }
        }.onFailure { DebugLog.log(ctx, "SpeakerFolder", "remove $fileName failed", it) }
    }

    data class Result(val exported: Int, val imported: Int, val error: String? = null)

    /** Union in both directions: whatever the phone has is written out,
     *  whatever the folder has and the phone lacks is imported. That second
     *  half is the restore path after a reinstall. */
    fun sync(ctx: Context): Result {
        val tree = uri(ctx) ?: return Result(0, 0, "no folder set")
        return try {
            val inFolder = listNames(ctx, tree)
            val styles = VoiceStore.styleList(ctx)
            val refs = VoiceStore.list(ctx)
            var exported = 0
            for (v in styles + refs) if (v.file.name !in inFolder && mirror(ctx, v)) exported++

            // a style and a recording may share a name — they are different
            // speakers to different engines — so each kind is checked separately
            val haveStyles = styles.map { it.name }.toSet()
            val haveRefs = refs.map { it.name }.toSet()
            var imported = 0
            for ((name, doc) in inFolder) {
                if (!VoiceStore.isSpeakerFile(name)) continue
                val isStyle = name.endsWith(".json", true)
                if (name.substringBeforeLast('.') in (if (isStyle) haveStyles else haveRefs)) continue
                runCatching { VoiceStore.importAny(ctx, doc, name) }
                    .onSuccess { imported++ }
                    .onFailure { DebugLog.log(ctx, "SpeakerFolder", "import $name failed", it) }
            }
            DebugLog.log(ctx, "SpeakerFolder", "sync: $exported out, $imported in")
            Result(exported, imported)
        } catch (e: Exception) {
            DebugLog.log(ctx, "SpeakerFolder", "sync failed", e)
            Result(0, 0, e.message ?: e.javaClass.simpleName)
        }
    }
}
