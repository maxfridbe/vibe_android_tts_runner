package com.techhurts.ttsrunner

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Saved conversations.
 *
 *  A chat is a folder: one JSON file describing the lines in order, and the
 *  generated clip for each of them beside it. Keeping the audio with the chat
 *  is what makes reopening one instant — the lines are already spoken, so it
 *  replays and exports without touching the engine. */
object ChatStore {

    class Line(
        val text: String,
        val voice: String,
        val fast: Boolean,
        var audio: File?,
        var secs: Double,
    )

    class Chat(val id: String, var name: String, val lines: MutableList<Line>) {
        val seconds: Double get() = lines.sumOf { it.secs }
    }

    /** What the list needs, without reading every clip. */
    class Meta(val id: String, val name: String, val lines: Int, val seconds: Double, val updated: Long)

    private fun root(ctx: Context) = File(ctx.filesDir, "chats").apply { mkdirs() }

    fun dir(ctx: Context, id: String) = File(root(ctx), id).apply { mkdirs() }

    private fun file(ctx: Context, id: String) = File(dir(ctx, id), "chat.json")

    fun list(ctx: Context): List<Meta> =
        root(ctx).listFiles { f -> f.isDirectory }
            ?.mapNotNull { d ->
                val f = File(d, "chat.json")
                if (!f.exists()) null else runCatching {
                    val o = JSONObject(f.readText())
                    val lines = o.optJSONArray("lines") ?: JSONArray()
                    var secs = 0.0
                    for (i in 0 until lines.length()) secs += lines.getJSONObject(i).optDouble("secs", 0.0)
                    Meta(d.name, o.optString("name", "Chat"), lines.length(), secs, f.lastModified())
                }.getOrNull()
            }
            ?.sortedByDescending { it.updated }
            ?: emptyList()

    fun create(ctx: Context, name: String): Chat {
        val id = "chat-" + System.currentTimeMillis()
        val chat = Chat(id, name.ifBlank { "New chat" }, mutableListOf())
        save(ctx, chat)
        return chat
    }

    fun load(ctx: Context, id: String): Chat? = runCatching {
        val o = JSONObject(file(ctx, id).readText())
        val arr = o.optJSONArray("lines") ?: JSONArray()
        val lines = (0 until arr.length()).map { i ->
            val l = arr.getJSONObject(i)
            val audio = l.optString("audio").takeIf { it.isNotBlank() }
                ?.let { File(dir(ctx, id), it) }?.takeIf { it.exists() }
            Line(l.optString("text"), l.optString("voice"), l.optBoolean("fast", true),
                audio, l.optDouble("secs", 0.0))
        }.toMutableList()
        Chat(id, o.optString("name", "Chat"), lines)
    }.getOrNull()

    fun save(ctx: Context, chat: Chat) {
        val arr = JSONArray()
        for (l in chat.lines) {
            arr.put(JSONObject()
                .put("text", l.text)
                .put("voice", l.voice)
                .put("fast", l.fast)
                // stored by name: the folder can move with the app's data
                .put("audio", l.audio?.name ?: "")
                .put("secs", l.secs))
        }
        runCatching {
            file(ctx, chat.id).writeText(
                JSONObject().put("name", chat.name).put("lines", arr).toString())
        }.onFailure { DebugLog.log(ctx, "ChatStore", "save ${chat.id} failed", it) }
    }

    fun rename(ctx: Context, id: String, name: String) {
        val c = load(ctx, id) ?: return
        c.name = name.trim().ifBlank { c.name }
        save(ctx, c)
    }

    /** Removes the chat and every clip it owns. */
    fun delete(ctx: Context, id: String) {
        dir(ctx, id).deleteRecursively()
    }

    fun defaultName(ctx: Context) = "Chat ${list(ctx).size + 1}"
}
