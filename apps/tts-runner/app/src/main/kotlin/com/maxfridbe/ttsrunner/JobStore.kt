package com.maxfridbe.ttsrunner

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Persistent job history (files/jobs.json), same idea as the audiobook
 *  maker's Jobs tab: what ran, with what voice/model, how it went, and the
 *  full input text so any job can be re-run — including with a different
 *  voice. Written from the :engine process, read by the UI. */
object JobStore {

    data class Job(
        val id: Long,               // start timestamp, doubles as identity
        var title: String,
        var text: String,
        var voice: String,
        var model: String,
        var backend: String,
        var save: Boolean,
        var status: String,         // running | done | stopped | failed
        var chunks: Int = 0,
        var audioSecs: Double = 0.0,
        var genMs: Long = 0,
        var output: String = "",    // "Music/TTS Runner/x.m4a" for save jobs
        var error: String = "",
    )

    private const val MAX_JOBS = 40
    private val lock = Any()

    private fun file(ctx: Context) = File(ctx.filesDir, "jobs.json")

    fun list(ctx: Context): List<Job> = synchronized(lock) { load(ctx) }

    fun add(ctx: Context, job: Job) = synchronized(lock) {
        val jobs = load(ctx)
        jobs.removeAll { it.id == job.id }
        jobs.add(0, job)
        while (jobs.size > MAX_JOBS) jobs.removeAt(jobs.size - 1)
        store(ctx, jobs)
    }

    fun update(ctx: Context, id: Long, mutate: (Job) -> Unit) = synchronized(lock) {
        val jobs = load(ctx)
        jobs.find { it.id == id }?.let { mutate(it); store(ctx, jobs) }
    }

    fun delete(ctx: Context, id: Long) = synchronized(lock) {
        val jobs = load(ctx)
        jobs.removeAll { it.id == id }
        store(ctx, jobs)
    }

    private fun load(ctx: Context): MutableList<Job> = try {
        val arr = JSONArray(file(ctx).readText())
        MutableList(arr.length()) { i ->
            val o = arr.getJSONObject(i)
            Job(
                id = o.getLong("id"),
                title = o.optString("title"),
                text = o.optString("text"),
                voice = o.optString("voice"),
                model = o.optString("model"),
                backend = o.optString("backend"),
                save = o.optBoolean("save"),
                status = o.optString("status", "done"),
                chunks = o.optInt("chunks"),
                audioSecs = o.optDouble("audioSecs", 0.0),
                genMs = o.optLong("genMs"),
                output = o.optString("output"),
                error = o.optString("error"),
            )
        }
    } catch (_: Exception) {
        mutableListOf()
    }

    private fun store(ctx: Context, jobs: List<Job>) {
        try {
            val arr = JSONArray()
            for (j in jobs) {
                arr.put(JSONObject()
                    .put("id", j.id).put("title", j.title)
                    .put("text", j.text.take(100_000))
                    .put("voice", j.voice).put("model", j.model)
                    .put("backend", j.backend).put("save", j.save)
                    .put("status", j.status).put("chunks", j.chunks)
                    .put("audioSecs", j.audioSecs).put("genMs", j.genMs)
                    .put("output", j.output).put("error", j.error))
            }
            file(ctx).writeText(arr.toString())
        } catch (e: Exception) {
            DebugLog.log(ctx, "JobStore", "store failed: $e")
        }
    }
}
