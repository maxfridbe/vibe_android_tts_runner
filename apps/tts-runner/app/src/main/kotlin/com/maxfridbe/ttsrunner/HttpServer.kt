package com.maxfridbe.ttsrunner

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedOutputStream
import java.io.File
import java.io.InputStream
import java.net.InetSocketAddress
import java.net.ServerSocket
import java.net.Socket
import java.util.concurrent.Executors

/** A very small HTTP/1.1 server exposing the phone's voices as an
 *  OpenAI-compatible speech API, plus a browser client for them.
 *
 *  Hand-rolled rather than pulled in: the whole surface is four routes and a
 *  static file, and a dependency here would mean a build that needs the
 *  network. Connections are handled one request each and closed, which is
 *  plenty for a phone serving a handful of clients on a home network.
 *
 *  Several callers can be in flight at once; the engine service queues their
 *  work and serves it in order, so a second request waits rather than cutting
 *  the first one off.
 *
 *  There is no authentication. It binds to every interface because the point
 *  is reaching it from a laptop, and the tab says so. */
class HttpServer(private val ctx: Context, val port: Int) {

    @Volatile private var socket: ServerSocket? = null
    private val pool = Executors.newFixedThreadPool(8)
    @Volatile var requests = 0; private set
    @Volatile var lastError: String? = null

    val running: Boolean get() = socket != null

    fun start() {
        val s = ServerSocket()
        s.reuseAddress = true
        s.bind(InetSocketAddress(port))
        socket = s
        Thread({
            while (true) {
                val client = try { s.accept() } catch (_: Exception) { break }
                pool.execute { handleSafely(client) }
            }
        }, "http-accept").start()
    }

    fun stop() {
        runCatching { socket?.close() }
        socket = null
        pool.shutdownNow()
    }

    private fun handleSafely(client: Socket) {
        try {
            client.soTimeout = 30_000
            handle(client)
        } catch (t: Throwable) {
            lastError = "${t.javaClass.simpleName}: ${t.message}"
            DebugLog.log(ctx, "HttpServer", "request failed", t as? Exception ?: Exception(t))
        } finally {
            runCatching { client.close() }
        }
    }

    // ---- request parsing ---------------------------------------------------

    private class Request(val method: String, val path: String, val headers: Map<String, String>, val body: ByteArray)

    private fun readRequest(input: InputStream): Request? {
        val line = StringBuilder()
        val head = StringBuilder()
        var prev = -1
        // read up to the blank line that ends the header block
        while (true) {
            val c = input.read()
            if (c == -1) return null
            head.append(c.toChar())
            if (c == '\n'.code && prev == '\n'.code) break
            if (c != '\r'.code) prev = c
            if (head.length > 32_768) return null
        }
        val lines = head.toString().split("\r\n", "\n").filter { it.isNotBlank() }
        if (lines.isEmpty()) return null
        val parts = lines[0].split(" ")
        if (parts.size < 2) return null
        val headers = lines.drop(1).mapNotNull {
            val i = it.indexOf(':')
            if (i <= 0) null else it.substring(0, i).trim().lowercase() to it.substring(i + 1).trim()
        }.toMap()
        val len = headers["content-length"]?.toIntOrNull() ?: 0
        val body = ByteArray(len)
        var read = 0
        while (read < len) {
            val n = input.read(body, read, len - read)
            if (n <= 0) break
            read += n
        }
        line.append("")
        return Request(parts[0].uppercase(), parts[1], headers, body)
    }

    private fun handle(client: Socket) {
        val req = readRequest(client.getInputStream()) ?: return
        requests++
        val out = BufferedOutputStream(client.getOutputStream())
        val path = req.path.substringBefore('?')
        val host = req.headers["host"] ?: "localhost:$port"

        if (req.method == "OPTIONS") { respond(out, 204, "text/plain", ByteArray(0)); return }

        when {
            path == "/" || path == "/index.html" ->
                respond(out, 200, "text/html; charset=utf-8", asset("webui.html"))

            path == "/openapi.yaml" ->
                respond(out, 200, "application/yaml; charset=utf-8",
                    asset("openapi.yaml").decodeToString()
                        .replace("{{SERVER}}", "http://$host").toByteArray(),
                    extra = mapOf("content-disposition" to "inline; filename=\"openapi.yaml\""))

            path == "/health" ->
                respond(out, 200, "application/json", JSONObject()
                    .put("ok", true)
                    .put("model", ModelManager.selectedModel(ctx)?.id ?: "")
                    .put("requests", requests)
                    .toString().toByteArray())

            path == "/v1/models" -> {
                val data = JSONArray()
                ModelManager.CATALOG.filter { ModelManager.isDownloaded(ctx, it) }.forEach {
                    data.put(JSONObject().put("id", it.id).put("object", "model")
                        .put("owned_by", "tts-runner"))
                }
                respond(out, 200, "application/json",
                    JSONObject().put("object", "list").put("data", data).toString().toByteArray())
            }

            path == "/v1/audio/voices" -> {
                val supertonic = ModelManager.selectedModel(ctx)?.engine == "supertonic"
                val data = JSONArray()
                val voices = if (supertonic) VoiceStore.styleList(ctx) else VoiceStore.list(ctx)
                voices.forEach {
                    data.put(JSONObject().put("id", it.name).put("name", it.name)
                        .put("kind", if (supertonic) "style" else "reference"))
                }
                respond(out, 200, "application/json",
                    JSONObject().put("object", "list").put("data", data).toString().toByteArray())
            }

            path == "/v1/audio/speech" && req.method == "POST" -> speech(out, req)

            else -> respond(out, 404, "application/json",
                JSONObject().put("error", "no route for $path").toString().toByteArray())
        }
    }

    private fun speech(out: BufferedOutputStream, req: Request) {
        val body = runCatching { JSONObject(req.body.decodeToString()) }.getOrNull()
            ?: return error(out, 400, "body must be JSON")
        val text = body.optString("input").trim()
        if (text.isEmpty()) return error(out, 400, "input is empty")
        if (text.length > 8000) return error(out, 400, "input is longer than 8000 characters")

        val supertonic = ModelManager.selectedModel(ctx)?.engine == "supertonic"
        val known = (if (supertonic) VoiceStore.styleList(ctx) else VoiceStore.list(ctx)).map { it.name }
        val wanted = body.optString("voice").trim()
        val voice = known.firstOrNull { it.equals(wanted, true) }
            ?: VoiceStore.defaultFor(ctx, supertonic)?.name
            ?: return error(out, 400, "no voices on this phone")
        if (wanted.isNotEmpty() && !known.any { it.equals(wanted, true) }) {
            return error(out, 400, "no voice named \"$wanted\"; see /v1/audio/voices")
        }

        val format = body.optString("response_format").ifBlank { "wav" }.lowercase()
        if (format in setOf("opus", "flac")) {
            return error(out, 400, "$format is not encodable on Android; use wav, pcm, aac or mp3")
        }

        val result = SynthBridge.synth(ctx, text, voice)
        val wav = result.wav ?: return error(out, 502, result.error ?: "generation failed")
        val clip = Wav.read(wav) ?: return error(out, 502, "unreadable output")

        when (format) {
            "wav" -> respond(out, 200, "audio/wav", wav.readBytes(),
                extra = mapOf("x-sample-rate" to clip.rate.toString()))
            "pcm" -> respond(out, 200, "audio/L16", clip.pcm,
                extra = mapOf("x-sample-rate" to clip.rate.toString()))
            // Android ships no MP3 encoder, so an mp3 request is answered with
            // AAC in MP4 — labelled honestly, and noted in the spec.
            "mp3", "aac", "m4a" -> {
                val m4a = File(ctx.cacheDir, "http/out.m4a")
                m4a.parentFile?.mkdirs()
                m4a.delete()
                val pfd = android.os.ParcelFileDescriptor.open(m4a,
                    android.os.ParcelFileDescriptor.MODE_CREATE or
                        android.os.ParcelFileDescriptor.MODE_READ_WRITE)
                val enc = AacM4aWriter(pfd, clip.rate)
                enc.write(clip.pcm)
                enc.finish()
                pfd.close()
                respond(out, 200, "audio/mp4", m4a.readBytes(), extra = mapOf(
                    "x-sample-rate" to clip.rate.toString(),
                    "x-format" to if (format == "mp3") "aac-in-mp4 (no MP3 encoder on Android)" else "aac-in-mp4"))
            }
            else -> error(out, 400, "response_format must be wav, pcm, aac or mp3")
        }
    }

    private fun error(out: BufferedOutputStream, code: Int, msg: String) =
        respond(out, code, "application/json", JSONObject().put("error", msg).toString().toByteArray())

    private fun respond(out: BufferedOutputStream, code: Int, type: String, body: ByteArray,
                        extra: Map<String, String> = emptyMap()) {
        val reason = when (code) {
            200 -> "OK"; 204 -> "No Content"; 400 -> "Bad Request"
            404 -> "Not Found"; 502 -> "Bad Gateway"; else -> "Error"
        }
        val head = StringBuilder("HTTP/1.1 $code $reason\r\n")
        head.append("content-type: $type\r\n")
        head.append("content-length: ${body.size}\r\n")
        // the browser client is served from this same origin, but a client on
        // a laptop page is not — allow it rather than half-work
        head.append("access-control-allow-origin: *\r\n")
        head.append("access-control-allow-headers: *\r\n")
        head.append("access-control-allow-methods: GET, POST, OPTIONS\r\n")
        extra.forEach { (k, v) -> head.append("$k: $v\r\n") }
        head.append("connection: close\r\n\r\n")
        out.write(head.toString().toByteArray())
        out.write(body)
        out.flush()
    }

    private fun asset(name: String): ByteArray =
        ctx.assets.open(name).use { it.readBytes() }

    companion object {
        /** The address a laptop on the same network should use. */
        fun lanAddress(): String? = try {
            java.net.NetworkInterface.getNetworkInterfaces().toList()
                .filter { it.isUp && !it.isLoopback }
                .flatMap { it.inetAddresses.toList() }
                .firstOrNull { it is java.net.Inet4Address && !it.isLoopbackAddress }
                ?.hostAddress
        } catch (_: Exception) { null }
    }
}
