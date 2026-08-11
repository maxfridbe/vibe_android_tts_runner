package com.techhurts.ttsrunner

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Mono 16-bit WAV read/write. Everything the app produces is in this format,
 *  which is what makes a conversation joinable into one track: same rate, same
 *  width, so the parts concatenate byte-for-byte. */
object Wav {

    class Clip(val rate: Int, val pcm: ByteArray) {
        val seconds: Double get() = pcm.size / 2.0 / rate
    }

    fun read(f: File): Clip? = try {
        val b = f.readBytes()
        if (b.size < 44 || String(b, 0, 4) != "RIFF") null else {
            val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
            var pos = 12
            var rate = 0
            var clip: Clip? = null
            while (pos + 8 <= b.size) {
                val id = String(b, pos, 4)
                val len = bb.getInt(pos + 4)
                if (id == "fmt ") rate = bb.getInt(pos + 12)
                if (id == "data") {
                    val end = minOf(pos + 8 + len, b.size)
                    clip = Clip(rate, b.copyOfRange(pos + 8, end))
                    break
                }
                pos += 8 + len + (len and 1)
            }
            clip
        }
    } catch (_: Exception) { null }

    /** Duration without loading the audio — for a list of clips this is the
     *  difference between instant and a visible stall. */
    fun seconds(f: File): Double = try {
        f.inputStream().use { input ->
            val head = ByteArray(44)
            if (input.read(head) < 44) 0.0 else {
                val bb = ByteBuffer.wrap(head).order(ByteOrder.LITTLE_ENDIAN)
                val rate = bb.getInt(24)
                if (rate <= 0) 0.0 else (f.length() - 44).coerceAtLeast(0) / 2.0 / rate
            }
        }
    } catch (_: Exception) { 0.0 }

    fun write(dest: File, rate: Int, pcm: ByteArray) {
        dest.outputStream().use { out ->
            val h = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            h.put("RIFF".toByteArray()); h.putInt(36 + pcm.size)
            h.put("WAVE".toByteArray()); h.put("fmt ".toByteArray())
            h.putInt(16); h.putShort(1); h.putShort(1)
            h.putInt(rate); h.putInt(rate * 2)
            h.putShort(2); h.putShort(16)
            h.put("data".toByteArray()); h.putInt(pcm.size)
            out.write(h.array())
            out.write(pcm)
        }
    }

    /** Joins clips into one file, with [gapMs] of silence between them so a
     *  conversation does not run together into one breathless sentence. When
     *  [trim] is set, each clip's leading and trailing near-silence (the padding
     *  the TTS engines bake onto every utterance) is removed first, so the only
     *  pause between speakers is the one [gapMs] asks for. Returns null if
     *  nothing was readable. */
    fun join(dest: File, sources: List<File>, gapMs: Int = 350, trim: Boolean = false): File? {
        val clips = sources.mapNotNull { read(it) }.filter { it.pcm.isNotEmpty() }
        if (clips.isEmpty()) return null
        val rate = clips.first().rate
        val gap = ByteArray(2 * rate * gapMs / 1000)
        val out = java.io.ByteArrayOutputStream()
        for ((i, c) in clips.withIndex()) {
            if (i > 0) out.write(gap)
            out.write(if (trim) trimSilence(c.pcm, rate) else c.pcm)
        }
        write(dest, rate, out.toByteArray())
        return dest
    }

    /** Drop leading/trailing near-silence from a 16-bit mono PCM buffer, keeping
     *  a short margin so consonants and natural decay are not clipped. The
     *  threshold is a fraction of full scale; speech sits well above it and the
     *  engines' padding well below. */
    fun trimSilence(pcm: ByteArray, rate: Int,
                    thresholdFrac: Double = 0.02, marginMs: Int = 40): ByteArray {
        val n = pcm.size / 2
        if (n == 0) return pcm
        val thr = (thresholdFrac * Short.MAX_VALUE).toInt()
        fun amp(i: Int): Int {
            val lo = pcm[2 * i].toInt() and 0xff
            val hi = pcm[2 * i + 1].toInt()
            return kotlin.math.abs((hi shl 8) or lo)
        }
        var first = 0
        while (first < n && amp(first) < thr) first++
        if (first == n) return ByteArray(0)          // all silence
        var last = n - 1
        while (last > first && amp(last) < thr) last--
        val margin = rate * marginMs / 1000
        val a = (first - margin).coerceAtLeast(0)
        val b = (last + margin).coerceAtMost(n - 1)
        return pcm.copyOfRange(2 * a, 2 * (b + 1))
    }

    /** "1m 43s" / "12s" */
    fun fmt(seconds: Double): String {
        val s = seconds.toInt()
        return if (s >= 60) "${s / 60}m ${s % 60}s" else "${s}s"
    }
}
