package com.techhurts.ttsrunner

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Decode any picked audio (wav/mp3/m4a/flac/ogg) to mono 16-bit PCM.
 *
 *  The cloner and the waveform both want raw samples, but the file picker
 *  hands back whatever the user chose. MediaExtractor + MediaCodec turn all of
 *  it into one shape; stereo is downmixed, and the native sample rate is kept
 *  (the cloner resamples to whatever its analyzer needs). */
object AudioDecoder {

    class Pcm(val rate: Int, val samples: ShortArray) {
        val seconds: Double get() = if (rate > 0) samples.size.toDouble() / rate else 0.0
    }

    /** Blocking; call off the main thread. Returns null if nothing decodes. */
    fun decode(ctx: Context, uri: Uri): Pcm? {
        val ex = MediaExtractor()
        return try {
            ctx.contentResolver.openFileDescriptor(uri, "r")?.use { pfd ->
                ex.setDataSource(pfd.fileDescriptor)
                decodeExtractor(ex)
            }
        } catch (_: Exception) {
            null
        } finally {
            runCatching { ex.release() }
        }
    }

    private fun decodeExtractor(ex: MediaExtractor): Pcm? {
        var track = -1
        var format: MediaFormat? = null
        for (i in 0 until ex.trackCount) {
            val f = ex.getTrackFormat(i)
            if (f.getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true) {
                track = i; format = f; break
            }
        }
        if (track < 0 || format == null) return null
        ex.selectTrack(track)

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val channels = if (format.containsKey(MediaFormat.KEY_CHANNEL_COUNT))
            format.getInteger(MediaFormat.KEY_CHANNEL_COUNT) else 1

        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()
        val info = MediaCodec.BufferInfo()
        val out = ShortBuf()
        var sawInputEnd = false
        var sawOutputEnd = false

        try {
            while (!sawOutputEnd) {
                if (!sawInputEnd) {
                    val inIx = codec.dequeueInputBuffer(10_000)
                    if (inIx >= 0) {
                        val buf = codec.getInputBuffer(inIx)!!
                        val n = ex.readSampleData(buf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(inIx, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            sawInputEnd = true
                        } else {
                            codec.queueInputBuffer(inIx, 0, n, ex.sampleTime, 0)
                            ex.advance()
                        }
                    }
                }
                val outIx = codec.dequeueOutputBuffer(info, 10_000)
                if (outIx >= 0) {
                    val buf = codec.getOutputBuffer(outIx)!!
                    buf.position(info.offset)
                    buf.limit(info.offset + info.size)
                    appendMono(buf, channels, out)
                    codec.releaseOutputBuffer(outIx, false)
                    if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEnd = true
                }
            }
        } finally {
            runCatching { codec.stop() }
            runCatching { codec.release() }
        }
        val samples = out.toArray()
        return if (samples.isEmpty()) null else Pcm(rate, samples)
    }

    /** 16-bit LE decoder output, downmixed to one channel. */
    private fun appendMono(buf: ByteBuffer, channels: Int, out: ShortBuf) {
        val sb = buf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val frames = sb.remaining() / channels
        for (f in 0 until frames) {
            var acc = 0
            for (c in 0 until channels) acc += sb.get().toInt()
            out.add((acc / channels).toShort())
        }
    }

    /** Trim to a sample window and write a mono 16-bit WAV. */
    fun writeTrimmed(dest: File, pcm: Pcm, startSample: Int, endSample: Int) {
        val a = startSample.coerceIn(0, pcm.samples.size)
        val b = endSample.coerceIn(a, pcm.samples.size)
        val bytes = ByteBuffer.allocate((b - a) * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (i in a until b) bytes.putShort(pcm.samples[i])
        Wav.write(dest, pcm.rate, bytes.array())
    }

    /** Growable short buffer; ArrayList<Short> boxes and a decode is millions. */
    private class ShortBuf {
        private var arr = ShortArray(1 shl 16)
        private var n = 0
        fun add(s: Short) {
            if (n == arr.size) arr = arr.copyOf(arr.size * 2)
            arr[n++] = s
        }
        fun toArray(): ShortArray = arr.copyOf(n)
    }
}
