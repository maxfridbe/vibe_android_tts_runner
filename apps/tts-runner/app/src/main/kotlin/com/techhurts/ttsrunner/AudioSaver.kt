package com.techhurts.ttsrunner

import android.content.ContentValues
import android.content.Context
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaFormat
import android.media.MediaMuxer
import android.net.Uri
import android.os.ParcelFileDescriptor
import android.provider.MediaStore

/** Streams s16le mono PCM into an AAC .m4a under Music/TTS Runner via
 *  MediaStore. Android ships no MP3 encoder; AAC-in-m4a is the universally
 *  playable equivalent. The entry stays IS_PENDING until finish() so players
 *  never see a half-written file. */
class AudioSaver(private val context: Context, title: String, private val sampleRate: Int = 24000) {
    private val fileName =
        title.replace(Regex("[^\\w .-]"), " ").trim().ifBlank { "tts" }.take(60) + ".m4a"
    val uri: Uri
    private val pfd: ParcelFileDescriptor
    private val encoder: AacM4aWriter

    init {
        val values = ContentValues().apply {
            put(MediaStore.Audio.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Audio.Media.MIME_TYPE, "audio/mp4")
            put(MediaStore.Audio.Media.RELATIVE_PATH, "Music/TTS Runner")
            put(MediaStore.Audio.Media.IS_PENDING, 1)
        }
        uri = context.contentResolver.insert(
            MediaStore.Audio.Media.getContentUri(MediaStore.VOLUME_EXTERNAL_PRIMARY), values)
            ?: throw IllegalStateException("MediaStore insert failed")
        pfd = context.contentResolver.openFileDescriptor(uri, "w")
            ?: throw IllegalStateException("cannot open $uri for writing")
        // must match the engine that produced the PCM (24 kHz Qwen codec,
        // 44.1 kHz Supertonic vocoder) or the file plays at the wrong speed
        encoder = AacM4aWriter(pfd, sampleRate)
    }

    fun write(pcm: ByteArray) = encoder.write(pcm)

    /** Finalize the m4a and publish it; returns the user-visible path. */
    fun finish(): String {
        encoder.finish()
        pfd.close()
        context.contentResolver.update(uri,
            ContentValues().apply { put(MediaStore.Audio.Media.IS_PENDING, 0) }, null, null)
        return "Music/TTS Runner/$fileName"
    }

    /** Drop the pending entry (job stopped or failed). */
    fun abort() {
        runCatching { encoder.finish() }
        runCatching { pfd.close() }
        runCatching { context.contentResolver.delete(uri, null, null) }
    }
}

/** Minimal synchronous MediaCodec AAC-LC encoder muxed to MPEG-4. */
class AacM4aWriter(pfd: ParcelFileDescriptor, private val sampleRate: Int) {
    private val codec = MediaCodec.createEncoderByType(MediaFormat.MIMETYPE_AUDIO_AAC)
    private val muxer = MediaMuxer(pfd.fileDescriptor, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
    private val info = MediaCodec.BufferInfo()
    private var track = -1
    private var ptsUs = 0L
    private var done = false

    init {
        val fmt = MediaFormat.createAudioFormat(MediaFormat.MIMETYPE_AUDIO_AAC, sampleRate, 1).apply {
            setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
            setInteger(MediaFormat.KEY_BIT_RATE, 64_000)
        }
        codec.configure(fmt, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        codec.start()
    }

    fun write(pcm: ByteArray) {
        var off = 0
        while (off < pcm.size) {
            val idx = codec.dequeueInputBuffer(10_000)
            if (idx >= 0) {
                val buf = codec.getInputBuffer(idx)!!
                buf.clear()
                val n = minOf(buf.remaining(), pcm.size - off)
                buf.put(pcm, off, n)
                codec.queueInputBuffer(idx, 0, n, ptsUs, 0)
                ptsUs += n * 1_000_000L / (2L * sampleRate)
                off += n
            }
            drain(0)
        }
    }

    fun finish() {
        if (done) return
        var spins = 0
        while (spins++ < 200) {
            val idx = codec.dequeueInputBuffer(10_000)
            if (idx >= 0) {
                codec.queueInputBuffer(idx, 0, 0, ptsUs, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                break
            }
        }
        spins = 0
        while (!done && spins++ < 200) drain(10_000)  // bounded: ~2 s worst case
        if (track >= 0) muxer.stop()
        muxer.release()
        codec.stop()
        codec.release()
    }

    private fun drain(timeoutUs: Long) {
        while (true) {
            val idx = codec.dequeueOutputBuffer(info, timeoutUs)
            when {
                idx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                    track = muxer.addTrack(codec.outputFormat)
                    muxer.start()
                }
                idx >= 0 -> {
                    val buf = codec.getOutputBuffer(idx)!!
                    if (info.size > 0 && (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0 && track >= 0) {
                        muxer.writeSampleData(track, buf, info)
                    }
                    val eos = (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                    codec.releaseOutputBuffer(idx, false)
                    if (eos) { done = true; return }
                }
                else -> return
            }
        }
    }
}
