package com.techhurts.ttsrunner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.view.View
import java.io.File
import java.io.RandomAccessFile
import kotlin.math.abs
import kotlin.math.max

/** A waveform that grows to the right as audio is generated.
 *
 *  Unlike [WaveformView], which downsamples a finished WAV into a fixed number
 *  of bars, this appends one bar per fixed slice of audio and widens itself as
 *  more PCM lands — so the owner can drop it in a horizontal scroller and watch
 *  the shape extend in step with generation. It reads the growing raw-PCM
 *  capture (`last_audio.pcm`, s16 mono) incrementally, only touching the bytes
 *  added since the last tick, and falls back to a finished WAV for replay. */
class LiveWaveformView(context: Context) : View(context) {

    private val peaks = ArrayList<Int>()
    private var readOffset = 0L        // bytes of the pcm file already folded in
    private var carryPeak = 0          // running max for the in-progress bar
    private var carryCount = 0         // samples accumulated in the in-progress bar
    private var samplesPerBar = 1600   // ~66 ms at 24 kHz; reset by setRate()
    private var progress = -1f         // playback cursor 0..1, <0 hidden

    private val density = resources.displayMetrics.density
    private val barW = 3f * density
    private val gapW = 2f * density
    private val minWidth = 240f * density

    private val played = Paint().apply { color = 0xFF3F51B5.toInt(); isAntiAlias = true }
    private val pending = Paint().apply { color = 0xFF9FA8DA.toInt(); isAntiAlias = true }
    private val cursorPaint = Paint().apply { color = 0xFFE53935.toInt(); strokeWidth = 2f * density }

    /** Bucket width follows the sample rate so a second of speech is always the
     *  same on-screen length regardless of engine (24 kHz Qwen vs 44.1 kHz). */
    fun setRate(rate: Int) {
        if (rate > 0) samplesPerBar = max(256, (rate * 0.066f).toInt())
    }

    fun reset() {
        peaks.clear(); readOffset = 0; carryPeak = 0; carryCount = 0; progress = -1f
        requestLayout(); invalidate()
    }

    fun barCount() = peaks.size

    /** Fold whatever is new in the growing raw-PCM file into the bars. */
    fun appendPcm(file: File) {
        try {
            val len = file.length()
            if (len < readOffset) reset()             // file was recreated for a new job
            if (len <= readOffset) return
            RandomAccessFile(file, "r").use { raf ->
                raf.seek(readOffset)
                var remaining = (len - readOffset)
                if (remaining % 2L == 1L) remaining -= 1   // keep sample alignment
                val buf = ByteArray(remaining.coerceAtMost(1L shl 22).toInt())  // ≤4 MB/tick
                raf.readFully(buf)
                readOffset += buf.size
                foldSamples(buf, buf.size)
            }
            requestLayout(); invalidate()
        } catch (_: Exception) { }
    }

    /** Load a finished WAV into bars — the replay path, and also how the whole
     *  shape is rebuilt on completion so it includes the final chunk (which
     *  lands in the WAV after the last live append). Safe to call off the UI
     *  thread: the file is read on the caller's thread but every view mutation
     *  is posted back to the UI thread, so a background caller can't trip
     *  requestLayout()'s thread check and silently wipe the bars. */
    fun loadWav(file: File) {
        try {
            val bytes = file.readBytes()
            if (bytes.size <= 44) return
            val fresh = ArrayList<Int>()
            var peak = 0
            var count = 0
            var i = 44
            while (i + 1 < bytes.size) {
                val lo = bytes[i].toInt() and 0xff
                val hi = bytes[i + 1].toInt()
                val s = abs((hi shl 8) or lo)
                if (s > peak) peak = s
                if (++count >= samplesPerBar) { fresh.add(peak); peak = 0; count = 0 }
                i += 2
            }
            if (count > 0) fresh.add(peak)   // keep the trailing partial bar
            post {
                peaks.clear(); peaks.addAll(fresh)
                readOffset = 0; carryPeak = 0; carryCount = 0; progress = -1f
                requestLayout(); invalidate()
            }
        } catch (_: Exception) { }
    }

    private fun foldSamples(buf: ByteArray, size: Int, offset: Int = 0) {
        var i = offset
        while (i + 1 < size) {
            val lo = buf[i].toInt() and 0xff
            val hi = buf[i + 1].toInt()
            val s = abs((hi shl 8) or lo)
            if (s > carryPeak) carryPeak = s
            if (++carryCount >= samplesPerBar) {
                peaks.add(carryPeak); carryPeak = 0; carryCount = 0
            }
            i += 2
        }
    }

    /** Playback position, 0..1 across the whole waveform (<0 hides the cursor). */
    fun setProgress(p: Float) { progress = p; invalidate() }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        // width tracks the bar count so a HorizontalScrollView parent can scroll
        val want = max(minWidth, peaks.size * (barW + gapW) + gapW).toInt()
        val h = resolveSize(0, heightMeasureSpec)
        setMeasuredDimension(want, h)
    }

    override fun onDraw(canvas: Canvas) {
        val h = height.toFloat()
        val mid = h / 2f
        val cutoff = if (progress in 0f..1f) progress * peaks.size else Float.MAX_VALUE
        var x = gapW
        for ((i, p) in peaks.withIndex()) {
            val amp = max(0.06f, p / 32767f) * (h / 2f - gapW)
            canvas.drawRoundRect(
                x, mid - amp, x + barW, mid + amp, barW / 2f, barW / 2f,
                if (i < cutoff) played else pending)
            x += barW + gapW
        }
        if (progress in 0f..1f) {
            val cx = gapW + progress * peaks.size * (barW + gapW)
            canvas.drawLine(cx, 0f, cx, h, cursorPaint)
        }
    }
}
