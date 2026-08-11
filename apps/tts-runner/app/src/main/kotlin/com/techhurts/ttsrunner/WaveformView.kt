package com.techhurts.ttsrunner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View
import java.io.File
import kotlin.math.abs
import kotlin.math.max

/** Amplitude-bar waveform for a 16-bit mono WAV, with a playback position
 *  overlay and tap/drag-to-seek. */
class WaveformView(context: Context) : View(context) {

    private var peaks = IntArray(0)      // 0..Short.MAX per bucket
    private var progress = 0f            // 0..1
    var onSeek: ((Float) -> Unit)? = null

    private val played = Paint().apply { color = 0xFF3F51B5.toInt() }
    private val pending = Paint().apply { color = 0xFFB0BEC5.toInt() }

    /** Downsample the wav's PCM into ~[buckets] max-amplitude bars. */
    fun loadWav(file: File, buckets: Int = 120) {
        peaks = try {
            val bytes = file.readBytes()
            if (bytes.size <= 44) IntArray(0) else {
                val n = (bytes.size - 44) / 2
                val out = IntArray(buckets)
                val per = max(1, n / buckets)
                for (b in 0 until buckets) {
                    var peak = 0
                    var i = b * per
                    val end = minOf(n, i + per)
                    // stride so huge files stay cheap (~64 probes per bucket)
                    val step = max(1, per / 64)
                    while (i < end) {
                        val lo = bytes[44 + 2 * i].toInt() and 0xff
                        val hi = bytes[44 + 2 * i + 1].toInt()
                        peak = max(peak, abs((hi shl 8) or lo))
                        i += step
                    }
                    out[b] = peak
                }
                out
            }
        } catch (_: Exception) {
            IntArray(0)
        }
        postInvalidate()   // loadWav may run off the UI thread
    }

    fun setProgress(p: Float) {
        progress = p.coerceIn(0f, 1f)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        if (peaks.isEmpty()) return
        val w = width.toFloat()
        val h = height.toFloat()
        val bar = w / peaks.size
        val cutoff = progress * peaks.size
        for ((i, p) in peaks.withIndex()) {
            val amp = max(0.05f, p / 32767f) * h / 2f
            canvas.drawRect(
                i * bar, h / 2f - amp, (i + 1) * bar - max(1f, bar * 0.2f), h / 2f + amp,
                if (i < cutoff) played else pending)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                if (width > 0) {
                    setProgress(event.x / width)
                    if (event.action == MotionEvent.ACTION_DOWN) performClick()
                }
                return true
            }
            MotionEvent.ACTION_UP -> {
                if (width > 0) onSeek?.invoke((event.x / width).coerceIn(0f, 1f))
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}
