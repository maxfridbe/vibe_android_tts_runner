package com.maxfridbe.ttsrunner

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.view.MotionEvent
import android.view.View
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

/** Waveform with two draggable handles marking a selection, plus a playback
 *  cursor. Selection is reported as fractions 0..1; the owner maps them to
 *  samples. Sibling to WaveformView, which is seek-only. */
class TrimWaveformView(context: Context) : View(context) {

    private var peaks = IntArray(0)
    var selStart = 0f; private set
    var selEnd = 1f;   private set
    private var cursor = -1f          // playback position, <0 = hidden

    /** Fired while dragging and on release, so labels track live. */
    var onSelection: ((Float, Float) -> Unit)? = null

    private val inSel = Paint().apply { color = 0xFF3F51B5.toInt() }
    private val outSel = Paint().apply { color = 0xFFB0BEC5.toInt() }
    private val shade = Paint().apply { color = 0x223F51B5 }
    private val handle = Paint().apply { color = 0xFF3F51B5.toInt(); strokeWidth = 6f }
    private val cursorPaint = Paint().apply { color = 0xFFE53935.toInt(); strokeWidth = 3f }

    fun setPeaks(p: IntArray) { peaks = p; postInvalidate() }

    fun setSelection(a: Float, b: Float) {
        selStart = a.coerceIn(0f, 1f); selEnd = b.coerceIn(selStart, 1f)
        invalidate()
    }

    fun setCursor(p: Float) { cursor = p; invalidate() }

    override fun onDraw(canvas: Canvas) {
        if (peaks.isEmpty()) return
        val w = width.toFloat(); val h = height.toFloat()
        val bar = w / peaks.size
        for ((i, p) in peaks.withIndex()) {
            val amp = max(0.05f, p / 32767f) * h / 2f
            val frac = (i + 0.5f) / peaks.size
            canvas.drawRect(i * bar, h / 2f - amp, (i + 1) * bar - max(1f, bar * 0.2f),
                h / 2f + amp, if (frac in selStart..selEnd) inSel else outSel)
        }
        val xa = selStart * w; val xb = selEnd * w
        canvas.drawRect(xa, 0f, xb, h, shade)
        for (x in listOf(xa, xb)) {
            canvas.drawLine(x, 0f, x, h, handle)
            canvas.drawCircle(x, h / 2f, 18f, handle)
        }
        if (cursor in 0f..1f) canvas.drawLine(cursor * w, 0f, cursor * w, h, cursorPaint)
    }

    private var drag = 0            // 1 = start handle, 2 = end handle, 0 = none

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (width == 0) return false
        val f = (event.x / width).coerceIn(0f, 1f)
        when (event.action) {
            MotionEvent.ACTION_DOWN -> {
                // grab whichever handle is nearer the touch
                drag = if (abs(f - selStart) <= abs(f - selEnd)) 1 else 2
                performClick(); applyDrag(f); return true
            }
            MotionEvent.ACTION_MOVE -> { applyDrag(f); return true }
            MotionEvent.ACTION_UP -> { applyDrag(f); drag = 0; return true }
        }
        return super.onTouchEvent(event)
    }

    private fun applyDrag(f: Float) {
        val minGap = 0.02f
        if (drag == 1) selStart = min(f, selEnd - minGap).coerceAtLeast(0f)
        else if (drag == 2) selEnd = max(f, selStart + minGap).coerceAtMost(1f)
        onSelection?.invoke(selStart, selEnd)
        invalidate()
    }

    override fun performClick(): Boolean { super.performClick(); return true }
}
