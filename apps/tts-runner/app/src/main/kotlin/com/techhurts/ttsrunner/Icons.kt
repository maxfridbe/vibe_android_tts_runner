package com.techhurts.ttsrunner

import android.content.Context
import android.graphics.Typeface
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.RelativeSizeSpan
import android.text.style.TypefaceSpan
import android.widget.Button
import android.widget.TextView

/** FontAwesome Free (Solid) glyphs, embedded as a TTF in assets/fonts.
 *
 *  Buttons across the app read as an icon followed by their label: the leading
 *  glyph is drawn in the FontAwesome face via a TypefaceSpan while the label
 *  keeps the default face, so a single Button carries both without a compound
 *  drawable. Codepoints below were verified against fa-solid-900.ttf (6.5.2). */
object Icons {
    const val PLAY = "\uf04b"
    const val PAUSE = "\uf04c"
    const val STOP = "\uf04d"
    const val SAVE = "\uf0c7"         // floppy-disk
    const val SHARE = "\uf064"        // share (arrow)
    const val ROTATE = "\uf021"       // arrows-rotate (play again)
    const val MIC = "\uf130"          // microphone
    const val PLUS = "\uf067"
    const val TRASH = "\uf2ed"        // trash-can
    const val EDIT = "\uf044"         // pen-to-square
    const val COPY = "\uf0c5"
    const val DOWNLOAD = "\uf019"
    const val GEAR = "\uf013"
    const val COMMENT = "\uf075"
    const val SERVER = "\uf233"
    const val INFO = "\uf05a"         // circle-info
    const val CLOSE = "\uf00d"        // xmark
    const val VOLUME = "\uf028"       // volume-high
    const val LIST = "\uf03a"
    const val IMPORT = "\uf56f"       // file-import
    const val MORE = "\uf141"         // ellipsis
    const val WAND = "\uf0d0"         // wand-magic (design)
    const val FOLDER = "\uf07c"       // folder-open
    const val CHECK = "\uf058"        // circle-check
    const val CLONE = "\uf24d"
    const val LINK = "\uf0c1"
    const val GLOBE = "\uf0ac"
    const val HEADPHONES = "\uf025"
    const val FILE = "\uf15c"         // file-lines

    @Volatile private var tf: Typeface? = null

    /** The embedded FontAwesome face, loaded once and cached. */
    fun font(ctx: Context): Typeface =
        tf ?: Typeface.createFromAsset(ctx.applicationContext.assets, "fonts/fa-solid-900.ttf")
            .also { tf = it }

    /** A "glyph  label" CharSequence: the glyph in the FA face, then the label
     *  in the default face after a gap. An empty label yields an icon-only
     *  string. */
    fun label(ctx: Context, glyph: String, text: String): CharSequence {
        val sb = SpannableStringBuilder(glyph)
        sb.setSpan(TypefaceSpan(font(ctx)), 0, glyph.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        if (text.isNotEmpty()) sb.append("  ").append(text)
        return sb
    }

    /** Prefix a button's text with an icon glyph. Pass [text] to also set the
     *  label; by default it keeps whatever the button already shows. */
    fun on(btn: Button, glyph: String, text: String = btn.text.toString()) {
        btn.text = label(btn.context, glyph, text)
    }

    /** Render a view as an icon-only glyph in the FA face (for round transport
     *  buttons drawn as text). [scale] enlarges the glyph relative to text. */
    fun only(view: TextView, glyph: String, scale: Float = 1f) {
        val sb = SpannableStringBuilder(glyph)
        sb.setSpan(TypefaceSpan(font(view.context)), 0, glyph.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        if (scale != 1f) sb.setSpan(RelativeSizeSpan(scale), 0, glyph.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        view.text = sb
    }
}
