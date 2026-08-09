package com.maxfridbe.ttsrunner

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.ParcelFileDescriptor
import androidx.core.content.FileProvider
import java.io.File

/** Share generated audio. Android has no MP3 encoder, so WAVs are converted
 *  to AAC .m4a (plays everywhere MP3 does) before sharing; MediaStore m4a
 *  outputs are shared directly by content uri. */
object AudioShare {

    fun shareUri(ctx: Context, uri: Uri, title: String) {
        ctx.startActivity(Intent.createChooser(
            Intent(Intent.ACTION_SEND)
                .setType("audio/mp4")
                .putExtra(Intent.EXTRA_STREAM, uri)
                .putExtra(Intent.EXTRA_SUBJECT, title)
                .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION),
            "Share audio").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }

    /** Convert a mono s16 WAV (our own output format) to m4a in the share
     *  cache and hand it to the chooser. Blocking; call off-main. */
    fun shareWavAsM4a(ctx: Context, wav: File, title: String) {
        val dir = File(ctx.cacheDir, "share").apply { mkdirs() }
        val out = File(dir, title.replace(Regex("[^\\w .-]"), " ").trim().ifBlank { "audio" }.take(60) + ".m4a")
        out.delete()
        val pcm = wav.readBytes()
        require(pcm.size > 44) { "empty audio" }
        // read the rate from the header: 24 kHz from Qwen, 44.1 kHz from
        // Supertonic — a fixed rate here resamples by accident
        val rate = java.nio.ByteBuffer.wrap(pcm, 24, 4)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN).int.takeIf { it in 8000..48000 } ?: 24000
        val pfd = ParcelFileDescriptor.open(out,
            ParcelFileDescriptor.MODE_CREATE or ParcelFileDescriptor.MODE_READ_WRITE)
        val enc = AacM4aWriter(pfd, rate)
        enc.write(pcm.copyOfRange(44, pcm.size))
        enc.finish()
        pfd.close()
        val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", out)
        shareUri(ctx, uri, title)
    }
}
