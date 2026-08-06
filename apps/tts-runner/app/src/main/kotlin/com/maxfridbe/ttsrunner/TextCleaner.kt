package com.maxfridbe.ttsrunner

import org.jsoup.Jsoup
import java.net.HttpURLConnection
import java.net.URL

/** Turns whatever lands via the share sheet into speakable prose.
 *  Blocking network when the share is a bare URL — call from a worker thread. */
object TextCleaner {

    private val BARE_URL = Regex("^https?://\\S+$")
    private val INLINE_URL = Regex("https?://\\S+")

    data class Cleaned(val title: String?, val text: String)

    fun clean(shared: String): Cleaned {
        val trimmed = shared.trim()
        if (BARE_URL.matches(trimmed)) {
            return fetchArticle(trimmed)
        }
        return Cleaned(null, cleanPlain(trimmed))
    }

    /** Strip things that read terribly aloud: URLs, markdown markup,
     *  reference brackets, repeated whitespace. */
    fun cleanPlain(text: String): String {
        var t = text
        t = t.replace(INLINE_URL, "")
        t = t.replace(Regex("!\\[[^\\]]*\\]\\([^)]*\\)"), "")      // md images
        t = t.replace(Regex("\\[([^\\]]+)\\]\\([^)]*\\)"), "$1")  // md links -> label
        t = t.replace(Regex("[*_#`>|]{1,3}"), " ")                    // md decoration
        t = t.replace(Regex("\\[\\d+\\]"), "")                     // [17] citations
        t = t.replace(Regex("[ \\t]+"), " ")
        t = t.replace(Regex("\\n{3,}"), "\n\n")
        return t.trim()
    }

    /** Fetch a shared URL and extract readable article text: title + the
     *  paragraphs of the densest text container (a small readability pass). */
    private fun fetchArticle(url: String): Cleaned {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 15000
        conn.readTimeout = 30000
        conn.instanceFollowRedirects = true
        conn.setRequestProperty("User-Agent", "Mozilla/5.0 (Android) TTSRunner/1.0")
        val html = conn.inputStream.bufferedReader().use { it.readText() }
        conn.disconnect()

        val doc = Jsoup.parse(html, url)
        doc.select("script, style, nav, header, footer, aside, form, iframe, noscript, figure figcaption").remove()

        val title = doc.selectFirst("h1")?.text()?.takeIf { it.isNotBlank() } ?: doc.title().takeIf { it.isNotBlank() }

        // prefer semantic article containers; else the element with the most <p> text
        val container = doc.selectFirst("article")
            ?: doc.selectFirst("main")
            ?: doc.select("div, section")
                .maxByOrNull { el -> el.children().filter { it.tagName() == "p" }.sumOf { it.text().length } }
            ?: doc.body()

        val paragraphs = container.select("p, h2, h3, li")
            .map { it.text().trim() }
            .filter { it.length > 40 || it.endsWith(".") }
        val body = if (paragraphs.isNotEmpty()) paragraphs.joinToString("\n\n") else container.text()

        return Cleaned(title, cleanPlain(body))
    }
}
