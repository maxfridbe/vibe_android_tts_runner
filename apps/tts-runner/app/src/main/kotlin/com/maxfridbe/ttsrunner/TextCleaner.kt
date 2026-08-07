package com.maxfridbe.ttsrunner

import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import org.jsoup.nodes.Element
import org.jsoup.nodes.TextNode
import java.net.HttpURLConnection
import java.net.URL

/** Turns whatever lands via the share sheet into speakable prose.
 *  Blocking network when the share is a bare URL — call from a worker thread. */
object TextCleaner {

    private val BARE_URL = Regex("^https?://\\S+$")
    private val INLINE_URL = Regex("https?://\\S+")

    /** id/class fragments that mark page furniture rather than article prose.
     *  Anchored to a word start so "main-content" survives while "content-nav"
     *  does not. */
    private val JUNK = Regex(
        "(^|[-_ ])(nav|navbar|menu|sidebar|side-bar|share|sharing|social|related|recirc|" +
        "recommend|promo|advert|ads?|banner|cookie|consent|gdpr|newsletter|subscribe|" +
        "signup|sign-up|paywall|comment|disqus|breadcrumb|pagination|pager|masthead|" +
        "toolbar|popup|modal|widget|footer|trending|most-read|read-next|author-box|" +
        "byline|caption|skip-link|screen-reader)([-_ ]|$)",
        RegexOption.IGNORE_CASE)

    /** Whole lines that are page chrome even when they survive the DOM pass. */
    private val BOILERPLATE = Regex(
        "^(advertisement|sponsored( content)?|share this|read more|related( stories| articles)?|" +
        "sign (up|in)|subscribe( now)?|newsletter|follow us|leave a comment|comments?|" +
        "photo|image|illustration|credit|getty images|copyright|all rights reserved|" +
        "cookie policy|privacy policy|terms of service|skip to (main )?content)\\b.{0,40}$",
        RegexOption.IGNORE_CASE)

    /** Emoji and dingbats: the model either mangles them or reads their name. */
    private val EMOJI = Regex("[\\uD83C-\\uD83E][\\uDC00-\\uDFFF]|[\\u2190-\\u21FF\\u2600-\\u27BF\\u2B00-\\u2BFF\\uFE0F\\u200D]")

    data class Cleaned(val title: String?, val text: String)

    fun clean(shared: String): Cleaned {
        val trimmed = shared.trim()
        // shares often arrive as "Some headline https://example.com/x"
        val url = BARE_URL.find(trimmed)?.value
            ?: trimmed.takeIf { it.length < 400 }?.let { INLINE_URL.find(it)?.value }
        if (url != null && trimmed.replace(url, "").trim().length < 120) {
            return try {
                fetchArticle(url)
            } catch (e: Exception) {
                // no network / hostile site: fall back to whatever text came with it
                Cleaned(null, cleanPlain(trimmed))
            }
        }
        return Cleaned(null, cleanPlain(trimmed))
    }

    /** Strip things that read terribly aloud: URLs, markdown markup,
     *  reference brackets, emoji, repeated whitespace. */
    fun cleanPlain(text: String): String {
        var t = text
        t = t.replace(INLINE_URL, "")
        t = t.replace(Regex("!\\[[^\\]]*\\]\\([^)]*\\)"), "")     // md images
        t = t.replace(Regex("\\[([^\\]]+)\\]\\([^)]*\\)"), "$1")  // md links -> label
        t = t.replace(Regex("[*_#`>|]{1,3}"), " ")                // md decoration
        t = t.replace(Regex("\\[\\d+\\]"), "")                    // [17] citations
        t = t.replace(EMOJI, "")
        t = t.replace(' ', ' ')                              // nbsp
        t = t.replace(Regex("[\\u2018\\u2019\\u201b]"), "'")      // smart quotes
        t = t.replace(Regex("[\\u201c\\u201d]"), "\"")
        t = t.replace(Regex("[ \\t]+"), " ")
        t = t.replace(Regex(" *\\n *"), "\n")
        t = t.replace(Regex("\\n{3,}"), "\n\n")
        // drop leftover chrome lines and immediate repeats (nav echoed in body)
        val out = mutableListOf<String>()
        for (line in t.split("\n")) {
            val l = line.trim()
            if (l.isEmpty()) { if (out.lastOrNull() != "") out.add(""); continue }
            if (BOILERPLATE.matches(l)) continue
            if (out.lastOrNull()?.equals(l, ignoreCase = true) == true) continue
            out.add(l)
        }
        return out.joinToString("\n").replace(Regex("\\n{3,}"), "\n\n").trim()
    }

    /** Fetch a shared URL and extract readable article text: headline plus the
     *  prose of the densest, least link-heavy container — a small readability
     *  pass, since page furniture (nav, share bars, cookie banners, comments,
     *  related-story rails) is unbearable read aloud. */
    private fun fetchArticle(url: String): Cleaned {
        val doc = fetchDoc(url, 0)
        val title = title(doc)          // before stripping: the h1 often lives in a <header>
        stripFurniture(doc)
        val container = bestContainer(doc)

        var body = prose(container)
        if (body.length < 200) body = prose(doc.body())          // over-eager strip
        if (body.length < 200) body = doc.body().text()          // last resort

        return Cleaned(title, cleanPlain(listOfNotNull(title, body).joinToString("\n\n")))
    }

    /** Fetch and parse, following redirects HttpURLConnection will not: a
     *  cross-protocol Location, and the `<meta http-equiv=refresh>` stubs that
     *  static hosts leave behind for moved pages (the Rust blog's old .html
     *  URLs land on a one-word "Redirect" page otherwise). */
    private fun fetchDoc(url: String, hops: Int): Document {
        if (hops > 4) throw java.io.IOException("too many redirects")
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.connectTimeout = 15000
        conn.readTimeout = 30000
        conn.instanceFollowRedirects = true
        conn.setRequestProperty("User-Agent",
            "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Mobile Safari/537.36")
        conn.setRequestProperty("Accept-Language", "en;q=0.9")
        val code = conn.responseCode
        if (code in 300..399) {
            val loc = conn.getHeaderField("Location")
            conn.disconnect()
            if (loc.isNullOrBlank()) throw java.io.IOException("redirect without a target")
            return fetchDoc(URL(URL(url), loc).toString(), hops + 1)
        }
        val type = conn.contentType ?: ""
        if (type.isNotBlank() && !type.contains("html", true) && !type.contains("xml", true)) {
            conn.disconnect()
            throw java.io.IOException("not a web page: $type")
        }
        val html = conn.inputStream.bufferedReader().use { it.readText() }
        conn.disconnect()

        val doc = Jsoup.parse(html, url)
        val refresh = doc.selectFirst("meta[http-equiv=refresh]")?.attr("content")
        val target = refresh?.let {
            Regex("url\\s*=\\s*(.+)", RegexOption.IGNORE_CASE).find(it)
                ?.groupValues?.get(1)?.trim()?.trim('\'', '"')
        }
        if (!target.isNullOrBlank()) return fetchDoc(URL(URL(url), target).toString(), hops + 1)
        return doc
    }

    private fun title(doc: Document): String? {
        val og = doc.selectFirst("meta[property=og:title], meta[name=twitter:title]")?.attr("content")
        val raw = og?.takeIf { it.isNotBlank() }
            ?: doc.selectFirst("h1")?.text()?.takeIf { it.isNotBlank() }
            ?: doc.title().takeIf { it.isNotBlank() }
            ?: return null
        // "Headline | The Site" / "Headline - The Site" -> "Headline". Split on
        // the LAST separator and only when the tail is short, so a headline
        // that legitimately contains a dash survives.
        val cut = Regex(" [|\\u2013\\u2014-] ").findAll(raw).lastOrNull()
        val head = cut?.let { raw.substring(0, it.range.first) }?.trim()
        return if (head != null && head.length > 12 && raw.length - head.length <= 30) head else raw.trim()
    }

    /** Headings after which a page is bibliography, not prose. */
    private val END_MATTER = Regex(
        "^(references|external links|see also|further reading|notes|citations|bibliography|" +
        "sources|footnotes|related articles|more from|read next|comments)\\b.{0,20}$",
        RegexOption.IGNORE_CASE)

    private fun stripFurniture(doc: Document) {
        doc.select("script, style, noscript, template, svg, canvas, iframe, object, embed, " +
            "form, input, textarea, select, button, nav, aside, footer, header, figcaption, " +
            "video, audio, table.infobox").remove()
        doc.select("[aria-hidden=true], [hidden], [role=navigation], [role=banner], " +
            "[role=complementary], [role=contentinfo], [role=search], [role=dialog]").remove()
        // by id/class, deepest first so a removed parent doesn't hide matches
        doc.select("[class], [id]").reversed().forEach { el ->
            if (JUNK.containsMatchIn(el.className()) || JUNK.containsMatchIn(el.id())) el.remove()
        }
    }

    /** Readability-style scoring: prose length minus twice the link text, so a
     *  link-dense rail never beats the article even when it is longer. */
    private fun bestContainer(doc: Document): Element {
        var best: Element? = null
        var bestScore = 0
        for (el in doc.select("article, main, div, section, td")) {
            val paras = el.select("p, blockquote, pre")
            if (paras.size < 2) continue
            val textLen = paras.sumOf { it.text().length }
            val linkLen = el.select("a").sumOf { it.text().length }
            val score = textLen - 2 * linkLen + if (el.tagName() == "article") 400 else 0
            // >= keeps the deepest of equally-scoring nested wrappers (document
            // order lists ancestors first), which trims stray wrapper chrome
            if (score > 0 && score >= bestScore) { bestScore = score; best = el }
        }
        return best ?: doc.body()
    }

    /** Block elements in document order, one per line, list items only when
     *  they read like sentences rather than menu entries. */
    private fun prose(container: Element): String {
        container.select("br").forEach { it.replaceWith(TextNode("\n")) }
        val nested = setOf("p", "blockquote", "li", "pre")
        val parts = mutableListOf<String>()
        for (el in container.select("h1, h2, h3, h4, p, blockquote, pre, li")) {
            if (el.parents().any { it.tagName() in nested }) continue
            var t = el.text().trim()
            if (t.isBlank()) continue
            // everything past "References"/"See also" is a link list
            if (el.tagName().startsWith("h") && END_MATTER.matches(t)) break
            if (el.tagName() == "li" && t.length < 60 && !t.endsWith(".") && !t.endsWith("?")) continue
            if (t.length < 25 && !t.endsWith(".") && !t.endsWith("?") && !t.endsWith("!")) {
                // headings are short by nature: keep them, but end the sentence
                // so the model pauses instead of running into the next line
                if (el.tagName().startsWith("h")) t = "$t." else continue
            }
            parts.add(t)
        }
        return parts.joinToString("\n\n")
    }
}
