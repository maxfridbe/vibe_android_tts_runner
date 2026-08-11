package com.techhurts.ttsrunner

/** Natural-break text chunking, same strategy as the audiobook maker's
 *  splitter: prefer paragraph boundaries, then sentence boundaries, then a
 *  hard slice. ~13 chars ≈ 1 s of speech; 200 chars ≈ 15 s per chunk. Small
 *  chunks matter on phones: first audio sooner, visible progress, and a
 *  runaway generation burns minutes not tens of minutes (a single 396-char
 *  chunk once ran 24 min on a throttled screen-off phone). */
object Chunker {
    const val DEFAULT_CHUNK_CHARS = 200

    private val SENTENCE_END = Regex("(?<=[.!?…])\\s+")

    fun split(text: String, maxChars: Int = DEFAULT_CHUNK_CHARS): List<String> {
        val chunks = mutableListOf<String>()
        var current = StringBuilder()

        fun flush() {
            val s = current.toString().trim()
            if (s.isNotEmpty()) chunks.add(s)
            current = StringBuilder()
        }

        for (para in text.split(Regex("\\n\\s*\\n"))) {
            val p = para.trim().replace(Regex("\\s+"), " ")
            if (p.isEmpty()) continue
            if (current.length + p.length + 1 <= maxChars) {
                if (current.isNotEmpty()) current.append(" ")
                current.append(p)
                continue
            }
            flush()
            if (p.length <= maxChars) {
                current.append(p)
                continue
            }
            // paragraph too big: pack sentences
            for (sentence in SENTENCE_END.split(p)) {
                var s = sentence.trim()
                while (s.length > maxChars) {
                    // no sentence break either: hard slice at a space if possible
                    val cut = s.lastIndexOf(' ', maxChars).let { if (it < maxChars / 2) maxChars else it }
                    flush()
                    chunks.add(s.substring(0, cut).trim())
                    s = s.substring(cut).trim()
                }
                if (current.length + s.length + 1 > maxChars) flush()
                if (current.isNotEmpty()) current.append(" ")
                current.append(s)
            }
            flush()
        }
        flush()
        return chunks
    }
}
