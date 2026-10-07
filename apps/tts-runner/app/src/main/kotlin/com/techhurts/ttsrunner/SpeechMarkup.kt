package com.techhurts.ttsrunner

/** Model-specific text controls; these strings are model input, not SSML. */
object SpeechMarkup {
    val kittenEmotions = listOf("angry", "contemplative", "excited", "joyful", "mundane",
        "nervous", "sad", "stern", "surprised", "tender")
    val kittenEvents = listOf("gasp", "giggle", "growl", "gulp", "laugh", "pause", "scoff", "sigh", "sob", "um")
    val supertonicEvents = listOf("laugh", "sigh", "breath", "cough", "cry", "yawn", "hmm", "um", "tsk", "kiss")
    private val leadingEmotion = Regex("^\\s*\\[(${kittenEmotions.joinToString("|")})]\\s*", RegexOption.IGNORE_CASE)
    val kittenExpression = Regex("\\[(${kittenEmotions.joinToString("|")})]|" +
        "<(${kittenEvents.joinToString("|")})>|\\(\\(\\([^()\\n]{1,80}\\)\\)\\)", RegexOption.IGNORE_CASE)

    fun withEmotion(text: String, emotion: String?): String {
        require(emotion == null || emotion in kittenEmotions)
        val body = text.replaceFirst(leadingEmotion, "")
        return if (emotion == null) body else "[$emotion] $body"
    }

    /** Preserve complete emphasis spans and apply a leading emotion to every
     *  chunk, since each chunk is an independent model invocation. */
    fun kittenChunks(text: String, maxChars: Int = Chunker.DEFAULT_CHUNK_CHARS): List<String> {
        val emotion = leadingEmotion.find(text)?.groupValues?.get(1)
        if (emotion == null && !text.contains("(((")) return Chunker.split(text, maxChars)
        val body = text.replaceFirst(leadingEmotion, "")
        val prefix = emotion?.let { "[$it] " } ?: ""
        val limit = (maxChars - prefix.length).coerceAtLeast(90)
        val chunks = mutableListOf<String>()
        var current = ""
        fun flush() {
            if (current.isNotBlank()) chunks.add(prefix + current)
            current = ""
        }
        for (match in Regex("\\(\\(\\([^()\\n]{1,80}\\)\\)\\)|\\S+").findAll(body)) {
            val word = match.value
            // Huge ordinary words/URLs can still be split. Supported markup fits the limit.
            val parts = if (word.length > limit) word.chunked(limit) else listOf(word)
            for (part in parts) {
                if (current.isNotEmpty() && current.length + 1 + part.length > limit) flush()
                current = if (current.isEmpty()) part else "$current $part"
                if (current.length >= limit / 2 && part.lastOrNull() in listOf('.', '!', '?', '…')) flush()
            }
        }
        flush()
        return chunks
    }
}
