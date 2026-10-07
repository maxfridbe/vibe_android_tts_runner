package com.techhurts.ttsrunner

import org.junit.Assert.*
import org.junit.Test

class SpeechMarkupTest {
    @Test fun plainTextKeepsExistingChunkBoundaries() {
        val text = "Hello, this is an ordinary sentence. ".repeat(25)
        assertEquals(Chunker.split(text), SpeechMarkup.kittenChunks(text))
    }

    @Test fun changingEmotionReplacesOnlyTheLeadingTag() {
        assertEquals("[joyful] Hello <sigh> section [3].",
            SpeechMarkup.withEmotion("[sad] Hello <sigh> section [3].", "joyful"))
        assertEquals("Hello <sigh> section [3].",
            SpeechMarkup.withEmotion("[joyful] Hello <sigh> section [3].", null))
    }

    @Test fun chunksKeepEmotionAndEmphasisIntact() {
        val body = "A quiet sentence. ".repeat(8) + "(((a very important promise))) <laugh> " + "Another sentence. ".repeat(9)
        val chunks = SpeechMarkup.kittenChunks("[tender] $body", 120)
        assertTrue(chunks.size > 1)
        assertTrue(chunks.all { it.startsWith("[tender] ") && it.length <= 120 })
        assertEquals(1, chunks.count { it.contains("(((a very important promise)))") })
        assertEquals(body.trim(), chunks.joinToString(" ") { it.removePrefix("[tender] ") })
    }

    @Test fun ordinaryBracketedTextDoesNotEnableExpressionMode() {
        assertFalse(SpeechMarkup.kittenExpression.containsMatchIn("See section [3], where x < 5."))
        assertFalse(SpeechMarkup.kittenExpression.containsMatchIn("[reverent] Hello."))
        assertTrue(SpeechMarkup.kittenExpression.containsMatchIn("[excited] Hello <gasp> (((friend)))!"))
    }

    @Test fun emptyInputDoesNotProduceASpeechChunk() {
        assertTrue(SpeechMarkup.kittenChunks("[sad]  ").isEmpty())
    }
}
