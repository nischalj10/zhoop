package com.noop.ai

import org.json.JSONArray
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Newer OpenAI chat and reasoning models reject `max_tokens` and `temperature` and want
 * `max_completion_tokens`. The non-streamed path always resent in that shape, but the coach streams,
 * and the streamed path sent the classic shape once and surfaced the 400. Both transports now share
 * one body builder and one rule for which 400s ask for the modern shape. The rejection table is shared
 * with Swift `AICoachModernParamsStreamTests`.
 */
class AiCoachModernParamsTest {

    private val messages = JSONArray()

    @Test fun classicStreamedBodyCarriesMaxTokensAndTemperature() {
        val body = AiCoach.openAiChatBody("m", messages, modernParams = false, stream = true)
        assertEquals(4096, body.getInt("max_tokens"))
        assertEquals(0.6, body.getDouble("temperature"), 0.0)
        assertTrue(body.getBoolean("stream"))
        assertFalse(body.has("max_completion_tokens"))
    }

    @Test fun modernStreamedBodyCarriesOnlyMaxCompletionTokens() {
        val body = AiCoach.openAiChatBody("m", messages, modernParams = true, stream = true)
        assertEquals(4096, body.getInt("max_completion_tokens"))
        assertFalse(body.has("max_tokens"))
        assertFalse(body.has("temperature"))
        assertTrue(body.getBoolean("stream"))
    }

    @Test fun unstreamedBodiesCarryNoStreamFlag() {
        assertFalse(AiCoach.openAiChatBody("m", messages, modernParams = false, stream = false).has("stream"))
        assertFalse(AiCoach.openAiChatBody("m", messages, modernParams = true, stream = false).has("stream"))
    }

    @Test fun whichRejectionsAskForTheModernShape() {
        assertTrue(
            AiCoach.shouldRetryOpenAiModernParams(
                """{"error":{"message":"Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}}""",
            ),
        )
        assertTrue(
            AiCoach.shouldRetryOpenAiModernParams(
                """{"error":{"message":"Unsupported value: 'temperature' does not support 0.6 with this model."}}""",
            ),
        )
        assertFalse(AiCoach.shouldRetryOpenAiModernParams("""{"error":{"message":"The model `nope` does not exist."}}"""))
        assertFalse(AiCoach.shouldRetryOpenAiModernParams(""))
    }
}
