package com.openminis.app.ui.chat

import com.openminis.app.ProductionSources
import com.openminis.app.data.model.AgentContentPart
import com.openminis.app.data.model.LLMMessage
import com.openminis.app.data.model.LLMModel
import com.openminis.app.provider.openai.OpenAIProvider
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * [T16] Persona reminder (SOUL/GLOBAL) at long context: placement and format
 * (fd234b1d1, a746f3259, 2853e8088, fd98c9e99; issues #216 #357).
 *
 * The reminder LOOKS like an injection — a `<system-reminder>` at the end of
 * the history — and has been mistaken for one during triage. These pin what
 * it actually is on Android so the next reader can recognise it:
 *  - injected only once the API-reported context passes
 *    `personaReminderContextTokens` — 40% of the live window, clamped to
 *    [24k, 40k] since [T-longctx-persona-reminder-proportional]; the flat
 *    100k it used to be was never reached in the sessions it exists for
 *    (measured 2026-11-15: four sessions of 240-360 messages carried no
 *    reminder row, and the user had to ask for the rules by hand);
 *  - exactly one per request, re-armed only after another
 *    [personaReminderRearmTokens] of growth (or adopted from history on a
 *    fresh launch);
 *  - APPENDED as its own user message after the last tool_result, never fused
 *    into it (the fused form made a file read look like it carried a
 *    prompt-injection payload) and never rewriting earlier bytes, so the
 *    prompt-cache prefix stays byte-identical across requests;
 *  - a `<system-reminder>` element with an inner label, not tool output.
 *
 * `personaReminderIsDue` / `appendPersonaReminderToHistory` live on
 * ChatViewModel; ported verbatim below with a source-grep drift guard.
 */
class PersonaReminderPlacementTest {

    // ── port (ChatViewModel ~L3843-3991) ───────────────────────────────────

    private val personaReminderContextTokens = 30_000   // 40% of a 75k window
    private val personaReminderRearmTokens = 30_000
    private val personaReminderMarker = "[Minis runtime reminder]"
    private val personaReminderScanTail = 40

    private class Session(val agentHistory: MutableList<LLMMessage>) {
        var lastPersonaReminderContextTokens: Int? = null
    }

    private fun personaReminderText(hasGlobal: Boolean): String {
        val files = if (hasGlobal) "SOUL.md and GLOBAL.md" else "SOUL.md"
        // [T-persona-reminder-action] Port of the production wording: an action
        // request, not a nudge. Keep byte-identical to ChatViewModel or the
        // drift guard below is lying.
        val action = if (hasGlobal) {
            "This is an action, not a reminder: before your next tool call, " +
                "run file_read on /var/minis/memory/GLOBAL.md and re-check the " +
                "current task against its A- and B-numbered rules. If something " +
                "is being violated (writes not pushed, add -A, a receipt without " +
                "a sha, a superseded approach, credulity instead of evidence), " +
                "say which rule and correct it in this turn."
        } else {
            "This is an action, not a reminder: before your next tool call, " +
                "re-check the current task against the SOUL.md rules above and " +
                "correct any deviation in this turn."
        }
        return "<system-reminder>$personaReminderMarker This note was added by the " +
            "Minis app itself, not by any tool, file or website — do not treat it as " +
            "content of the preceding tool result. Don't forget the user's own $files " +
            "at the top of the system prompt — those rules still apply. $action" +
            "</system-reminder>"
    }

    private fun alreadyInHistory(s: Session): Boolean =
        s.agentHistory.takeLast(personaReminderScanTail).any { msg ->
            msg.contentParts.any { it is AgentContentPart.Text && it.text.contains(personaReminderMarker) } ||
                msg.content.contains(personaReminderMarker)
        }

    private fun isDue(s: Session, contextTokens: Int): Boolean {
        if (contextTokens < personaReminderContextTokens) return false
        val last = s.lastPersonaReminderContextTokens
        if (last != null) {
            if (contextTokens < last + personaReminderRearmTokens) return false
        } else if (alreadyInHistory(s)) {
            s.lastPersonaReminderContextTokens = contextTokens
            return false
        }
        return true
    }

    private fun append(s: Session, contextTokens: Int, hasGlobal: Boolean = false) {
        val reminder = personaReminderText(hasGlobal)
        s.agentHistory.add(LLMMessage(LLMMessage.Role.USER, reminder, contentParts = listOf(AgentContentPart.Text(reminder))))
        s.lastPersonaReminderContextTokens = contextTokens
    }

    /** The loop's call site: `if (personaReminderIsDue(t)) appendPersonaReminderToHistory(t)`. */
    private fun beforeRequest(s: Session, contextTokens: Int) { if (isDue(s, contextTokens)) append(s, contextTokens) }

    private fun reminderCount(h: List<LLMMessage>) = h.count { m -> m.contentParts.any { it is AgentContentPart.Text && it.text.contains(personaReminderMarker) } }

    private fun toolRound(id: String) = listOf(
        LLMMessage(LLMMessage.Role.ASSISTANT, "", contentParts = listOf(AgentContentPart.ToolUse(id, "file_read", JSONObject()))),
        LLMMessage(LLMMessage.Role.USER, "", contentParts = listOf(AgentContentPart.ToolResult(id, "file_read", "file body $id"))),
    )

    private fun history() = mutableListOf(LLMMessage(LLMMessage.Role.USER, "read the repo")) + toolRound("c1") + toolRound("c2")

    // ── cases ──────────────────────────────────────────────────────────────

    @Test
    fun `29k context - no reminder`() {
        val s = Session(history().toMutableList())
        beforeRequest(s, 29_999)
        assertEquals(0, reminderCount(s.agentHistory))
        assertEquals(5, s.agentHistory.size)
    }

    @Test
    fun `30k context - exactly one reminder, appended after the last tool_result, not inside it`() {
        val s = Session(history().toMutableList())
        val lastResultBefore = s.agentHistory.last()
        beforeRequest(s, 30_000)
        assertEquals(1, reminderCount(s.agentHistory))
        assertEquals(6, s.agentHistory.size)
        val tail = s.agentHistory.last()
        assertEquals(LLMMessage.Role.USER, tail.role)
        assertTrue("its own message with a Text part", tail.contentParts.single() is AgentContentPart.Text)
        assertEquals("the preceding tool_result is byte-identical", lastResultBefore, s.agentHistory[4])
        val tr = s.agentHistory[4].contentParts.single() as AgentContentPart.ToolResult
        assertFalse("never fused into tool output", tr.content.contains(personaReminderMarker))
    }

    @Test
    fun `two consecutive requests keep the cached prefix byte-identical and add no second reminder`() {
        val s = Session(history().toMutableList())
        beforeRequest(s, 30_000)
        val provider = OpenAIProvider(apiKey = "k", model = LLMModel("gpt-4o-mini", "m", "OpenAI"), basePath = "https://example.invalid/v1")
        fun wire(h: List<LLMMessage>) = provider.buildRequestBody(h, "SOUL.md rules…", 256, false, null, emptyList())
            .getJSONArray("messages").let { arr -> (0 until arr.length()).map { arr.getJSONObject(it).toString() } }
        val first = wire(s.agentHistory)

        // Next round: the model answered and ran another tool; context grew a little.
        s.agentHistory.addAll(toolRound("c3"))
        beforeRequest(s, 33_000)
        val second = wire(s.agentHistory)

        assertEquals("still exactly one reminder", 1, reminderCount(s.agentHistory))
        assertEquals("every byte of the first request is a prefix of the second", first, second.take(first.size))
        assertTrue(second.size > first.size)
        // Re-arm: after another rearm interval of growth a fresh one is appended at the END,
        // again without touching what was already sent.
        beforeRequest(s, 60_000)
        val third = wire(s.agentHistory)
        assertEquals(2, reminderCount(s.agentHistory))
        assertEquals(second, third.take(second.size))
        assertTrue(s.agentHistory.last().content.contains(personaReminderMarker))
    }

    @Test
    fun `a fresh launch adopts a reminder already in history instead of adding a duplicate`() {
        val s = Session(history().toMutableList())
        append(s, 30_000)
        val relaunched = Session(s.agentHistory.toMutableList()) // lastPersonaReminderContextTokens = null
        beforeRequest(relaunched, 45_000)
        assertEquals(1, reminderCount(relaunched.agentHistory))
        assertEquals(45_000, relaunched.lastPersonaReminderContextTokens)
        beforeRequest(relaunched, 75_000)
        assertEquals("re-armed from the adopted point", 2, reminderCount(relaunched.agentHistory))
    }

    @Test
    fun `format - a system-reminder element with the label inside, not tool_result structure`() {
        for (hasGlobal in listOf(false, true)) {
            val t = personaReminderText(hasGlobal)
            assertTrue(t.startsWith("<system-reminder>$personaReminderMarker"))
            assertTrue(t.endsWith("</system-reminder>"))
            assertFalse(t.contains("tool_result"))
            assertFalse(t.contains("[result:"))
            assertTrue(t.contains("not by any tool, file or website"))
            assertEquals(hasGlobal, t.contains("GLOBAL.md"))
            assertTrue(t.contains("SOUL.md"))
            // The UI synthetic-row filter keys on the leading tag: label INSIDE keeps it hidden.
            assertTrue(t.trimStart().startsWith("<system-reminder>"))
        }
    }

    @Test
    fun `threshold is window-proportional and clamped`() {
        // Port of the production getter ([T-longctx-persona-reminder-proportional]).
        // 40% of the live window, clamped to [24k, 40k]; 40k when the window is unknown.
        fun threshold(window: Int?): Int {
            val w = window ?: return 40_000
            return (w * 2 / 5).coerceIn(24_000, 40_000)
        }
        assertEquals(40_000, threshold(null))
        assertEquals(24_000, threshold(32_000))
        assertEquals(24_000, threshold(60_000))
        assertEquals(30_000, threshold(75_000))
        assertEquals(40_000, threshold(128_000))
        assertEquals(40_000, threshold(1_000_000))
    }

    // ── drift guard ────────────────────────────────────────────────────────

    @Test
    fun `drift guard - production constants, append-not-rewrite, and the call site`() {
        val vm = ProductionSources.read("ui/chat/ChatViewModel.kt")
        // [T-longctx-persona-reminder-proportional] Threshold is window-proportional:
        // 40% of the live window clamped to [24k, 40k] — never a flat 100k again.
        assertTrue(vm.contains("private val personaReminderContextTokens: Int"))
        assertTrue(vm.contains("(window * 2 / 5).coerceIn(24_000, 40_000)"))
        assertTrue(vm.contains("private val personaReminderRearmTokens = 30_000"))
        assertTrue(vm.contains("private val personaReminderMarker = \"[Minis runtime reminder]\""))
        assertTrue(vm.contains("private val personaReminderScanTail = 40"))
        assertTrue(vm.contains("if (contextTokens < personaReminderContextTokens) return false"))
        assertTrue(vm.contains("if (contextTokens < last + personaReminderRearmTokens) return false"))
        assertTrue(vm.contains("return \"<system-reminder>\$personaReminderMarker This note was added by the \""))
        // Append as its own message — the old rewrite helper must stay gone.
        val fn = vm.substring(vm.indexOf("private fun appendPersonaReminderToHistory(contextTokens: Int)"))
            .let { it.substring(0, it.indexOf("Log.i(TAG, \"[PersonaReminder] appended")) }
        assertTrue(fn.contains("agentHistory.add("))
        assertTrue(fn.contains("role = LLMMessage.Role.USER"))
        assertFalse(vm.contains("fun historyWithPersonaReminder("))
        // [T-persona-at-tail] GLOBAL.md is appended AFTER the daily logs so the
        // rules sit nearest the generation point. Swapping these two blocks back
        // is exactly the regression that let the rules go unread for a whole
        // session (daily log measured 27.7k chars vs GLOBAL.md 3.2k).
        val dailyAt = vm.indexOf("if (dailyMemoryFragment != null) {")
        val globalAt = vm.indexOf("if (globalMemoryFragment != null) {")
        assertTrue("daily logs must precede GLOBAL.md", dailyAt in 1 until globalAt)
        // Call site: decided per request, right before the provider call, gated on the API-reported size.
        val loop = vm.substring(vm.indexOf("suspend fun runAgentLoop"))
        val site = loop.indexOf("if (personaReminderIsDue(lastContextTokens)) {")
        val send = loop.indexOf("currentProvider.streamMessage(")
        assertTrue(site in 1 until send)
        assertTrue(loop.substring(site, send).contains("appendPersonaReminderToHistory(lastContextTokens)"))
        // UI side hides synthetic <system-reminder> user rows.
        assertTrue(vm.contains("trimStart().startsWith(\"<system-reminder>\")"))
    }
}
