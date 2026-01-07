
import asyncio
import re
from typing import List, Optional, Dict, Any
from playwright.async_api import async_playwright, Page, Browser
WAIT_BEFORE_RETRY_SECONDS = 60
WAIT_POLL_INTERVAL = 3

POLL_INTERVAL = 0.5
STABLE_SECONDS_REQUIRED = 2
FOOTER_TEXT = ["google terms", "privacy policy", "gemini can make mistakes"]


async def _wait_for_avatar_animation_complete(page, timeout=240.0, poll_interval=0.15):
    """
    Wait until the avatar/lottie animation indicates completion or timeout.
    Returns True if completed, False if timed out.
    """
    waited = 0.0
    # Candidate selectors observed in your snippet
    selectors = [
        'div[data-test-lottie-animation-status]',            # explicit attribute
        '.avatar_primary_animation[data-test-lottie-animation-status]',
        'div.lottie-animation[data-test-lottie-animation-status]',
        '.avatar_primary_animation',                         # fallback: any avatar animation node
    ]

    while waited < timeout:
        for sel in selectors:
            try:
                exists = await page.query_selector(sel)
            except Exception:
                exists = None
            if not exists:
                continue
            try:
                # read attribute if present
                status = await page.evaluate("(el) => el.getAttribute('data-test-lottie-animation-status')", exists)
            except Exception:
                status = None
            # If attribute explicitly says completed, we're done
            if status and status.lower() == 'completed':
                # debug
                print(f"[gemini debug] avatar animation status=completed (selector={sel})")
                return True
            # If element contains an <svg> while writing and later becomes an <img> or static PNG,
            # check for presence of <svg> vs <img>
            try:
                has_svg = await page.evaluate("(el) => !!el.querySelector('svg')", exists)
                has_img = await page.evaluate("(el) => !!el.querySelector('img')", exists)
            except Exception:
                has_svg = False
                has_img = False
            # heuristics: if svg disappears or img appears, treat as completed
            if not has_svg and has_img:
                print(f"[gemini debug] avatar switched to img (selector={sel})")
                return True
        await asyncio.sleep(poll_interval)
        waited += poll_interval

    print("[gemini debug] avatar animation wait timed out")
    return False

async def _get_last_ai_message(page: Page, required_blocks: Optional[List[str]] = None) -> str:
    """
    Waits for avatar completion then extracts cleaned plain text.
    - Always preserves the <response>...</response> wrapper.
    - Preserves any additional wrappers listed in required_blocks.
    - Returns plain text; the <response> wrapper is kept intact in the returned string.
    """
    try:
        await _wait_for_avatar_animation_complete(page, timeout=240.0, poll_interval=0.12)
    except Exception:
        pass

    required_blocks = required_blocks or []
    # Ensure 'response' is always treated as required/preserved
    required_lower = set([b.lower() for b in required_blocks] + ['response'])

    return await page.evaluate(
        """(requiredBlocks) => {
        function unescapeHtmlEntities(s) {
            if (!s) return '';
            return s.replace(/&lt;/g, '<')
                    .replace(/&gt;/g, '>')
                    .replace(/&amp;/g, '&')
                    .replace(/&quot;/g, '"')
                    .replace(/&#39;/g, "'");
        }
        function nodeVisible(el) {
            try { return el && el.offsetParent !== null; } catch (e) { return false; }
        }
        const preferredSelectors = [
            'message-content',
            '[id^="model-response-message-content"]',
            '[id^="message-content-id-"]',
            '.tutor-markdown-rendering',
            '.markdown-main-panel'
        ];
        let candidates = [];
        for (const sel of preferredSelectors) {
            const found = Array.from(document.querySelectorAll(sel)).filter(nodeVisible);
            if (found.length) { candidates = found; break; }
        }
        if (!candidates.length) {
            candidates = Array.from(document.querySelectorAll('[class*="message-content"], [class*="model-response"], [data-path-to-node]')).filter(nodeVisible);
        }

        // Helper: strip wrapper markers only for tags NOT in requiredBlocks
        function stripWrapperMarkersIfNotRequired(text, tagName) {
            if (!text) return text;
            const lowerTag = (tagName || '').toLowerCase();
            if (requiredBlocks && requiredBlocks.map(t => t.toLowerCase()).includes(lowerTag)) {
                return text; // preserve wrapper for required tags
            }
            // remove both literal and escaped wrappers for non-required tags
            const openLiteral = new RegExp('^\\\\s*<' + tagName + '>\\\\s*', 'i');
            const closeLiteral = new RegExp('\\\\s*<\\\\/' + tagName + '>\\\\s*$', 'i');
            const openEscaped = new RegExp('^\\\\s*&lt;' + tagName + '&gt;\\\\s*', 'i');
            const closeEscaped = new RegExp('\\\\s*&lt;\\\\/' + tagName + '&gt;\\\\s*$', 'i');
            let out = text.replace(openLiteral, '').replace(closeLiteral, '').replace(openEscaped, '').replace(closeEscaped, '');
            return out;
        }

        for (let i = candidates.length - 1; i >= 0; i--) {
            const container = candidates[i];
            try {
                const ariaBusy = container.getAttribute && container.getAttribute('aria-busy');
                if (ariaBusy && ariaBusy.toLowerCase() === 'true') continue;

                const paragraphs = Array.from(container.querySelectorAll('p[data-path-to-node]')).filter(nodeVisible);
                let pieces = [];
                if (paragraphs.length) {
                    for (const p of paragraphs) {
                        const txt = (p.innerText || '').trim();
                        if (!txt) continue;
                        pieces.push(unescapeHtmlEntities(txt));
                    }
                } else {
                    const txt = (container.innerText || '').trim();
                    if (txt) pieces.push(unescapeHtmlEntities(txt));
                }
                if (!pieces.length) continue;

                // Join paragraphs with a blank line
                let joined = pieces.join('\\n\\n').trim();

                // If the UI included a <response> wrapper (literal or escaped), preserve it.
                // Detect presence first
                const hasLiteralResponseOpen = /^\\s*<response>/i.test(joined);
                const hasLiteralResponseClose = /<\\/response>\\s*$/i.test(joined);
                const hasEscapedResponseOpen = /^\\s*&lt;response&gt;/i.test(joined);
                const hasEscapedResponseClose = /&lt;\\/response&gt;\\s*$/i.test(joined);

                // If response markers are present, do NOT strip them (we always preserve response)
                if (!(hasLiteralResponseOpen || hasEscapedResponseOpen)) {
                    // If response wasn't present in the scraped text, wrap the cleaned content in <response>...</response>
                    joined = '<response>' + '\\n' + joined + '\\n' + '</response>';
                } else {
                    // If escaped markers were present, unescape them but keep the wrapper intact
                    if (hasEscapedResponseOpen || hasEscapedResponseClose) {
                        joined = joined.replace(/^\\s*&lt;response&gt;\\s*/i, '<response>');
                        joined = joined.replace(/\\s*&lt;\\/response&gt;\\s*$/i, '</response>');
                    }
                    // If literal markers present, leave them as-is
                }

                // For other wrapper candidates (div, span, etc.), strip only if NOT required
                const wrapperCandidates = ['div', 'span'];
                for (const w of wrapperCandidates) {
                    joined = stripWrapperMarkersIfNotRequired(joined, w);
                }

                // Remove simple tags that are not required (but keep required wrappers)
                joined = joined.replace(/<\\/?(div|span)[^>]*>/gi, (m, tag) => {
                    return requiredBlocks && requiredBlocks.map(t => t.toLowerCase()).includes(tag.toLowerCase()) ? m : '';
                });

                // Collapse excessive blank lines
                joined = joined.replace(/\\n{3,}/g, '\\n\\n').trim();

                return joined;
            } catch (e) {
                continue;
            }
        }

        // fallback: any visible p[data-path-to-node]
        const anyP = Array.from(document.querySelectorAll('p[data-path-to-node]')).filter(nodeVisible);
        if (anyP.length) {
            try {
                const pieces = anyP.map(p => (p.innerText || '').trim()).filter(Boolean).map(unescapeHtmlEntities);
                let joined = pieces.join('\\n\\n').trim();

                const hasLiteralResponseOpen = /^\\s*<response>/i.test(joined);
                const hasEscapedResponseOpen = /^\\s*&lt;response&gt;/i.test(joined);

                if (!(hasLiteralResponseOpen || hasEscapedResponseOpen)) {
                    joined = '<response>' + '\\n' + joined + '\\n' + '</response>';
                } else {
                    if (hasEscapedResponseOpen) {
                        joined = joined.replace(/^\\s*&lt;response&gt;\\s*/i, '<response>');
                        joined = joined.replace(/\\s*&lt;\\/response&gt;\\s*$/i, '</response>');
                    }
                }

                const wrapperCandidates = ['div', 'span'];
                for (const w of wrapperCandidates) {
                    joined = stripWrapperMarkersIfNotRequired(joined, w);
                }

                joined = joined.replace(/\\n{3,}/g, '\\n\\n').trim();
                return joined;
            } catch (e) {}
        }
        return '';
    }
    """,
        list(required_lower),
    )



def _extract_blocks(text: str, tags: list) -> dict:
    results = {}
    for tag in tags:
        # allow optional spaces/newlines after opening and before closing tags, case-insensitive
        pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.DOTALL | re.IGNORECASE)
        results[tag] = pattern.findall(text)
    return results
def _compose_prompt(system_prompt: Optional[str], user_prompt: str) -> str:
    """
    Compose a single message for UIs that lack a separate system field.
    Preserves a clear split for the model to follow.
    """
    system_prompt_prefix = (
        "System Prompt='''You are a Deep Reasoning Helper that plans there responses to provide the best answers inside <think> blocks you plan the steps you should take to properly analyze the user prompt to determine the best way to respond then based on that plan the steps you should take to prepare your response finally execute all steps within the <think> block to form your final response"
        "to user prompts. Always provide a honest response following instructions exactly as given to all prompts omit the <think> steps in your response perform them before you start writing your reply to the user and only include you final response.'''"
    )
    sys_text = f"{system_prompt_prefix}\n{(system_prompt or '').strip()}" if system_prompt else system_prompt_prefix
    return f"System Prompt:\n{sys_text}\n\nUser Prompt:\n{user_prompt.strip()}"

def _common_prefix_len(a: str, b: str) -> int:
    """Return length of common prefix of a and b."""
    l = min(len(a), len(b))
    i = 0
    while i < l and a[i] == b[i]:
        i += 1
    return i

class Gemini:
    def __init__(
        self,
        url: str = "https://gemini.google.com/app",
        headless: bool = True,
        poll_interval: float = POLL_INTERVAL,
        stable_seconds_required: float = STABLE_SECONDS_REQUIRED,
    ) -> None:
        self.url = url
        self.headless = headless
        self.poll_interval = poll_interval
        self.stable_seconds_required = stable_seconds_required

        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None
        self._open = False

    async def start(self) -> None:
        if self._open:
            return
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page()
        await self._page.goto(self.url)
        self._open = True

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.end_conversation()

    async def _send_message_to_ui(self, message: str) -> None:
        assert self._page is not None, "Session not started. Call start() first."
        editor = await self._page.wait_for_selector('textarea, [contenteditable="true"]')
        try:
            await editor.fill(message)
        except Exception:
            await self._page.evaluate(
                "(el, val) => { if (el.isContentEditable) el.innerText = val; else el.value = val; }",
                editor,
                message,
            )
        await editor.press("Enter")

    async def _stream_ai_reply(self) -> str:
        """
        Streams the reply by printing deltas to stdout.
        Returns the final full text of the last AI message after it stabilizes.
        """
        assert self._page is not None, "Session not started. Call start() first."
        full_text = ""
        stable_time = 0.0

        # Wait until some AI text appears
        while True:
            current = await _get_last_ai_message(self._page)
            if current.strip():
                break
            await asyncio.sleep(self.poll_interval)

        # Stream until stable
        while True:
            current = await _get_last_ai_message(self._page)
            cur = (current or "")
            prev = (full_text or "")

            if cur.strip() != prev.strip():
                # compute common prefix then print the new tail
                cp = _common_prefix_len(prev, cur)
                new_part = cur[cp:]
                # Trim leading whitespace to avoid weird spacing
                print(new_part, end="", flush=True)
                full_text = cur
                stable_time = 0.0
            else:
                stable_time += self.poll_interval

            if stable_time >= self.stable_seconds_required:
                break

            await asyncio.sleep(self.poll_interval)

        print("\n--- Reply complete ---\n")
        return full_text

    async def send(
            self,
            user_prompt: str,
            system_prompt: Optional[str] = None,
            *,
            return_blocks: Optional[List[str]] = ["response"],
            required_blocks: Optional[List[str]] = None,
            stream: bool = True,
            max_retries: int = 3
    ) -> Dict[str, Any]:
        assert self._open and self._page is not None, "Session not started. Call start() first."
        composed = _compose_prompt(system_prompt, user_prompt)
        attempt = 0

        def _normalize(s: str) -> str:
            if not s:
                return ""
            s = s.strip()
            s = re.sub(r"^['\"]|['\"]$", "", s)
            s = re.sub(r"\s+", " ", s).lower()
            return s

        def _is_echo(candidate: str, user_text: str) -> bool:
            if not candidate:
                return True
            c = _normalize(candidate)
            u = _normalize(user_text)
            if not c or not u:
                return True
            if c == u:
                return True
            cp_len = _common_prefix_len(c, u)
            if cp_len >= int(len(u) * 0.95) and len(c) <= len(u) + 40:
                return True
            return False

        async def _wait_for_completion_and_parse():
            """
            Wait up to WAIT_BEFORE_RETRY_SECONDS, polling every WAIT_POLL_INTERVAL,
            and re-checking for all required blocks on every iteration.
            """
            waited = 0

            while waited < WAIT_BEFORE_RETRY_SECONDS:
                current = await _get_last_ai_message(self._page, required_blocks=required_blocks)
                if current and current.strip():
                    blocks = _extract_blocks(current, return_blocks or [])

                    # Remove echoes
                    for tag in return_blocks or []:
                        blocks[tag] = [e for e in blocks.get(tag, []) if not _is_echo(e, user_prompt)]

                    # Check that all required blocks are present
                    if required_blocks:
                        missing = [b for b in required_blocks if not blocks.get(b)]
                        if not missing:
                            print("[gemini] Detected completed response during wait window.")
                            return current, blocks
                    else:
                        # No required blocks, just return first valid blocks
                        if any(blocks.values()):
                            print("[gemini] Detected completed response during wait window.")
                            return current, blocks

                await asyncio.sleep(WAIT_POLL_INTERVAL)
                waited += WAIT_POLL_INTERVAL

            return None, None

        while True:
            attempt += 1
            await self._send_message_to_ui(composed)
            await _wait_for_avatar_animation_complete(self._page, timeout=240.0)

            if stream:
                full_text = await self._stream_ai_reply()
            else:
                full_text = ""
                stable_time = 0.0
                while True:
                    current = await _get_last_ai_message(self._page, required_blocks=required_blocks)
                    if current.strip():
                        break
                    await asyncio.sleep(self.poll_interval)
                while True:
                    current = await _get_last_ai_message(self._page, required_blocks=required_blocks)
                    if current.strip() != full_text.strip():
                        full_text = current
                        stable_time = 0.0
                    else:
                        stable_time += self.poll_interval
                    if stable_time >= self.stable_seconds_required:
                        break
                    await asyncio.sleep(self.poll_interval)

            blocks = _extract_blocks(full_text, return_blocks or [])

            for tag in return_blocks or []:
                blocks[tag] = [e for e in blocks.get(tag, []) if not _is_echo(e, user_prompt)]

            if required_blocks:
                missing = [b for b in required_blocks if not blocks.get(b)]
                if missing:
                    print(f"[gemini] Missing blocks {missing}. Waiting for completion before retry...")
                    waited_text, waited_blocks = await _wait_for_completion_and_parse()
                    if waited_blocks:
                        return {
                            "full_text": waited_text,
                            "blocks": waited_blocks,
                            "used_prompt": composed,
                        }

                    if attempt < max_retries:
                        print(f"[gemini] Timeout reached. Retrying send ({attempt}/{max_retries})...")
                        continue

            break

        return {
            "full_text": full_text,
            "blocks": blocks,
            "used_prompt": composed,
        }

    async def end_conversation(self) -> None:
        """
        Closes the browser and stops Playwright. Idempotent and safe to call multiple times.
        """
        if not self._open:
            # ensure playwright stop if partially started
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
            return

        try:
            if self._page is not None:
                try:
                    await self._page.close()
                except Exception:
                    pass
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    pass
        finally:
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
            self._page = None
            self._browser = None
            self._playwright = None
            self._open = False

    async def restart_conversation(self) -> None:
        await self.end_conversation()
        await self.start()


async def example():
    async with Gemini(headless=False) as session:
        base_system_instructions = (
            "You are roleplaying as 'Roger'. Respond naturally and helpfully to the user's query.\n\n"
        )

        # Force final reply to be wrapped in <response> tags and avoid echoing user
        system_prompt = (
                base_system_instructions +
                "When you produce the reply, place only the final assistant reply inside a single "
                "<response>...</response> block. Do NOT include any other explanatory text outside those tags.\n\n"
                "Important: Do NOT echo the user's input back verbatim. Provide a helpful reply that does not simply repeat the user's words."
        )

        # List of messages to send
        messages = [
            "Hey Roger whats up",
            "Hey Roger can you tell me what is happening in the world right now?",
            "Hey Roger whats up",
            "Hey Roger can you tell me what is happening in the world right now?"
        ]

        for user_prompt in messages:
            result = await session.send(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                return_blocks=["response"],
                stream=True
            )
            print("Result blocks:", result["blocks"])

            # Pause between messages to allow the model to stabilize
            await asyncio.sleep(60)
if __name__ == "__main__":
    print("Testing TTT.py...")
    # Run with headless=False so we can see what's happening
    async def test():
        async with Gemini(headless=False) as session:
            result = await session.send(
                user_prompt="Say hello in one sentence.",
                system_prompt="Keep it short.",
                stream=True
            )
            print("Success! Got response:", result["blocks"])
    asyncio.run(test())
