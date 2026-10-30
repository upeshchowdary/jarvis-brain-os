"""Browser agent: action parsing, guard rails, and a live run in headless Chromium when available."""

import asyncio
import glob
import os

import pytest

from automation.browser_agent import BrowserAgent, format_elements, guard, parse_action

ELEMENTS = {
    0: {"id": 0, "tag": "input", "type": "text", "name": "Search"},
    1: {"id": 1, "tag": "input", "type": "password", "name": "Password"},
    2: {"id": 2, "tag": "button", "type": "submit", "name": "Place order"},
    3: {"id": 3, "tag": "a", "type": "", "name": "Next page"},
}


def test_parse_action_handles_fences_and_garbage():
    assert parse_action('```json\n{"action":"CLICK","id":3}\n```') == {"action": "click", "id": 3}
    assert parse_action("no idea")["action"] == "ask_user"


def test_guard_blocks_passwords_and_risky_clicks():
    assert "password" in guard({"action": "type", "id": 1, "text": "x"}, ELEMENTS).lower()
    assert "Place order" in guard({"action": "click", "id": 2}, ELEMENTS)
    assert guard({"action": "click", "id": 3}, ELEMENTS) is None
    assert guard({"action": "type", "id": 0, "text": "mug"}, ELEMENTS) is None


def test_format_elements_marks_input_types():
    text = format_elements(list(ELEMENTS.values()))
    assert "[1] input[password] “Password”" in text and "[3] a “Next page”" in text


def _chromium():
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM") or next(iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux64/chrome")), None)
    try:
        import playwright  # noqa: F401
    except ImportError:
        return None
    return exe if exe and os.path.exists(exe) else None


HTML = """<html><body><input placeholder='Search products'>
<button onclick="document.body.insertAdjacentHTML('beforeend','<p>Result: blue mug</p><button>Buy now</button>')">Search</button>
</body></html>"""


@pytest.mark.skipif(_chromium() is None, reason="Chromium/Playwright not available")
def test_live_agent_types_clicks_and_stops_before_buying():
    from playwright.async_api import async_playwright

    async def scenario():
        async with async_playwright() as p:
            browser = await p.chromium.launch(executable_path=_chromium(), args=["--no-sandbox"])
            page = await browser.new_page()
            await page.set_content(HTML)
            plan = iter([{"action": "type", "id": 0, "text": "mug"}, {"action": "click", "id": 1}, {"action": "click", "id": 2}])

            async def decide(goal, snap, history):
                return next(plan)

            outcome = await BrowserAgent().run("buy a mug", page=page, decide=decide)
            body = await page.inner_text("body")
            await browser.close()
            return outcome, body

    outcome, body = asyncio.run(scenario())
    assert "blue mug" in body  # the click really happened
    assert outcome.needs_user and "Buy now" in outcome.message
    assert [s.ok for s in outcome.steps] == [True, True]
