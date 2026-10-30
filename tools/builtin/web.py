"""Web tools: live search (DuckDuckGo, no key) and page reading. Output is untrusted."""

from __future__ import annotations

import asyncio
import re
from html import unescape
from typing import Any, Dict, List

import httpx

from core.logging import logger
from tools.base import Tool, ToolContext, ToolResult, obj
from tools.registry import register

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


async def ddg_search(query: str, max_results: int = 5, kind: str = "text") -> List[Dict[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError:
        raise RuntimeError("Web search needs the 'ddgs' package (pip install ddgs).")

    def _search() -> List[Dict[str, str]]:
        with DDGS() as ddgs:
            if kind == "news":
                raw = list(ddgs.news(query, max_results=max_results))
                return [{"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("url", ""), "date": r.get("date", "")} for r in raw]
            raw = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")} for r in raw]

    return await asyncio.wait_for(asyncio.to_thread(_search), timeout=12)


def html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h\d>", "\n", html, flags=re.IGNORECASE)
    text = unescape(re.sub(r"<[^>]+>", " ", html))
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


async def fetch_text(url: str, max_chars: int = 6000) -> str:
    async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        ctype = resp.headers.get("content-type", "")
        if "html" not in ctype and "text" not in ctype:
            raise RuntimeError(f"Unsupported content type: {ctype or 'unknown'}")
        return html_to_text(resp.text)[:max_chars]


@register
class WebSearch(Tool):
    name = "web_search"
    description = (
        "Search the internet for current information: news, weather, prices, scores, recent events, "
        "or facts you are unsure about. Returns result snippets with URLs."
    )
    parameters = obj({
        "query": {"type": "string", "description": "Search query"},
        "news": {"type": "boolean", "description": "Search recent news instead of the web", "default": False},
    }, ["query"])
    untrusted_output = True
    final = False

    async def run(self, ctx: ToolContext, query: str = "", news: bool = False, **_: Any) -> ToolResult:
        query = query.strip()
        if not query:
            return ToolResult.error("Search query is empty.")
        try:
            results = await ddg_search(query, 6, "news" if news else "text")
        except Exception as exc:
            logger.warning(f"[web_search] {exc}")
            return ToolResult.error(f"Web search failed: {exc}")
        if not results:
            return ToolResult.error("No search results found.")
        lines = [f"[{i}] {r['title']} — {r['snippet']} ({r['url']}{', ' + r['date'] if r.get('date') else ''})" for i, r in enumerate(results, 1)]
        return ToolResult.ok(f"Found {len(results)} results for “{query}”.", data=results, content_for_llm="\n".join(lines))


@register
class ReadWebpage(Tool):
    name = "read_webpage"
    description = "Fetch a web page and return its readable text (for summarising or answering questions about it)."
    parameters = obj({"url": {"type": "string"}}, ["url"])
    untrusted_output = True
    final = False

    async def run(self, ctx: ToolContext, url: str = "", **_: Any) -> ToolResult:
        if not url.startswith(("http://", "https://")):
            return ToolResult.error("Please give a full http(s) URL.")
        try:
            text = await fetch_text(url)
        except Exception as exc:
            return ToolResult.error(f"Couldn't read {url}: {exc}")
        return ToolResult.ok(f"Read {len(text)} characters from {url}.", data={"url": url}, content_for_llm=text)
