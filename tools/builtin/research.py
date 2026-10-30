"""Web research: several searches + reading the best pages concurrently → material with numbered sources."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from tools.base import Tool, ToolContext, ToolResult, obj
from tools.builtin.web import ddg_search, fetch_text
from tools.registry import register


def query_variants(question: str) -> List[str]:
    q = question.strip().rstrip("?")
    return list(dict.fromkeys([q, f"{q} explained", f"{q} latest 2026"]))


@register
class Research(Tool):
    name = "research"
    description = (
        "Research a topic on the web in depth: runs several searches, reads the top pages and returns numbered "
        "sources. Use for questions that need up-to-date or multi-source answers; cite sources as [1], [2]."
    )
    parameters = obj({"question": {"type": "string"}, "depth": {"type": "integer", "minimum": 2, "maximum": 6, "default": 4}}, ["question"])
    untrusted_output = True
    final = False

    async def run(self, ctx: ToolContext, question: str = "", depth: int = 4, **_: Any) -> ToolResult:
        searches = await asyncio.gather(*(ddg_search(q, 5) for q in query_variants(question)), return_exceptions=True)
        seen, results = set(), []
        for batch in searches:
            if not isinstance(batch, list):
                continue
            for r in batch:
                if r["url"] and r["url"] not in seen:
                    seen.add(r["url"])
                    results.append(r)
        if not results:
            return ToolResult.error("Web research failed — no search results (are you online?).")
        top = results[: max(2, min(int(depth), 6))]
        pages = await asyncio.gather(*(fetch_text(r["url"], 3500) for r in top), return_exceptions=True)
        sources: List[Dict[str, str]] = []
        blocks = []
        for i, (r, page) in enumerate(zip(top, pages), 1):
            body = page if isinstance(page, str) and len(page) > 200 else r["snippet"]
            sources.append({"n": str(i), "title": r["title"], "url": r["url"]})
            blocks.append(f"[{i}] {r['title']} — {r['url']}\n{body}")
        return ToolResult.ok(f"Read {len(sources)} sources.", data=sources, content_for_llm="\n\n".join(blocks))
