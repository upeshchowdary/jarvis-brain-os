"""Live Internet Search Tool for real-time web knowledge retrieval."""

import urllib.parse
import httpx
import re
from typing import Dict, Any, List
from brain.tool_router import BaseBrainTool
from brain.logger import logger


class InternetSearchTool(BaseBrainTool):
    """Tool for fetching real-time search engine results from the web."""

    name: str = "internet_search"
    description: str = "Perform real-time live internet web search to retrieve current news, recent facts, weather, and up-to-date web information."
    version: str = "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string to look up on the internet.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search result snippets to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        query = kwargs.get("query", "").strip()
        max_results = int(kwargs.get("max_results", 5))

        if not query:
            return {"success": False, "results": [], "error": "Search query cannot be empty."}

        clean_query = self._extract_search_keywords(query)
        logger.info(f"InternetSearchTool executing search: raw='{query}' -> cleaned='{clean_query}'")

        try:
            results = []
            headers = {
                "User-Agent": "JarvisAI/1.0 (https://github.com/upeshchowdary/jarvis-brain-os)",
            }

            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={urllib.parse.quote(clean_query)}&limit={max_results}&namespace=0&format=json"
                wiki_res = await client.get(wiki_url, headers=headers)
                if wiki_res.status_code == 200:
                    wiki_data = wiki_res.json()
                    if isinstance(wiki_data, list) and len(wiki_data) >= 4:
                        titles = wiki_data[1]
                        snippets = wiki_data[2]
                        urls = wiki_data[3]
                        for idx in range(min(len(titles), max_results)):
                            if titles[idx]:
                                snippet_text = snippets[idx] if idx < len(snippets) and snippets[idx] else f"Information regarding {titles[idx]}"
                                page_url = urls[idx] if idx < len(urls) and urls[idx] else f"https://en.wikipedia.org/wiki/{urllib.parse.quote(titles[idx])}"
                                results.append({
                                    "title": titles[idx],
                                    "snippet": snippet_text,
                                    "url": page_url,
                                })

                logger.info(f"InternetSearchTool retrieved {len(results)} live web search result snippets.")
                return {
                    "success": True,
                    "query": query,
                    "count": len(results),
                    "results": results,
                }

        except Exception as exc:
            logger.error(f"InternetSearchTool execution error: {exc}")
            return {"success": False, "results": [], "error": str(exc)}

    @staticmethod
    def _extract_search_keywords(raw: str) -> str:
        """Clean conversational words from search query."""
        words = raw.split()
        stopwords = {"what", "is", "the", "latest", "news", "on", "about", "today", "current", "tell", "me", "find", "search", "can", "you", "show", "recent"}
        filtered = [w for w in words if w.lower().strip("?,.!") not in stopwords]
        result = " ".join(filtered).strip("?,.!")
        return result if len(result) > 2 else raw
