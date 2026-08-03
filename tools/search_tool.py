"""Live Internet Search Tool powered by DDGS engine for real-time web knowledge retrieval."""

import asyncio
from typing import Dict, Any, List
from ddgs import DDGS
from brain.tool_router import BaseBrainTool
from brain.logger import logger
from brain.utils import is_internet_available


class InternetSearchTool(BaseBrainTool):
    """Tool for fetching real-time search engine results from the live web using DDGS."""

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

        if not is_internet_available():
            logger.warning("InternetSearchTool: No active internet connection. Web search skipped.")
            return {"success": False, "results": [], "error": "No active internet connection."}

        logger.info(f"InternetSearchTool executing live web search for: '{query}'")

        try:
            # Run blocking DDGS text search in an async executor thread
            loop = asyncio.get_running_loop()
            raw_results = await loop.run_in_executor(
                None,
                lambda: list(DDGS().text(query, max_results=max_results))
            )

            formatted_results: List[Dict[str, str]] = []
            for item in raw_results:
                title = item.get("title", "").strip()
                snippet = item.get("body", "").strip()
                url = item.get("href", "").strip()
                if title and snippet:
                    formatted_results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                    })

            logger.info(f"InternetSearchTool successfully retrieved {len(formatted_results)} live web search result snippets.")
            return {
                "success": True,
                "query": query,
                "count": len(formatted_results),
                "results": formatted_results,
            }

        except Exception as exc:
            logger.error(f"InternetSearchTool execution error: {exc}")
            return {"success": False, "results": [], "error": str(exc)}
