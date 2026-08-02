"""Web Scraper Tool for extracting text and markdown from web pages."""

import re
import httpx
from typing import Dict, Any
from brain.tool_router import BaseBrainTool
from brain.logger import logger


class WebScraperTool(BaseBrainTool):
    """Tool for scraping and extracting text content from web URLs."""

    name: str = "web_scraper"
    description: str = "Fetch the text content of a web page from a URL for reading articles, documentation, or web pages."
    version: str = "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The HTTP or HTTPS URL of the web page to scrape.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters of text content to extract (default 4000).",
                    "default": 4000,
                },
            },
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        url = kwargs.get("url", "").strip()
        max_chars = int(kwargs.get("max_chars", 4000))

        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return {"success": False, "content": "", "error": "Invalid HTTP or HTTPS URL provided."}

        logger.info(f"WebScraperTool fetching URL: '{url}'")

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(url, headers=headers)
                if response.status_code != 200:
                    return {
                        "success": False,
                        "content": "",
                        "error": f"Failed to fetch page. HTTP status: {response.status_code}",
                    }

                clean_text = self._clean_html_to_text(response.text)
                truncated_text = clean_text[:max_chars]

                logger.info(f"WebScraperTool extracted {len(truncated_text)} characters from {url}")
                return {
                    "success": True,
                    "url": url,
                    "character_count": len(truncated_text),
                    "content": truncated_text,
                }

        except Exception as exc:
            logger.error(f"WebScraperTool error: {exc}")
            return {"success": False, "content": "", "error": str(exc)}

    @staticmethod
    def _clean_html_to_text(html: str) -> str:
        """Strip HTML tags, scripts, and styles to leave clean readable text."""
        # Remove script and style tags
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
