"""Pure helpers for turning spoken/typed web targets into URLs (no side effects)."""

from __future__ import annotations

import re
import urllib.parse
from typing import Optional

KNOWN_SITES = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "google drive": "https://drive.google.com",
    "drive": "https://drive.google.com",
    "google calendar": "https://calendar.google.com",
    "calendar": "https://calendar.google.com",
    "google maps": "https://maps.google.com",
    "maps": "https://maps.google.com",
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "chatgpt": "https://chatgpt.com",
    "claude": "https://claude.ai",
    "reddit": "https://www.reddit.com",
    "wikipedia": "https://www.wikipedia.org",
    "linkedin": "https://www.linkedin.com",
    "twitter": "https://x.com",
    "x": "https://x.com",
    "instagram": "https://www.instagram.com",
    "facebook": "https://www.facebook.com",
    "netflix": "https://www.netflix.com",
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "spotify web": "https://open.spotify.com",
    "whatsapp web": "https://web.whatsapp.com",
    "telegram web": "https://web.telegram.org",
    "leetcode": "https://leetcode.com",
    "hackerrank": "https://www.hackerrank.com",
    "coursera": "https://www.coursera.org",
    "python compiler": "https://www.online-python.com/",
    "online python compiler": "https://www.online-python.com/",
    "news": "https://news.google.com",
    "weather": "https://www.google.com/search?q=weather",
}

_DOMAIN_RE = re.compile(r"^(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:[/?#].*)?$", re.IGNORECASE)


def site_url(name: str) -> Optional[str]:
    """Map a site name / domain / URL to a URL, or None if it isn't recognisably a site."""
    target = name.strip().strip("'\"").rstrip(".")
    low = re.sub(r"\b(?:the\s+)?(?:website|site|webpage|web\s+page|page|homepage)\b", "", target.lower()).strip()
    if low in KNOWN_SITES:
        return KNOWN_SITES[low]
    if low.startswith(("http://", "https://")):
        return target
    compact = low.replace(" dot ", ".").replace(" ", "")
    if _DOMAIN_RE.match(compact):
        return compact if compact.startswith("http") else f"https://{compact}"
    return None


def search_url(query: str, engine: str = "google") -> str:
    q = urllib.parse.quote_plus(query.strip())
    engine = engine.lower()
    if engine == "youtube":
        return f"https://www.youtube.com/results?search_query={q}"
    if engine in ("bing",):
        return f"https://www.bing.com/search?q={q}"
    if engine in ("duckduckgo", "ddg"):
        return f"https://duckduckgo.com/?q={q}"
    if engine == "github":
        return f"https://github.com/search?q={q}"
    if engine in ("maps", "google maps"):
        return f"https://www.google.com/maps/search/{q}"
    if engine == "amazon":
        return f"https://www.amazon.in/s?k={q}"
    if engine == "wikipedia":
        return f"https://en.wikipedia.org/w/index.php?search={q}"
    return f"https://www.google.com/search?q={q}"
