"""API authentication, rate limiting and request IDs."""

from __future__ import annotations

import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import HTTPException, Request, status

from core.config import ensure_api_token, settings
from core.logging import request_id_var


def _token_from(request: Request) -> str:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.headers.get("x-jarvis-token", "").strip()


async def require_auth(request: Request) -> None:
    """Every route except /health requires the bearer token from .env (JARVIS_API_TOKEN)."""
    expected = settings.JARVIS_API_TOKEN or ensure_api_token()
    supplied = _token_from(request)
    if not supplied or not secrets.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API token. Send 'Authorization: Bearer <JARVIS_API_TOKEN>' (see .env).",
            headers={"WWW-Authenticate": "Bearer"},
        )


class RateLimiter:
    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True


rate_limiter = RateLimiter()


async def rate_limit(request: Request) -> None:
    client = request.client.host if request.client else "unknown"
    if not rate_limiter.check(client, settings.RATE_LIMIT_PER_MINUTE):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests — slow down.")


async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    token = request_id_var.set(rid)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Response-Time-ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
    return response
