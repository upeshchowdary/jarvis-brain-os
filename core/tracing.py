"""Optional Langfuse tracing (Langfuse Python SDK v3).

A strict no-op unless ``settings.LANGFUSE_ENABLED`` is true *and* the ``langfuse`` package imports; when off,
``langfuse`` is never imported. Tracing must never break or slow a turn: every SDK call is wrapped, the first
failure is logged at debug level and the trace is dropped. The SDK exports in a background thread, so there is
no blocking network in the turn path (we never call ``flush``).

Everything that touches the SDK lives in :class:`_Adapter` (assumed v3 API: ``Langfuse(public_key=, secret_key=,
host=)``, ``client.start_span(name=, input=, metadata=)``, ``span.start_span(...)`` / ``span.start_generation(
name=, model=, usage_details=, metadata=)``, ``obs.update(...)``, ``obs.end()``).
"""

from __future__ import annotations

import contextvars
from typing import Any, Dict, Optional

from core.logging import logger

from core.config import settings

_client: Any = None
_client_failed = False
_warned = False
_current: contextvars.ContextVar[Optional["TurnTrace"]] = contextvars.ContextVar("jarvis_turn_trace", default=None)


def _debug_once(exc: BaseException) -> None:
    global _warned
    if not _warned:
        _warned = True
        logger.debug(f"[Tracing] disabled after error: {exc!r}")


def _reset() -> None:
    """Forget the cached client (tests)."""
    global _client, _client_failed, _warned
    _client, _client_failed, _warned = None, False, False


def _get_client() -> Any:
    global _client, _client_failed
    if _client is not None or _client_failed or not settings.LANGFUSE_ENABLED:
        return _client
    try:
        from langfuse import Langfuse  # imported only when the user opted in

        kw: Dict[str, Any] = {}
        if settings.LANGFUSE_PUBLIC_KEY:
            kw["public_key"] = settings.LANGFUSE_PUBLIC_KEY
        if settings.LANGFUSE_SECRET_KEY:
            kw["secret_key"] = settings.LANGFUSE_SECRET_KEY
        if settings.LANGFUSE_HOST:
            kw["host"] = settings.LANGFUSE_HOST
        _client = Langfuse(**kw)
    except Exception as exc:  # missing package, bad config, ...
        _client_failed = True
        _debug_once(exc)
    return _client


class _Adapter:
    """The only code that talks to the SDK."""

    def __init__(self, client: Any, name: str, input: Any, metadata: Dict[str, Any]) -> None:
        self.root = client.start_span(name=name, input=input, metadata=metadata)

    def span(self, name: str, input: Any = None, output: Any = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        obs = self.root.start_span(name=name, input=input, metadata=metadata or {})
        if output is not None:
            obs.update(output=output)
        obs.end()

    def generation(self, name: str, model: str, usage: Dict[str, int], metadata: Dict[str, Any]) -> None:
        obs = self.root.start_generation(name=name, model=model, usage_details=usage, metadata=metadata)
        obs.end()

    def finish(self, output: Any, metadata: Dict[str, Any]) -> None:
        self.root.update(output=output, metadata=metadata)
        self.root.end()


class TurnTrace:
    """One turn. Does nothing (and is safe to call) when tracing is off or has failed."""

    def __init__(self, query: str, session_id: str, channel: str) -> None:
        self._a: Optional[_Adapter] = None
        self._token: Optional[contextvars.Token] = None
        client = _get_client()
        if client is not None:
            try:
                self._a = _Adapter(client, "turn", query, {"session_id": session_id, "channel": channel})
            except Exception as exc:
                _debug_once(exc)

    def _safe(self, fn, *args, **kw) -> None:
        if self._a is None:
            return
        try:
            fn(*args, **kw)
        except Exception as exc:
            self._a = None  # drop the rest of this trace
            _debug_once(exc)

    def intent(self, intent: Any, ms: float) -> None:
        if self._a is not None:
            self._safe(self._a.span, "intent", input=getattr(intent, "request", ""),
                       metadata={"kind": intent.kind, "source": intent.source, "ms": ms})

    def llm(self, resp: Any, ms: float) -> None:
        if self._a is not None:
            usage = {k: v for k, v in (resp.usage or {}).items() if isinstance(v, int)}
            self._safe(self._a.generation, "llm", resp.model,
                       usage, {"provider": resp.provider, "latency_ms": round(ms, 1)})

    def tool(self, name: str, ok: bool, state: str, ms: float) -> None:
        if self._a is not None:
            self._safe(self._a.span, f"tool:{name}", metadata={"ok": ok, "state": state, "ms": round(ms, 1)})

    def finish(self, result: Any) -> None:
        if self._a is not None:
            self._safe(self._a.finish, result.response, {"route": result.route, "status": result.status})
            self._a = None

    def __enter__(self) -> "TurnTrace":
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc) -> None:
        if self._a is not None:  # the turn ended without a result (error / abandoned)
            self._safe(self._a.finish, None, {"status": "aborted"})
            self._a = None
        _current.set(None)


def trace_turn(query: str, session_id: str, channel: str = "text") -> TurnTrace:
    """``with trace_turn(...) as tr:`` — the pipeline's hooks below find ``tr`` through a context variable."""
    return TurnTrace(query, session_id, channel)


def record_intent(intent: Any, ms: float) -> None:
    tr = _current.get()
    if tr is not None:
        tr.intent(intent, ms)


def record_llm(resp: Any, ms: float) -> None:
    tr = _current.get()
    if tr is not None:
        tr.llm(resp, ms)


def record_tool(name: str, ok: bool, state: str, ms: float) -> None:
    tr = _current.get()
    if tr is not None:
        tr.tool(name, ok, state, ms)
