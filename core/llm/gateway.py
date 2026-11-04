"""One LLM gateway for every provider (OpenAI-compatible Chat Completions).

All eight providers expose an OpenAI-compatible ``/chat/completions`` endpoint, so a
single client covers them: Groq, Gemini, OpenAI, Anthropic, OpenRouter, DeepSeek,
Ollama and LM Studio. Features:

* native tool calling (``tools`` / ``tool_calls``) and streaming (text + tool calls);
* per-call timeouts, retries with exponential backoff on 429 / 5xx / network errors;
* hedged calls: a cloud call still unanswered after ``LLM_HEDGE_AFTER_SECONDS`` gets one duplicate request on
  the provider's next key (or the next fallback provider); the first to succeed wins, the other is cancelled;
* a fallback chain from config (``DEFAULT_MODEL`` → ``LLM_FALLBACKS``);
* per-model ``reasoning_effort``, structured output (``response_format``) and keep-warm pings
  (``GET /models``: no completion quota is spent);
* one shared ``httpx.AsyncClient`` per event loop (connection reuse = lower latency);
* ``local_only`` calls for PRIVACY_MODE=local (file/screen content never leaves the PC).

LiteLLM was evaluated and deliberately not adopted: its PyPI package was compromised
twice in 2026 and a thin first-party client is faster to import and easier to audit.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from core import connectivity
from core.config import settings
from core.logging import logger
from core.metrics import metrics


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    local: bool = False

    def base_url(self) -> str:
        return {
            "openai": "https://api.openai.com/v1",
            "groq": settings.GROQ_BASE_URL.rstrip("/"),
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
            "anthropic": "https://api.anthropic.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "ollama": settings.OLLAMA_BASE_URL.rstrip("/").replace("localhost", "127.0.0.1") + "/v1",
            "lmstudio": settings.LM_STUDIO_BASE_URL.rstrip("/").replace("localhost", "127.0.0.1"),
        }[self.name]

    def keys(self) -> List[str]:
        return [] if self.local else settings.api_keys_for(self.name)

    def api_key(self) -> Optional[str]:
        keys = self.keys()
        return keys[_KEY_CURSOR.get(self.name, 0) % len(keys)] if keys else None

    def rotate_key(self) -> None:
        """Move to the provider's next key; the choice sticks for later calls."""
        if len(self.keys()) > 1:
            _KEY_CURSOR[self.name] = (_KEY_CURSOR.get(self.name, 0) + 1) % len(self.keys())
            logger.warning(f"[LLM] {self.name}: switching to API key #{_KEY_CURSOR[self.name] + 1} of {len(self.keys())}")

    def configured(self) -> bool:
        return self.local or bool(self.api_key())


# Upper bound for one keep-warm ping (seconds).
_WARM_TIMEOUT = 3.0
# Keep the connection warm only this long after the last real request (free-tier request caps are per day).
KEEPWARM_WINDOW_SECONDS = 15 * 60.0

# Index of the key currently in use per provider (only matters when a provider has several keys).
_KEY_CURSOR: Dict[str, int] = {}


def _key_exhausted(status: int, body: str) -> bool:
    """True when a response means *this key* is out of quota or rejected, so another key may work."""
    return status in (401, 403, 429) or (status == 400 and "api key" in body.lower())


PROVIDERS: Dict[str, ProviderSpec] = {
    name: ProviderSpec(name, local=name in ("ollama", "lmstudio"))
    for name in ("openai", "groq", "gemini", "anthropic", "openrouter", "deepseek", "ollama", "lmstudio")
}

_PREFIXES: Sequence[Tuple[str, str]] = (
    ("gemini", "gemini"), ("gemma", "ollama"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
    ("o4", "openai"), ("claude", "anthropic"), ("deepseek", "deepseek"), ("llama-3", "groq"),
    ("llama3", "ollama"), ("mixtral", "groq"), ("qwen", "ollama"), ("mistral", "ollama"), ("phi", "ollama"),
)


def resolve_model(spec: str) -> Tuple[str, str]:
    """'groq/llama-3.3-70b' → ('groq', 'llama-3.3-70b'); bare names are mapped by prefix."""
    spec = spec.strip()
    if "/" in spec:
        head, tail = spec.split("/", 1)
        if head.lower() in PROVIDERS:
            return head.lower(), tail
    low = spec.lower()
    for prefix, provider in _PREFIXES:
        if low.startswith(prefix):
            return provider, spec
    return "ollama", spec


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]
    raw_arguments: str = ""
    # Provider data that must be echoed back verbatim, e.g. Gemini's thought_signature.
    extra_content: Optional[Dict[str, Any]] = None

    def to_message(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.raw_arguments or json.dumps(self.arguments)},
        }
        if self.extra_content:
            msg["extra_content"] = self.extra_content
        return msg


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    latency_ms: float = 0.0
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    fallback_used: bool = False

    def assistant_message(self) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_message() for tc in self.tool_calls]
        return msg


class ProviderError(RuntimeError):
    """One provider call failed. ``status`` / ``headers`` are the HTTP response's (None / {} on transport errors)."""

    def __init__(self, message: str, status: Optional[int] = None, headers: Any = None,
                 timed_out: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.headers: Dict[str, str] = dict(headers or {})  # httpx gives lower-case names
        self.timed_out = timed_out


class LLMUnavailable(RuntimeError):
    """Raised when every candidate provider failed.

    ``status`` / ``headers`` / ``timed_out`` describe the last provider failure (e.g. a 429 with ``retry-after``),
    so callers that run their own fallback order can back off.
    """

    def __init__(self, errors: List[str], causes: Optional[List[BaseException]] = None) -> None:
        super().__init__("; ".join(errors) or "No LLM provider is configured.")
        self.errors = errors
        last = next((c for c in reversed(causes or []) if isinstance(c, ProviderError)), None)
        self.status: Optional[int] = last.status if last else None
        self.headers: Dict[str, str] = last.headers if last else {}
        self.timed_out: bool = bool(last and last.timed_out)


class _HedgeFailed(RuntimeError):
    """The primary request (and the hedge, if it was started) failed: ``errors`` = [primary's, hedge's]."""

    def __init__(self, errors: List[BaseException]) -> None:
        super().__init__(str(errors[0]))
        self.errors = errors


class _RetryableError(RuntimeError):
    def __init__(self, message: str, status: int, headers: Any) -> None:
        super().__init__(message)
        self.status, self.headers = status, dict(headers or {})


def _parse_arguments(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except (TypeError, ValueError):
        from brain.utils import extract_and_clean_json

        return extract_and_clean_json(str(raw))


# Gemini's documented placeholder for function calls the model didn't produce itself.
_PLACEHOLDER_SIGNATURE = {"google": {"thought_signature": "skip_thought_signature_validator"}}


def _sign_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Give unsigned tool calls (fast-path injections, other providers' calls) Gemini's placeholder.

    Gemini 3 rejects a request whose history holds a tool call without a thought_signature.
    Returns new message dicts; the caller's list is left untouched.
    """
    out = []
    for msg in messages:
        calls = msg.get("tool_calls")
        if msg.get("role") == "assistant" and calls and any(not c.get("extra_content") for c in calls):
            msg = {**msg, "tool_calls": [c if c.get("extra_content") else {**c, "extra_content": _PLACEHOLDER_SIGNATURE}
                                         for c in calls]}
        out.append(msg)
    return out


def _strip_extra_content(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop Gemini-only ``extra_content`` from tool calls for every other provider (new dicts; input untouched)."""
    out = []
    for msg in messages:
        calls = msg.get("tool_calls")
        if msg.get("role") == "assistant" and calls and any("extra_content" in c for c in calls):
            msg = {**msg, "tool_calls": [{k: v for k, v in c.items() if k != "extra_content"} for c in calls]}
        out.append(msg)
    return out


def _parse_tool_calls(message: Dict[str, Any]) -> List[ToolCall]:
    calls = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw = fn.get("arguments") or ""
        calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), _parse_arguments(raw),
                              raw if isinstance(raw, str) else json.dumps(raw), tc.get("extra_content")))
    return calls


class LLMGateway:
    def __init__(self) -> None:
        self._clients: Dict[int, httpx.AsyncClient] = {}
        self._transport: Optional[httpx.AsyncBaseTransport] = None
        # Monotonic time of the last real request (None: none yet). Keep-warm skips busy cycles and stops
        # KEEPWARM_WINDOW_SECONDS after it.
        self._last_request: Optional[float] = None
        self._keepwarm_task: Optional[asyncio.Task] = None

    # Tests inject an httpx.MockTransport here.
    def set_transport(self, transport: Optional[httpx.AsyncBaseTransport]) -> None:
        self._transport = transport
        self._clients.clear()

    def _client(self) -> httpx.AsyncClient:
        loop_id = id(asyncio.get_running_loop())
        client = self._clients.get(loop_id)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(settings.TIMEOUT_SECONDS, connect=settings.CONNECT_TIMEOUT_SECONDS),
                limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60),
            )
            self._clients[loop_id] = client
        return client

    async def aclose(self) -> None:
        for client in list(self._clients.values()):
            await client.aclose()
        self._clients.clear()

    # ── Candidate selection ─────────────────────────────────────────────
    def configured_models(self) -> List[str]:
        chain = [settings.DEFAULT_MODEL, settings.FAST_MODEL, *settings.LLM_FALLBACKS]
        out = []
        for spec in chain:
            prov, mod = resolve_model(spec)
            label = f"{prov}/{mod}"
            if PROVIDERS[prov].configured() and label not in out:
                out.append(label)
        return out

    async def candidates(self, model: Optional[str], purpose: str = "chat", local_only: bool = False,
                         fallbacks: bool = True) -> List[Tuple[str, str]]:
        """Providers to try, in order. ``fallbacks=False`` → only the requested model (if configured)."""
        primary = model or (settings.FAST_MODEL if purpose == "fast" else settings.DEFAULT_MODEL)
        chain: List[Tuple[str, str]] = []
        for spec in [primary, *(settings.LLM_FALLBACKS if fallbacks else [])]:
            pair = resolve_model(spec)
            if pair not in chain:
                chain.append(pair)
        chain = [p for p in chain if PROVIDERS[p[0]].configured()]
        if local_only:
            chain = [p for p in chain if PROVIDERS[p[0]].local] or [("ollama", settings.OLLAMA_MODEL)]
        elif connectivity.last_known() is False:
            chain.sort(key=lambda p: not PROVIDERS[p[0]].local)  # offline: try local first
        return chain

    # ── HTTP ────────────────────────────────────────────────────────────
    def _headers(self, provider: ProviderSpec, key: Optional[str] = None) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = key or provider.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
            if provider.name == "anthropic":
                headers["x-api-key"] = key
                headers["anthropic-version"] = "2023-06-01"
        if provider.name == "openrouter":
            headers["X-Title"] = "JARVIS"
        return headers

    def _payload(self, model: str, messages, tools, tool_choice, temperature, max_tokens, stream, provider: str = "",
                 response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": _sign_tool_calls(messages) if provider == "gemini" else _strip_extra_content(messages),
            "temperature": settings.TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or settings.MAX_TOKENS,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        effort = settings.REASONING_EFFORT.get(model)
        if effort and not (provider in PROVIDERS and PROVIDERS[provider].local):
            payload["reasoning_effort"] = effort
        if response_format:
            payload["response_format"] = response_format
        return payload

    async def _post(self, provider: ProviderSpec, payload: Dict[str, Any], timeout: Optional[float],
                    retries: Optional[int] = None, key: Optional[str] = None) -> Dict[str, Any]:
        """``key``: use exactly this key (a hedge request) and never rotate the shared key cursor."""
        url = f"{provider.base_url()}/chat/completions"
        max_retries = settings.MAX_RETRIES if retries is None else max(0, retries)
        self._last_request = time.monotonic()
        attempt = 0
        keys_left = 0 if key else len(provider.keys()) - 1
        while True:
            try:
                resp = await self._client().post(url, json=payload, headers=self._headers(provider, key),
                                                 timeout=timeout or settings.TIMEOUT_SECONDS)
                if keys_left > 0 and _key_exhausted(resp.status_code, resp.text):
                    provider.rotate_key()
                    keys_left -= 1
                    continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}", resp.status_code, resp.headers)
                if resp.status_code != 200:
                    raise ProviderError(f"HTTP {resp.status_code}: {resp.text[:300]}", resp.status_code, resp.headers)
                return resp.json()
            except (_RetryableError, httpx.TransportError) as exc:
                if isinstance(exc, httpx.ConnectError) and provider.local:
                    raise RuntimeError(f"{provider.name} is not running at {provider.base_url()}") from exc
                if attempt >= max_retries:
                    raise ProviderError(str(exc) or exc.__class__.__name__, getattr(exc, "status", None),
                                        getattr(exc, "headers", None), isinstance(exc, httpx.TimeoutException)) from exc
                delay = 0.4 * (2 ** attempt)
                retry_after = (getattr(exc, "headers", None) or {}).get("retry-after")
                if retry_after and str(retry_after).isdigit():
                    delay = min(float(retry_after), 8.0)
                attempt += 1
                await asyncio.sleep(delay)

    # ── Hedging (tail latency) ──────────────────────────────────────────
    @staticmethod
    def _hedge_plan(chain: List[Tuple[str, str]], hedge: bool) -> Optional[int]:
        """Chain index the one hedge for ``chain[0]`` goes to, or None (off / local / nowhere): 0 = the same provider
        on another key (it has two or more), else 1 = the next (cloud) provider of the chain. Never onto the intent
        model's provider when that has a single key: the hedge would spend the intent step's quota."""
        if not hedge or settings.LLM_HEDGE_AFTER_SECONDS <= 0 or not chain or PROVIDERS[chain[0][0]].local:
            return None
        if len(PROVIDERS[chain[0][0]].keys()) >= 2:
            return 0
        if len(chain) > 1 and not PROVIDERS[chain[1][0]].local:
            target = PROVIDERS[chain[1][0]]
            intent = (settings.INTENT_MODEL or "").strip()
            if intent and resolve_model(intent)[0] == target.name and len(target.keys()) < 2:
                return None
            return 1
        return None

    @staticmethod
    def _hedge_key(provider: ProviderSpec, same_provider: bool, start: int) -> Optional[str]:
        """The key a hedge request uses, chosen when the hedge starts (None: no usable key → no hedge).

        Another provider: its current key. The same provider: the key after the one the primary is on now, but
        never one the primary already went through since it started on key ``start`` (those were exhausted)."""
        if not same_provider:
            return provider.api_key()
        keys = provider.keys()
        if len(keys) < 2:
            return None
        current = _KEY_CURSOR.get(provider.name, 0) % len(keys)
        used = (current - start) % len(keys) + 1  # keys start..current
        return keys[(current + 1) % len(keys)] if used < len(keys) else None

    @staticmethod
    def _failures(exc: BaseException, primary: Tuple[str, str], hedge: Optional[Tuple[str, str]]):
        """(provider, model, error) per request of a failed attempt (two when a hedge ran and failed too)."""
        if not isinstance(exc, _HedgeFailed):
            return [(*primary, exc)]
        return [(*primary, exc.errors[0])] + ([(*hedge, exc.errors[1])] if hedge and len(exc.errors) > 1 else [])

    @staticmethod
    async def _race(primary: Awaitable[Any], start_hedge: Callable[[], Optional[Awaitable[Any]]],
                    after: float) -> Tuple[Any, bool]:
        """Await ``primary``; if it hasn't finished within ``after`` s, start ``start_hedge()`` too (None: no hedge
        possible) and take the first success. Returns ``(result, hedge_won)``; raises :class:`_HedgeFailed` when
        every started request failed (a primary that fails before ``after`` starts no hedge). The loser is
        cancelled and awaited."""
        tasks = [asyncio.ensure_future(primary)]
        try:
            done, pending = await asyncio.wait(tasks, timeout=after)
            if not done:
                second = start_hedge()
                if second is not None:
                    metrics.incr("llm.hedge.fired")
                    tasks.append(asyncio.ensure_future(second))
                pending = set(tasks)
            while True:
                for task in tasks:  # the primary first when both finished together
                    if task in done and task.exception() is None:
                        if task is not tasks[0]:
                            metrics.incr("llm.hedge.won")
                        return task.result(), task is not tasks[0]
                if not pending:
                    raise _HedgeFailed([t.exception() for t in tasks])
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def _race_streams(primary: AsyncIterator[Tuple[str, Any]],
                            start_hedge: Callable[[], Optional[AsyncIterator[Tuple[str, Any]]]],
                            after: float) -> AsyncIterator[Tuple[str, Any]]:
        """Stream ``primary``; if it has produced no event within ``after`` s, start ``start_hedge()`` too. The
        first stream to produce an event wins: the other is cancelled (never interleaved) and only the winner's
        events are passed on. Raises :class:`_HedgeFailed` when every started stream failed before any output."""
        queue: asyncio.Queue = asyncio.Queue()

        async def pump(tag: int, gen: AsyncIterator[Tuple[str, Any]]) -> None:
            try:
                async for item in gen:
                    await queue.put((tag, item, None))
                await queue.put((tag, None, None))
            except Exception as exc:  # handed to the consumer
                await queue.put((tag, None, exc))
            finally:
                await gen.aclose()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + after
        tasks = [asyncio.ensure_future(pump(0, primary))]
        errors: Dict[int, BaseException] = {}
        hedge_due = True
        try:
            while True:
                wait = max(0.0, deadline - loop.time()) if hedge_due else None
                try:
                    tag, event, exc = await asyncio.wait_for(queue.get(), wait)
                except asyncio.TimeoutError:
                    hedge_due = False
                    second = start_hedge()
                    if second is not None:
                        metrics.incr("llm.hedge.fired")
                        tasks.append(asyncio.ensure_future(pump(1, second)))
                    continue
                if event is not None:
                    break
                errors[tag] = exc or RuntimeError("stream ended without output")
                if len(tasks) == 1 or len(errors) == 2:
                    raise _HedgeFailed([errors[i] for i in sorted(errors)])
            winner = tag
            losers = [t for i, t in enumerate(tasks) if i != winner]
            for task in losers:
                task.cancel()
            await asyncio.gather(*losers, return_exceptions=True)
            if winner:
                metrics.incr("llm.hedge.won")
            while event is not None:
                yield event
                if event[0] == "done":
                    return
                tag, event, exc = await queue.get()
                while tag != winner:
                    tag, event, exc = await queue.get()
                if exc is not None:
                    raise exc
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    # ── Public API ──────────────────────────────────────────────────────
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        purpose: str = "chat",
        local_only: bool = False,
        timeout: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        fallbacks: bool = True,
        retries: Optional[int] = None,
        hedge: Optional[bool] = None,
    ) -> LLMResponse:
        """``fallbacks=False`` tries only ``model`` (callers that run their own fallback order, e.g. the intent step).

        ``retries`` overrides ``settings.MAX_RETRIES`` for 429 / 5xx / network errors (0 = fail fast, no backoff sleep).
        ``hedge``: allow one duplicate request after ``LLM_HEDGE_AFTER_SECONDS`` (default: on, except for
        ``fallbacks=False`` / ``purpose="intent"`` calls, which keep their own deadline).
        """
        errors: List[str] = []
        causes: List[BaseException] = []
        chain = await self.candidates(model, purpose, local_only, fallbacks)
        plan = self._hedge_plan(chain, (fallbacks and purpose != "intent") if hedge is None else hedge)
        tried = set()  # chain indexes a failed hedge already used
        for index, (prov_name, mod) in enumerate(chain):
            if index in tried:
                continue
            provider = PROVIDERS[prov_name]
            started = time.perf_counter()
            backup: Optional[Tuple[str, str]] = None
            try:
                payload = self._payload(mod, messages, tools, tool_choice, temperature, max_tokens, False, prov_name,
                                        response_format)
                if index == 0 and plan is not None:
                    h_index = plan
                    backup = chain[h_index]
                    h_provider = PROVIDERS[backup[0]]
                    start_key = _KEY_CURSOR.get(prov_name, 0) % max(1, len(PROVIDERS[prov_name].keys()))
                    h_payload = payload if h_index == index else self._payload(
                        backup[1], messages, tools, tool_choice, temperature, max_tokens, False, backup[0],
                        response_format)

                    def start_hedge():  # one attempt on a pinned key: no retry sleep, no key rotation
                        key = self._hedge_key(h_provider, h_index == index, start_key)
                        return self._post(h_provider, h_payload, timeout, 0, key=key) if key else None

                    data, hedge_won = await self._race(self._post(provider, payload, timeout, retries), start_hedge,
                                                       settings.LLM_HEDGE_AFTER_SECONDS)
                    if hedge_won and h_index != index:
                        index, (prov_name, mod) = h_index, backup
                else:
                    data = await self._post(provider, payload, timeout, retries)
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                elapsed = (time.perf_counter() - started) * 1000
                usage = data.get("usage") or {}
                metrics.observe(f"llm.{prov_name}", elapsed)
                metrics.incr(f"llm.calls.{prov_name}")
                return LLMResponse(
                    content=(message.get("content") or "").strip(),
                    provider=prov_name,
                    model=data.get("model") or mod,
                    tool_calls=_parse_tool_calls(message),
                    latency_ms=round(elapsed, 1),
                    usage={k: int(usage.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
                    finish_reason=choice.get("finish_reason"),
                    fallback_used=index > 0,
                )
            except Exception as exc:
                for p_name, p_mod, err in self._failures(exc, (prov_name, mod), backup):
                    metrics.incr(f"llm.errors.{p_name}")
                    errors.append(f"{p_name}/{p_mod}: {err}")
                    causes.append(err)
                    logger.warning(f"[LLM] {p_name}/{p_mod} failed: {err}")
                if isinstance(exc, _HedgeFailed) and len(exc.errors) > 1 and plan is not None and plan != index:
                    tried.add(plan)  # the hedge already tried that provider
        raise LLMUnavailable(errors, causes)

    async def stream_events(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        purpose: str = "chat",
        local_only: bool = False,
        timeout: Optional[float] = None,
        hedge: Optional[bool] = None,
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Yield ("text", delta) events, then a final ("done", LLMResponse).

        Tool-call deltas are accumulated and returned in the final response. A provider
        that fails before producing output is skipped in favour of the next candidate.
        ``hedge`` (default: on, except ``purpose="intent"``): when the first candidate has produced no output after
        ``LLM_HEDGE_AFTER_SECONDS``, one duplicate stream starts; the first to produce output is used.
        """
        errors: List[str] = []
        chain = await self.candidates(model, purpose, local_only)
        plan = self._hedge_plan(chain, purpose != "intent" if hedge is None else hedge)
        tried = set()  # chain indexes a failed hedge already used
        for index, (prov_name, mod) in enumerate(chain):
            if index in tried:
                continue
            backup: Optional[Tuple[str, str]] = None
            sent: List[str] = []  # text already passed on: once there is any, never restart elsewhere
            produced = False
            try:
                payload = self._payload(mod, messages, tools, None, temperature, max_tokens, True, prov_name)
                if index == 0 and plan is not None:
                    h_index = plan
                    backup = chain[h_index]
                    h_provider = PROVIDERS[backup[0]]
                    start_key = _KEY_CURSOR.get(prov_name, 0) % max(1, len(PROVIDERS[prov_name].keys()))
                    h_payload = payload if h_index == index else self._payload(
                        backup[1], messages, tools, None, temperature, max_tokens, True, backup[0])

                    def start_hedge():  # one attempt on a pinned key: no key rotation
                        key = self._hedge_key(h_provider, h_index == index, start_key)
                        if not key:
                            return None
                        return self._stream_one(backup[0], backup[1], h_payload, timeout, h_index > 0, key)

                    source = self._race_streams(self._stream_one(prov_name, mod, payload, timeout, False), start_hedge,
                                                settings.LLM_HEDGE_AFTER_SECONDS)
                else:
                    source = self._stream_one(prov_name, mod, payload, timeout, index > 0)
                try:
                    async for event in source:
                        produced = True
                        if event[0] == "text":
                            sent.append(event[1])
                        yield event
                finally:
                    await source.aclose()  # a consumer that stops early must not leave a request running
                return
            except Exception as exc:
                for p_name, p_mod, err in self._failures(exc, (prov_name, mod), backup):
                    metrics.incr(f"llm.errors.{p_name}")
                    errors.append(f"{p_name}/{p_mod}: {err}")
                    logger.warning(f"[LLM] stream {p_name}/{p_mod} failed: {err}")
                if produced:  # partial output already sent; don't restart on another provider
                    yield "done", LLMResponse("".join(sent).strip(), prov_name, mod, [], 0.0, {}, "error", index > 0)
                    return
                if isinstance(exc, _HedgeFailed) and len(exc.errors) > 1 and plan is not None and plan != index:
                    tried.add(plan)
        raise LLMUnavailable(errors)

    async def _stream_one(self, prov_name: str, mod: str, payload: Dict[str, Any], timeout: Optional[float],
                          fallback_used: bool, key: Optional[str] = None) -> AsyncIterator[Tuple[str, Any]]:
        """One provider's stream: ("text", delta)..., then ("done", LLMResponse). Raises when it fails before any
        output; a failure after output ends it with an ``"error"`` response (partial output is never restarted).
        ``key``: use exactly this key (a hedge stream), never rotating the shared key cursor."""
        provider = PROVIDERS[prov_name]
        keys_left = 0 if key else len(provider.keys()) - 1
        started = time.perf_counter()
        self._last_request = time.monotonic()
        produced = False
        text_parts: List[str] = []
        calls: Dict[int, Dict[str, Any]] = {}
        finish = None
        try:
            rotated = True
            while rotated:
                rotated = False
                async with self._client().stream(
                    "POST", f"{provider.base_url()}/chat/completions", json=payload,
                    headers=self._headers(provider, key), timeout=timeout or settings.TIMEOUT_SECONDS,
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")[:300]
                        if keys_left > 0 and _key_exhausted(resp.status_code, body):
                            provider.rotate_key()
                            keys_left -= 1
                            rotated = True
                            continue
                        raise RuntimeError(f"HTTP {resp.status_code}: {body}")
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        chunk_str = line[5:].strip()
                        if chunk_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(chunk_str)
                        except ValueError:
                            continue
                        choice = (chunk.get("choices") or [{}])[0]
                        delta = choice.get("delta") or {}
                        finish = choice.get("finish_reason") or finish
                        if delta.get("content"):
                            produced = True
                            text_parts.append(delta["content"])
                            yield "text", delta["content"]
                        for tc in delta.get("tool_calls") or []:
                            produced = True
                            slot = calls.setdefault(int(tc.get("index", len(calls))), {"id": "", "name": "", "args": "", "extra": None})
                            slot["id"] = tc.get("id") or slot["id"]
                            slot["extra"] = tc.get("extra_content") or slot["extra"]
                            fn = tc.get("function") or {}
                            slot["name"] = fn.get("name") or slot["name"]
                            slot["args"] += fn.get("arguments") or ""
        except Exception as exc:
            if not produced:
                raise
            metrics.incr(f"llm.errors.{prov_name}")
            logger.warning(f"[LLM] stream {prov_name}/{mod} failed: {exc}")
            yield "done", LLMResponse("".join(text_parts).strip(), prov_name, mod, [], 0.0, {}, "error", fallback_used)
            return
        elapsed = (time.perf_counter() - started) * 1000
        metrics.observe(f"llm.{prov_name}", elapsed)
        metrics.incr(f"llm.calls.{prov_name}")
        tool_calls = [
            ToolCall(c["id"] or f"call_{i}", c["name"], _parse_arguments(c["args"]), c["args"], c["extra"])
            for i, c in sorted(calls.items()) if c["name"]
        ]
        yield "done", LLMResponse("".join(text_parts).strip(), prov_name, mod, tool_calls, round(elapsed, 1), {},
                                  finish, fallback_used)

    async def stream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        """Plain text streaming convenience wrapper."""
        async for kind, value in self.stream_events(messages, **kwargs):
            if kind == "text":
                yield value

    # ── Keep-warm ───────────────────────────────────────────────────────
    async def warm_up(self) -> None:
        """``GET {base_url}/models`` on each configured cloud provider used by the default, action and intent models.

        Opens the TLS connection so the first real turn isn't cold, without spending completion quota (Groq's and
        Gemini's OpenAI-compatible APIs both list models at ``/models``). Never raises.
        """
        async def ping(spec: str) -> None:
            try:
                prov, _ = resolve_model(spec)
                provider = PROVIDERS[prov]
                if provider.local or not provider.configured():
                    return
                headers = {k: v for k, v in self._headers(provider).items() if k != "Content-Type"}
                # Single shot: no retries or key rotation, so an outage can't stall startup or every cycle.
                await asyncio.wait_for(
                    self._client().get(f"{provider.base_url()}/models", headers=headers, timeout=_WARM_TIMEOUT),
                    _WARM_TIMEOUT,
                )
            except Exception as exc:
                logger.debug(f"[LLM] warm-up {spec} failed: {exc}")

        specs: Dict[str, str] = {}  # one ping per provider
        for spec in (settings.DEFAULT_MODEL, settings.ACTION_MODEL, settings.INTENT_MODEL):
            if spec and spec.strip():
                try:
                    specs.setdefault(resolve_model(spec)[0], spec)
                except Exception:
                    pass
        await asyncio.gather(*(ping(spec) for spec in specs.values()))

    def _keepwarm_due(self, interval: float) -> bool:
        """Idle for ``interval`` but within KEEPWARM_WINDOW_SECONDS of the last real request; online; cloud mode."""
        if self._last_request is None or connectivity.last_known() is False or settings.PRIVACY_MODE == "local":
            return False
        return interval <= time.monotonic() - self._last_request <= KEEPWARM_WINDOW_SECONDS

    def start_keepwarm(self, interval: float = 50.0) -> None:
        """Re-run warm_up in the background while idle (no-op when LLM_KEEPWARM is off)."""
        if not settings.LLM_KEEPWARM or (self._keepwarm_task and not self._keepwarm_task.done()):
            return

        async def loop() -> None:
            while True:
                await asyncio.sleep(interval)
                if self._keepwarm_due(interval):
                    await self.warm_up()

        self._keepwarm_task = asyncio.get_running_loop().create_task(loop())

    def stop_keepwarm(self) -> Optional[asyncio.Task]:
        """Cancel the keep-warm loop; returns the cancelled task (await it to let it finish) or None."""
        task, self._keepwarm_task = self._keepwarm_task, None
        if task:
            task.cancel()
        return task

    async def health(self) -> Dict[str, Any]:
        return {
            "configured_models": self.configured_models(),
            "default_model": "/".join(resolve_model(settings.DEFAULT_MODEL)),
            "online": connectivity.last_known(),
        }


llm = LLMGateway()
