"""One LLM gateway for every provider (OpenAI-compatible Chat Completions).

All eight providers expose an OpenAI-compatible ``/chat/completions`` endpoint, so a
single client covers them: Groq, Gemini, OpenAI, Anthropic, OpenRouter, DeepSeek,
Ollama and LM Studio. Features:

* native tool calling (``tools`` / ``tool_calls``) and streaming (text + tool calls);
* per-call timeouts, retries with exponential backoff on 429 / 5xx / network errors;
* a fallback chain from config (``DEFAULT_MODEL`` → ``LLM_FALLBACKS``);
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
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Tuple

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

    def api_key(self) -> Optional[str]:
        return None if self.local else settings.api_key_for(self.name)

    def configured(self) -> bool:
        return self.local or bool(self.api_key())


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

    def to_message(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.raw_arguments or json.dumps(self.arguments)},
        }


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


class LLMUnavailable(RuntimeError):
    """Raised when every candidate provider failed."""

    def __init__(self, errors: List[str]) -> None:
        super().__init__("; ".join(errors) or "No LLM provider is configured.")
        self.errors = errors


class _RetryableError(RuntimeError):
    pass


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


def _parse_tool_calls(message: Dict[str, Any]) -> List[ToolCall]:
    calls = []
    for i, tc in enumerate(message.get("tool_calls") or []):
        fn = tc.get("function") or {}
        raw = fn.get("arguments") or ""
        calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), _parse_arguments(raw), raw if isinstance(raw, str) else json.dumps(raw)))
    return calls


class LLMGateway:
    def __init__(self) -> None:
        self._clients: Dict[int, httpx.AsyncClient] = {}
        self._transport: Optional[httpx.AsyncBaseTransport] = None

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

    async def candidates(self, model: Optional[str], purpose: str = "chat", local_only: bool = False) -> List[Tuple[str, str]]:
        primary = model or (settings.FAST_MODEL if purpose == "fast" else settings.DEFAULT_MODEL)
        chain: List[Tuple[str, str]] = []
        for spec in [primary, *settings.LLM_FALLBACKS]:
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
    def _headers(self, provider: ProviderSpec) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = provider.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
            if provider.name == "anthropic":
                headers["x-api-key"] = key
                headers["anthropic-version"] = "2023-06-01"
        if provider.name == "openrouter":
            headers["X-Title"] = "JARVIS"
        return headers

    def _payload(self, model: str, messages, tools, tool_choice, temperature, max_tokens, stream) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": settings.TEMPERATURE if temperature is None else temperature,
            "max_tokens": max_tokens or settings.MAX_TOKENS,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        return payload

    async def _post(self, provider: ProviderSpec, payload: Dict[str, Any], timeout: Optional[float]) -> Dict[str, Any]:
        url = f"{provider.base_url()}/chat/completions"
        attempt = 0
        while True:
            try:
                resp = await self._client().post(url, json=payload, headers=self._headers(provider), timeout=timeout or settings.TIMEOUT_SECONDS)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise _RetryableError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                return resp.json()
            except (_RetryableError, httpx.TransportError) as exc:
                if isinstance(exc, httpx.ConnectError) and provider.local:
                    raise RuntimeError(f"{provider.name} is not running at {provider.base_url()}") from exc
                if attempt >= settings.MAX_RETRIES:
                    raise RuntimeError(str(exc) or exc.__class__.__name__) from exc
                delay = 0.4 * (2 ** attempt)
                retry_after = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after") if hasattr(exc, "response") else None
                if retry_after and str(retry_after).isdigit():
                    delay = min(float(retry_after), 8.0)
                attempt += 1
                await asyncio.sleep(delay)

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
    ) -> LLMResponse:
        errors: List[str] = []
        chain = await self.candidates(model, purpose, local_only)
        for index, (prov_name, mod) in enumerate(chain):
            provider = PROVIDERS[prov_name]
            started = time.perf_counter()
            try:
                data = await self._post(provider, self._payload(mod, messages, tools, tool_choice, temperature, max_tokens, False), timeout)
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
                metrics.incr(f"llm.errors.{prov_name}")
                errors.append(f"{prov_name}/{mod}: {exc}")
                logger.warning(f"[LLM] {prov_name}/{mod} failed: {exc}")
        raise LLMUnavailable(errors)

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
    ) -> AsyncIterator[Tuple[str, Any]]:
        """Yield ("text", delta) events, then a final ("done", LLMResponse).

        Tool-call deltas are accumulated and returned in the final response. A provider
        that fails before producing output is skipped in favour of the next candidate.
        """
        errors: List[str] = []
        chain = await self.candidates(model, purpose, local_only)
        for index, (prov_name, mod) in enumerate(chain):
            provider = PROVIDERS[prov_name]
            started = time.perf_counter()
            produced = False
            text_parts: List[str] = []
            calls: Dict[int, Dict[str, Any]] = {}
            finish = None
            try:
                payload = self._payload(mod, messages, tools, None, temperature, max_tokens, True)
                async with self._client().stream(
                    "POST", f"{provider.base_url()}/chat/completions", json=payload,
                    headers=self._headers(provider), timeout=timeout or settings.TIMEOUT_SECONDS,
                ) as resp:
                    if resp.status_code != 200:
                        body = (await resp.aread()).decode("utf-8", "replace")[:300]
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
                            slot = calls.setdefault(int(tc.get("index", len(calls))), {"id": "", "name": "", "args": ""})
                            slot["id"] = tc.get("id") or slot["id"]
                            fn = tc.get("function") or {}
                            slot["name"] = fn.get("name") or slot["name"]
                            slot["args"] += fn.get("arguments") or ""
                elapsed = (time.perf_counter() - started) * 1000
                metrics.observe(f"llm.{prov_name}", elapsed)
                metrics.incr(f"llm.calls.{prov_name}")
                tool_calls = [
                    ToolCall(c["id"] or f"call_{i}", c["name"], _parse_arguments(c["args"]), c["args"])
                    for i, c in sorted(calls.items()) if c["name"]
                ]
                yield "done", LLMResponse("".join(text_parts).strip(), prov_name, mod, tool_calls, round(elapsed, 1), {}, finish, index > 0)
                return
            except Exception as exc:
                metrics.incr(f"llm.errors.{prov_name}")
                errors.append(f"{prov_name}/{mod}: {exc}")
                logger.warning(f"[LLM] stream {prov_name}/{mod} failed: {exc}")
                if produced:  # partial output already sent; don't restart on another provider
                    yield "done", LLMResponse("".join(text_parts).strip(), prov_name, mod, [], 0.0, {}, "error", index > 0)
                    return
        raise LLMUnavailable(errors)

    async def stream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        """Plain text streaming convenience wrapper."""
        async for kind, value in self.stream_events(messages, **kwargs):
            if kind == "text":
                yield value

    async def health(self) -> Dict[str, Any]:
        return {
            "configured_models": self.configured_models(),
            "default_model": "/".join(resolve_model(settings.DEFAULT_MODEL)),
            "online": connectivity.last_known(),
        }


llm = LLMGateway()
