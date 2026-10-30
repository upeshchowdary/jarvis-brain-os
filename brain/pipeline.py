"""The JARVIS brain pipeline: session → (confirmation | fast path | agent) → respond.

1. **Pending confirmation** — if the session has an action waiting, "yes" runs it
   and anything else cancels it (and is then handled as a new request).
2. **Fast path** — anchored command grammar → tool, no LLM (~ms latency).
3. **Agent** — LLM with native tool calling (≤ AGENT_MAX_STEPS rounds).
   * Questions/statements are offered read-only tools only.
   * After a tool returns untrusted content (web/file/screen), side-effect tools
     are withdrawn for the rest of the turn (prompt-injection taint).
   * PRIVACY_MODE=local: once file/screen content is in context, only local models are used.
   * A tool that needs confirmation ends the turn with the preview.
   * Action-only rounds skip the second LLM call (lower latency).

Every side effect happens inside tools → ActionGateway (risk policy, dry-run, audit).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from automation.gateway import gateway
from brain import router
from brain.conversation_manager import conversation_manager
from brain.prompts import build_system_prompt, wrap_untrusted
from core.config import settings
from core.events import bus
from core.llm import LLMResponse, LLMUnavailable, llm, resolve_model
from core.logging import logger
from core.metrics import metrics
from core.session import sessions
from tools.base import Tool, ToolCallRecord, ToolContext, ToolResult
from tools.registry import load_builtin_tools, registry
from tools.selection import select_tools


@dataclass
class TurnResult:
    query: str
    response: str
    session_id: str
    route: str = "agent"  # confirmation | fast_path | agent | error
    intent: str = "conversation"  # tool name or "conversation"
    status: str = "ok"  # ok | needs_confirmation | dry_run | error
    pending: Optional[Dict[str, Any]] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "response": self.response,
            "session_id": self.session_id,
            "route": self.route,
            "status": self.status,
            "pending": self.pending,
            "tools": self.tools,
            # Backwards-compatible shape used by chat.py / older clients.
            "intent": {"intent": self.intent, "confidence": 1.0 if self.route == "fast_path" else 0.9, "summary": self.route},
            "metadata": self.metadata,
        }


class Brain:
    def __init__(self) -> None:
        load_builtin_tools()

    # ── Public API ──────────────────────────────────────────────────────
    async def handle(self, query: str, session_id: Optional[str] = None, **kwargs: Any) -> TurnResult:
        result: Optional[TurnResult] = None
        async for event in self.stream(query, session_id, stream_tokens=False, **kwargs):
            if event["type"] == "done":
                result = event["result"]
        assert result is not None
        return result

    async def stream(
        self,
        query: str,
        session_id: Optional[str] = None,
        *,
        channel: str = "text",
        personality: Optional[str] = None,
        model_override: Optional[str] = None,
        dry_run: Optional[bool] = None,
        stream_tokens: bool = True,
        allow_confirm: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Yield events: {"type": "status"|"text"|"tool"|"done", ...}. The last one carries the TurnResult."""
        started = time.perf_counter()
        timing: Dict[str, float] = {}
        query = (query or "").strip()
        sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
        state = sessions.get(sid)
        await conversation_manager.get_or_create_session_async(sid)
        history = conversation_manager.get_history(sid)
        conversation_manager.add_user_message(sid, query)
        ctx = ToolContext(session_id=sid, channel=channel, dry_run=dry_run, source=channel)
        note = ""

        def finish(response: str, **kw: Any) -> Dict[str, Any]:
            total = (time.perf_counter() - started) * 1000
            metrics.observe(f"turn.{kw.get('route', 'agent')}", total)
            result = TurnResult(query=query, response=response, session_id=sid, **kw)
            result.metadata.setdefault("provider", "jarvis")
            result.metadata.setdefault("model", "fast-path" if result.route != "agent" else "")
            result.metadata.update(session_id=sid, total_latency_ms=round(total, 1), timing_breakdown_ms=timing)
            conversation_manager.add_assistant_message(sid, response)
            bus.publish("turn", {"session_id": sid, "channel": channel, "query": query, "response": response[:2000],
                                 "status": result.status, "route": result.route})
            self._after_turn(result)
            return {"type": "done", "result": result}

        if not query:
            yield finish("I didn't catch that.", route="error", status="error")
            return

        # 1 ── Pending confirmation ───────────────────────────────────────
        # Remote agents (MCP) may never confirm their own actions: the human approves them elsewhere.
        pending = gateway.pending_for(sid) if allow_confirm else None
        if pending is not None:
            if router.is_affirmative(query):
                res = await gateway.confirm(pending.pending_id, sid)
                yield finish(res.message, route="confirmation", intent="confirm",
                             status="ok" if res.success else "error", tools=[{"name": "confirm", "steps": res.steps}])
                return
            gateway.cancel(pending.pending_id, "user said something else")
            if router.is_negative(query):
                yield finish("Okay, cancelled — nothing was done.", route="confirmation", intent="cancel")
                return
            note = "(I cancelled the action that was waiting for confirmation.)\n\n"

        # 2 ── Fast path ──────────────────────────────────────────────────
        t = time.perf_counter()
        decision = router.fast_route(query)
        timing["router_ms"] = round((time.perf_counter() - t) * 1000, 2)
        tool = registry.get(decision.tool) if decision is not None else None
        if decision is not None and tool is not None:
            record = await self._run_tool(tool, decision.args, ctx, origin="fast")
            timing["tool_ms"] = round(record.elapsed_ms, 1)
            yield {"type": "tool", "name": tool.name, "result": record.to_dict()}
            if tool.final or not record.result.success or record.result.status != "ok":
                yield finish(note + record.result.message, route="fast_path", intent=tool.name, status=record.result.status,
                             pending=self._pending_dict(record.result), tools=[record.to_dict()],
                             metadata={"rule": decision.rule})
                return
            prefilled = (tool, record)  # e.g. "summarize today's commits": the agent writes the answer from the result
        else:
            prefilled = None

        # 3 ── Agent ──────────────────────────────────────────────────────
        async for event in self._agent(query, history, state, ctx, personality, model_override, timing, stream_tokens, note, prefilled):
            if event["type"] == "final":
                yield finish(**event["kwargs"])
                return
            yield event

    # ── Agent loop ──────────────────────────────────────────────────────
    async def _agent(self, query, history, state, ctx, personality, model_override, timing, stream_tokens, note,
                     prefilled=None) -> AsyncIterator[Dict[str, Any]]:
        allow_actions = router.is_imperative(query)
        base = registry.all() if allow_actions else registry.read_only()
        recent = " ".join(m.get("content", "")[:200] for m in history[-4:])
        offered: Dict[str, Tool] = {t.name: t for t in select_tools(query, base, recent)}

        t = time.perf_counter()
        facts, memories = await self._memory_context(query, allow_actions)
        timing["memory_ms"] = round((time.perf_counter() - t) * 1000, 1)

        model = model_override or state.model_override
        model_label = "/".join(resolve_model(model or settings.DEFAULT_MODEL))
        system = build_system_prompt(personality=personality or state.personality, channel=ctx.channel,
                                     facts=facts, memories=memories, model_label=model_label)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        messages += [m for m in history[-settings.MAX_HISTORY_MESSAGES:] if m.get("role") in ("user", "assistant")]
        messages.append({"role": "user", "content": query})

        tainted = False
        sensitive = False
        records: List[ToolCallRecord] = []
        if prefilled is not None:
            ptool, precord = prefilled
            call_id = "fast_0"
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": ptool.name, "arguments": json.dumps(precord.arguments)}}]})
            content = precord.result.message
            if precord.result.content_for_llm:
                content += "\n" + (wrap_untrusted(ptool.name, precord.result.content_for_llm) if ptool.untrusted_output else precord.result.content_for_llm)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": content})
            records.append(precord)
            tainted, sensitive = ptool.untrusted_output, ptool.sensitive_output
        last: Optional[LLMResponse] = None
        llm_ms = 0.0

        for step in range(settings.AGENT_MAX_STEPS):
            tools_now = [tl for tl in offered.values() if not (tainted and tl.side_effect)]
            local_only = settings.PRIVACY_MODE == "local" and sensitive
            t = time.perf_counter()
            resp: Optional[LLMResponse] = None
            try:
                async for kind, value in self._llm_events(messages, tools_now, model, local_only, stream_tokens):
                    if kind == "text":
                        yield {"type": "text", "delta": value}
                    else:
                        resp = value
            except LLMUnavailable as exc:
                logger.error(f"[Brain] No LLM available: {exc}")
                yield {"type": "final", "kwargs": dict(
                    response=note + "I can't reach a language model right now, so I can't answer that. "
                             "Start Ollama or add an API key to .env. Direct commands like “volume up” or “open chrome” still work.",
                    route="error", intent="conversation", status="error", tools=[r.to_dict() for r in records],
                    metadata={"error": str(exc)})}
                return
            llm_ms += (time.perf_counter() - t) * 1000
            assert resp is not None
            last = resp

            if not resp.tool_calls:
                break

            messages.append(resp.assistant_message())
            round_records: List[ToolCallRecord] = []
            for call in resp.tool_calls:
                tool = offered.get(call.name)
                if tool is None or (tainted and tool.side_effect):
                    reason = "not available for this request" if tool is None else "blocked: untrusted content is in context"
                    logger.warning(f"[Brain] Refused tool call {call.name}: {reason}")
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": f"Error: tool '{call.name}' is {reason}."})
                    continue
                record = await self._run_tool(tool, call.arguments, ctx, origin="llm")
                records.append(record)
                round_records.append(record)
                yield {"type": "tool", "name": tool.name, "result": record.to_dict()}
                content = record.result.message
                if record.result.content_for_llm:
                    content += "\n" + (wrap_untrusted(tool.name, record.result.content_for_llm) if tool.untrusted_output else record.result.content_for_llm)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content or ("ok" if record.result.success else "failed")})
                tainted = tainted or tool.untrusted_output
                sensitive = sensitive or tool.sensitive_output

                if record.result.status == "needs_confirmation":
                    timing["llm_ms"] = round(llm_ms, 1)
                    yield {"type": "final", "kwargs": dict(
                        response=note + record.result.message, route="agent", intent=tool.name, status="needs_confirmation",
                        pending=self._pending_dict(record.result), tools=[r.to_dict() for r in records],
                        metadata=self._llm_meta(resp, model_label))}
                    return

            # Latency: if every call this round was a self-describing action, answer without another LLM round.
            if round_records and all(registry.get(r.name).final for r in round_records):
                text = "\n".join(r.result.message for r in round_records if r.result.message)
                lead = (resp.content + "\n\n") if resp.content else ""
                timing["llm_ms"] = round(llm_ms, 1)
                status = "dry_run" if any(r.result.status == "dry_run" for r in round_records) else (
                    "ok" if all(r.result.success for r in round_records) else "error")
                yield {"type": "final", "kwargs": dict(
                    response=note + lead + text, route="agent", intent=round_records[-1].name, status=status,
                    tools=[r.to_dict() for r in records], metadata=self._llm_meta(resp, model_label))}
                return
        else:
            logger.warning("[Brain] Agent hit max steps")

        timing["llm_ms"] = round(llm_ms, 1)
        answer = (last.content if last else "") or "Done."
        yield {"type": "final", "kwargs": dict(
            response=note + answer, route="agent", intent=records[-1].name if records else "conversation",
            status="ok", tools=[r.to_dict() for r in records], metadata=self._llm_meta(last, model_label))}

    async def _llm_events(self, messages, tools: List[Tool], model, local_only: bool, stream_tokens: bool):
        schemas = [t.schema() for t in tools] or None
        if stream_tokens:
            async for kind, value in llm.stream_events(messages, model=model, tools=schemas, local_only=local_only):
                yield kind, value
        else:
            resp = await llm.chat(messages, model=model, tools=schemas, local_only=local_only)
            yield "done", resp

    # ── Helpers ─────────────────────────────────────────────────────────
    async def _run_tool(self, tool: Tool, args: Dict[str, Any], ctx: ToolContext, origin: str) -> ToolCallRecord:
        started = time.perf_counter()
        clean = {k: v for k, v in (args or {}).items() if v is not None}
        try:
            result = await tool.run(ctx, **clean)
        except TypeError as exc:
            result = ToolResult.error(f"Bad arguments for {tool.name}: {exc}")
        except Exception as exc:
            logger.exception(f"[Brain] tool {tool.name} crashed")
            result = ToolResult.error(f"{tool.name} failed: {exc}")
        elapsed = (time.perf_counter() - started) * 1000
        metrics.observe(f"tool.{tool.name}", elapsed)
        sessions.get(ctx.session_id).last_tool = tool.name
        return ToolCallRecord(tool.name, clean, result, elapsed, origin)

    @staticmethod
    def _pending_dict(result: ToolResult) -> Optional[Dict[str, Any]]:
        if result.status != "needs_confirmation" or not result.pending_id:
            return None
        pending = gateway.get_pending(result.pending_id)
        return pending.to_dict() if pending else {"pending_id": result.pending_id}

    @staticmethod
    def _llm_meta(resp: Optional[LLMResponse], requested: str) -> Dict[str, Any]:
        if resp is None:
            return {"requested_model": requested}
        return {
            "provider": resp.provider,
            "model": resp.model,
            "requested_model": requested,
            "fallback_used": resp.fallback_used,
            "model_latency_ms": resp.latency_ms,
            "usage": resp.usage,
        }

    async def _memory_context(self, query: str, is_command: bool):
        from memory.long_term import long_term_memory

        try:
            facts = await long_term_memory.relevant_facts(query, settings.MAX_FACTS_PER_TURN)
        except Exception as exc:
            logger.warning(f"[Brain] memory unavailable: {exc}")
            facts = {}
        memories: List[str] = []
        if not is_command and len(query) > 15:
            try:
                from memory.vector_memory import vector_memory

                for doc in vector_memory.search_similar(query, top_k=3):
                    if doc.score >= settings.MEMORY_RECALL_MIN_SCORE:
                        memories.append(doc.content[:300])
            except Exception:
                pass
        return facts, memories

    def _after_turn(self, result: TurnResult) -> None:
        """Telemetry + memory, off the critical path."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._persist(result))

    async def _persist(self, result: TurnResult) -> None:
        try:
            from app.database.connection import db_manager
            from app.database.models import ChatLogRecord
            from memory.long_term import long_term_memory

            meta = result.metadata
            usage = meta.get("usage") or {}
            await db_manager.log_chat_telemetry(ChatLogRecord(
                session_id=result.session_id, user_query=result.query, intent=result.intent,
                confidence=1.0 if result.route == "fast_path" else 0.9, provider=meta.get("provider", "jarvis"),
                model=meta.get("model") or "fast-path", latency_ms=meta.get("total_latency_ms", 0.0),
                prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0), response_text=result.response,
            ))
            if result.route == "agent" and result.intent == "conversation":
                await long_term_memory.auto_extract_facts(result.query)
                if settings.ROUTER_USE_LLM:
                    await long_term_memory.llm_extract_facts(result.query)
                if len(result.query) > 25:
                    from memory.vector_memory import vector_memory

                    await vector_memory.add_document_async(
                        doc_id=f"chat_{result.session_id}_{uuid.uuid4().hex[:6]}",
                        content=f"User asked: {result.query} | JARVIS answered: {result.response[:300]}",
                        metadata={"session_id": result.session_id},
                    )
        except Exception as exc:  # never let bookkeeping break a turn
            logger.debug(f"[Brain] post-turn persistence skipped: {exc}")


brain = Brain()

__all__ = ["Brain", "TurnResult", "brain", "json"]
