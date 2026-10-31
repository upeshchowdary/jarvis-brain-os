"""The JARVIS brain pipeline: session → (confirmation | fast path | agent) → respond.

1. **Pending confirmation** — if the session has an action waiting, "yes" runs it
   and anything else cancels it (and is then handled as a new request).
2. **Fast path** — anchored command grammar → tool, no LLM (~ms latency). While a task is active a bare
   context word ("next", "again", …) is not a command; safety rules (stop, resume, dry run) always win.
3. **Understand** — the intent step (rewrite to a standalone request + classify) and the memory lookup run
   concurrently. A rewritten request that is an exact command runs on the fast path.
4. **Current task** — a new action starts a task; a continuation revives the last one; questions leave it alone.
5. **Agent** — LLM with native tool calling (≤ AGENT_MAX_STEPS rounds, AGENT_MAX_STEPS_ACTION for actions).
   * Questions/statements are offered read-only tools only.
   * After a tool returns untrusted content (web/file/screen), side-effect tools
     are withdrawn for the rest of the turn (prompt-injection taint).
   * PRIVACY_MODE=local: once file/screen content is in context, only local models are used.
   * A tool that needs confirmation ends the turn with the preview.
   * Action-only rounds skip the second LLM call only for single-step requests; multi-step tasks run to
     completion and emit short ``progress`` events.
   * Assistant history turns end with ``[did: tool(arg) ✓/✗, …]`` action notes (not shown to the user).

Every side effect happens inside tools → ActionGateway (risk policy, dry-run, audit).
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from automation.gateway import gateway
from brain import router
from brain.conversation_manager import conversation_manager
from brain.intent import Intent, analyze, legacy_intent
from brain.prompts import build_system_prompt, wrap_untrusted
from brain.task_state import STEP_MARKS, recent_turns, tasks
from core.config import settings
from core.events import bus
from core.llm import LLMResponse, LLMUnavailable, llm, resolve_model
from core.logging import logger
from core.metrics import metrics
from core.session import sessions
from tools.base import Tool, ToolCallRecord, ToolContext, ToolResult, from_gateway
from tools.registry import load_builtin_tools, registry
from tools.selection import select_tools

_NOTES_LIMIT = 200  # max chars of the " [did: …]" suffix on assistant history turns
_NOTE_ARG_CLIP = 40
_PROGRESS_CLIP = 60
_BOOKKEEPING_TOOLS = {"task_plan", "clarify"}  # not task steps, not in action notes / progress
_STOPPED = "Stopped."
_STOPPED_BEFORE = "Emergency stop is on, so I won't act. Say “resume automation” to allow actions again."
_SENTENCE = re.compile(r"(.+?[.!?])(?:\s|$)")


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _step_state(record: ToolCallRecord) -> str:
    """How a tool call ended, as a key of STEP_MARKS: ok | failed | pending | dry_run | cancelled."""
    status = record.result.status
    if status == "needs_confirmation":
        return "pending"
    if status in ("dry_run", "cancelled"):
        return status
    return "ok" if record.result.success and status == "ok" else "failed"


def _done_ok(record: ToolCallRecord) -> bool:
    return _step_state(record) == "ok"


def _note_arg(args: Dict[str, Any]) -> str:
    value = next(iter(args.values()), "") if args else ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    return _clip(str(value).replace("[", "(").replace("]", ")"), _NOTE_ARG_CLIP)


def _action_notes(records: List[ToolCallRecord], limit: int = _NOTES_LIMIT) -> str:
    """`` [did: open_app(whatsapp) ✓, send_message(whatsapp) ✗]`` — one trailing segment, ≤ ``limit`` chars.

    Marks (STEP_MARKS): ✓ done, ✗ failed / blocked, … awaiting confirmation, (dry run), (cancelled).
    When too long, the oldest items are dropped first (replaced by "…").
    """
    items = [f"{r.name}({_note_arg(r.arguments)}) {STEP_MARKS[_step_state(r)]}"
             for r in records if r.name not in _BOOKKEEPING_TOOLS]
    if not items:
        return ""
    dropped = False
    while True:
        note = " [did: " + ", ".join((["…"] if dropped else []) + items) + "]"
        if len(note) <= limit or len(items) == 1:
            break
        items.pop(0)
        dropped = True
    return note if len(note) <= limit else note[: limit - 2] + "…]"


def _progress_text(message: str) -> str:
    """First sentence of a tool result, ≤ 60 chars (shown while a multi-step task runs)."""
    text = " ".join((message or "").split())
    m = _SENTENCE.match(text)
    return _clip(m.group(1) if m else text, _PROGRESS_CLIP)


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
        self._persisting: set = set()  # post-turn log + index tasks (kept referenced until done)
        self._extracting: set = set()  # post-turn fact-extraction tasks
        # session → (tool, args, task id or None) of the action waiting for confirmation there
        self._awaiting: Dict[str, Tuple[str, Dict[str, Any], Optional[str]]] = {}

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
        carried: List[ToolCallRecord] = []  # e.g. the confirmation this message cancelled

        def finish(response: str, records: Optional[List[ToolCallRecord]] = None, **kw: Any) -> Dict[str, Any]:
            total = (time.perf_counter() - started) * 1000
            metrics.observe(f"turn.{kw.get('route', 'agent')}", total)
            result = TurnResult(query=query, response=response, session_id=sid, **kw)
            result.metadata.setdefault("provider", "jarvis")
            result.metadata.setdefault("model", "fast-path" if result.route != "agent" else "")
            result.metadata.update(session_id=sid, total_latency_ms=round(total, 1), timing_breakdown_ms=timing)
            result.metadata["channel"] = channel
            # History (and the intent step) see what was done; the user-visible response stays clean.
            remembered = response + _action_notes(carried + (records or []))
            conversation_manager.add_assistant_message(sid, remembered)
            recent_turns.add(channel, sid, query, remembered)
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
            awaiting = self._awaiting.pop(sid, None)
            if router.is_affirmative(query):
                res = await gateway.confirm(pending.pending_id, sid)
                record = self._confirmation_record(pending, awaiting, from_gateway(res))
                if self._task_owns(awaiting):
                    self._record_step(record)
                    self._close_task("continued", [record])
                yield finish(res.message, records=[record], route="confirmation", intent="confirm",
                             status="ok" if res.success else "error", tools=[{"name": "confirm", "steps": res.steps}])
                return
            gateway.cancel(pending.pending_id, "user said something else")
            record = self._confirmation_record(pending, awaiting, ToolResult(
                False, "Cancelled by the user; nothing was done.", status="cancelled"))
            if self._task_owns(awaiting):
                self._record_step(record)
            if router.is_negative(query):
                yield finish("Okay, cancelled — nothing was done.", records=[record], route="confirmation",
                             intent="cancel")
                return
            carried.append(record)
            note = "(I cancelled the action that was waiting for confirmation.)\n\n"

        # 2 ── Fast path ──────────────────────────────────────────────────
        active = tasks.current()
        t = time.perf_counter()
        decision = router.fast_route(query, task_active=active is not None)
        timing["router_ms"] = round((time.perf_counter() - t) * 1000, 2)
        prefilled = None
        hit = self._fast_hit(decision)
        if hit is not None:
            tool, args = hit
            record = await self._run_tool(tool, args, ctx, origin="fast")
            timing["tool_ms"] = round(record.elapsed_ms, 1)
            yield {"type": "tool", "name": tool.name, "result": record.to_dict()}
            linked = active is not None and tool.side_effect
            if linked:
                self._record_step(record)
            self._remember_pending(sid, record, linked)
            if active is not None and decision.rule == "stop" and gateway.is_stopped:
                tasks.set_status("shelved")  # emergency stop ends the task the same way everywhere
            if self._fast_path_done(tool, record):
                yield finish(note + record.result.message, records=[record], route="fast_path", intent=tool.name,
                             status=record.result.status, pending=self._pending_dict(record.result),
                             tools=[record.to_dict()], metadata={"rule": decision.rule})
                return
            prefilled = (tool, record)  # e.g. "summarize today's commits": the agent writes the answer from the result

        # 3 ── Understand: intent step ∥ memory lookup ────────────────────
        if prefilled is None:
            turns = recent_turns.recent(settings.INTENT_HISTORY_TURNS, settings.INTENT_HISTORY_WINDOW_SECONDS)
            intent, (facts, memories) = await asyncio.gather(
                self._timed(timing, "intent_ms", analyze(query, turns, tasks.current())),
                self._timed(timing, "memory_ms", self._memory_context(query, router.is_imperative(query), sid)),
            )
        else:  # the fast path already understood it; source "fast": this turn never starts/continues/closes a task
            intent = replace(legacy_intent(query), source="fast")
            facts, memories = await self._timed(timing, "memory_ms",
                                                self._memory_context(query, intent.kind == "action", sid))

        # 4 ── Current task (questions and chat leave it alone) ───────────
        task_mode = self._begin_task(intent, ctx)
        book = task_mode is not None  # this turn's tool calls are steps of the current task

        # 5 ── The rewritten request may be an exact command ("yes do that" → "open whatsapp") ──
        if prefilled is None and intent.request.strip() != query:
            decision = router.fast_route(intent.request, task_active=False)
            hit = self._fast_hit(decision)
            if hit is not None and hit[0].side_effect and intent.kind != "action":
                hit = None  # questions never act, even if their rewrite reads like a command
            if hit is not None:
                tool, args = hit
                record = await self._run_tool(tool, args, ctx, origin="fast")
                yield {"type": "tool", "name": tool.name, "result": record.to_dict()}
                if book:
                    self._record_step(record)
                self._remember_pending(sid, record, book)
                if self._fast_path_done(tool, record):
                    self._close_task(task_mode, [record])
                    yield finish(note + record.result.message, records=[record], route="fast_path", intent=tool.name,
                                 status=record.result.status, pending=self._pending_dict(record.result),
                                 tools=[record.to_dict()], metadata={"rule": decision.rule, "request": intent.request})
                    return
                prefilled = (tool, record)

        # 6 ── Agent ──────────────────────────────────────────────────────
        async for event in self._agent(query, intent, facts, memories, history, state, ctx, personality,
                                       model_override, timing, stream_tokens, note, prefilled, book):
            if event["type"] == "final":
                records = event.get("records") or []
                self._close_task(task_mode, records, event.get("outcome"))
                yield finish(records=records, **event["kwargs"])
                return
            yield event

    # ── Fast path / task helpers ────────────────────────────────────────
    @staticmethod
    def _fast_hit(decision: Optional[router.RouteDecision]):
        tool = registry.get(decision.tool) if decision is not None else None
        return (tool, decision.args) if tool is not None else None

    @staticmethod
    def _fast_path_done(tool: Tool, record: ToolCallRecord) -> bool:
        return tool.final or not record.result.success or record.result.status != "ok"

    @staticmethod
    async def _timed(timing: Dict[str, float], key: str, coro):
        t = time.perf_counter()
        try:
            return await coro
        finally:
            timing[key] = round((time.perf_counter() - t) * 1000, 1)

    @staticmethod
    def _begin_task(intent: Intent, ctx: ToolContext) -> Optional[str]:
        """Bind this turn to the current task. Returns the mode, used by :meth:`_close_task`:

        * ``None`` — the turn has no task: questions / chat, and fast-path turns (source "fast").
        * ``"started"`` — a new action (the previous task is shelved), or a continuation with nothing to continue.
        * ``"continued"`` — the intent step said it continues the task (waiting_for_user → active, or revived).
        * ``"fallback"`` — the intent step failed (source "legacy") while a task is current: we can't tell a
          follow-up ("next") from a new request, so the task is kept and its steps booked, but never closed.
        """
        if intent.kind != "action" or intent.source == "fast":
            return None
        current = tasks.current()
        if intent.source == "legacy" and current is not None and not intent.continues_task:
            if current.status != "active":
                tasks.set_status("active")
            return "fallback"
        if not intent.continues_task:
            tasks.start(intent.request, ctx.channel)
            return "started"
        if current is None:
            if tasks.revive_last() is None:
                tasks.start(intent.request, ctx.channel)  # nothing to continue: track this request as its own task
                return "started"
            return "continued"
        if current.status != "active":
            tasks.set_status("active")
        return "continued"

    @staticmethod
    def _record_step(record: ToolCallRecord) -> None:
        if record.name in _BOOKKEEPING_TOOLS:
            return
        state = _step_state(record)
        tasks.record_step(record.name, record.arguments, record.result.message, state == "ok", state=state)

    @staticmethod
    def _close_task(mode: Optional[str], records: List[ToolCallRecord], outcome: Optional[str] = None) -> None:
        """Turn end for the task this turn started or continued (``mode`` from :meth:`_begin_task`).

        Emergency stop (outcome "stopped", or the stop is on) → shelved, so "continue" can bring it back after
        "resume"; outcome "waiting" (clarify asked the user) → waiting_for_user; "incomplete" (step budget /
        no LLM) → unchanged. A "fallback" turn never finishes the task. Otherwise: done when the last step
        succeeded and no plan item is left; failed after a failed step.
        """
        current = tasks.current()
        if mode is None or current is None or outcome == "incomplete":
            return
        if outcome == "stopped" or gateway.is_stopped:
            tasks.set_status("shelved")
            return
        if outcome == "waiting":
            tasks.set_status("waiting_for_user")
            return
        if mode == "fallback":
            return  # it idles out normally if this was really something else
        steps = [r for r in records if r.name not in _BOOKKEEPING_TOOLS]
        if not steps:
            return
        state = _step_state(steps[-1])
        if state == "ok" and not any(p.status in ("todo", "doing") for p in current.plan):
            tasks.set_status("done")
        elif state == "failed":
            tasks.set_status("failed")

    def _remember_pending(self, sid: str, record: ToolCallRecord, linked: bool) -> None:
        """Note which tool queued a confirmation (and for which task) so "yes" / "no" can be booked on it."""
        if record.result.status == "needs_confirmation":
            current = tasks.current() if linked else None
            self._awaiting[sid] = (record.name, dict(record.arguments), current.id if current else None)

    @staticmethod
    def _task_owns(awaiting: Optional[Tuple[str, Dict[str, Any], Optional[str]]]) -> bool:
        current = tasks.current()
        return bool(awaiting and awaiting[2] and current is not None and current.id == awaiting[2])

    @staticmethod
    def _confirmation_record(pending, awaiting, result: ToolResult) -> ToolCallRecord:
        if awaiting:
            name, args = awaiting[0], awaiting[1]
        else:  # queued before this process saw it (e.g. via the API): name it after the gateway op
            first = pending.actions[0] if getattr(pending, "actions", None) else None
            name, args = (first.op, dict(first.args)) if first is not None else ("confirm", {})
        return ToolCallRecord(name, args, result, 0.0, "confirm")

    # ── Agent loop ──────────────────────────────────────────────────────
    async def _agent(self, query, intent: Intent, facts, memories, history, state, ctx, personality, model_override,
                     timing, stream_tokens, note, prefilled=None, book: bool = False) -> AsyncIterator[Dict[str, Any]]:
        """Final events carry ``records`` (every tool run this turn) and ``outcome``:
        None | "incomplete" (step budget / no LLM) | "waiting" (asked the user) | "stopped" (emergency stop)."""
        is_action = intent.kind == "action"
        base = registry.all() if is_action else registry.read_only()  # questions never act
        recent = " ".join(m.get("content", "")[:200] for m in history[-4:])
        offered: Dict[str, Tool] = {t.name: t for t in select_tools(
            intent.request, base, recent, groups=intent.tool_groups or None,
            include=("task_plan",) if is_action else ())}  # the rules tell the model to plan multi-step work
        if not is_action:
            offered.pop("task_plan", None)  # a question must not edit the current task's plan
        multi_step = is_action and not intent.single_step

        model = model_override or state.model_override or (settings.ACTION_MODEL if is_action else settings.DEFAULT_MODEL)
        model_label = "/".join(resolve_model(model))
        current = tasks.current()
        system = build_system_prompt(personality=personality or state.personality, channel=ctx.channel,
                                     facts=facts, memories=memories, model_label=model_label,
                                     task_summary=current.summary() if current else None)
        messages: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        messages += [m for m in history[-settings.MAX_HISTORY_MESSAGES:] if m.get("role") in ("user", "assistant")]
        request = intent.request.strip() or query
        messages.append({"role": "user", "content": request if request == query
                         else f"{request}\n(user's exact words: {query})"})
        if multi_step:
            yield {"type": "progress", "text": "On it."}

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

        def stopped(text: str = _STOPPED) -> Dict[str, Any]:
            timing["llm_ms"] = round(llm_ms, 1)
            return {"type": "final", "kwargs": dict(
                response=note + text, route="agent", intent="automation_control", status="error",
                tools=[r.to_dict() for r in records], metadata={**self._llm_meta(last, model_label), "stopped": True}),
                "records": records, "outcome": "stopped"}

        max_steps = settings.AGENT_MAX_STEPS_ACTION if is_action else settings.AGENT_MAX_STEPS
        incomplete = False
        for step in range(max_steps):
            if is_action and gateway.is_stopped:  # emergency stop ends a running task at once
                yield stopped(_STOPPED if step or records else _STOPPED_BEFORE)
                return
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
                    metadata={"error": str(exc)}), "records": records, "outcome": "incomplete"}
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
                if book:
                    self._record_step(record)
                if multi_step and _done_ok(record) and tool.name not in _BOOKKEEPING_TOOLS and record.result.message:
                    yield {"type": "progress", "text": _progress_text(record.result.message)}
                content = record.result.message
                if record.result.content_for_llm:
                    content += "\n" + (wrap_untrusted(tool.name, record.result.content_for_llm) if tool.untrusted_output else record.result.content_for_llm)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content or ("ok" if record.result.success else "failed")})
                tainted = tainted or tool.untrusted_output
                sensitive = sensitive or tool.sensitive_output
                self._remember_pending(ctx.session_id, record, book)

                if is_action and gateway.is_stopped:
                    yield stopped()
                    return
                if tool.ends_turn and record.result.success:  # clarify: the question is the reply; wait for the user
                    timing["llm_ms"] = round(llm_ms, 1)
                    lead = (resp.content + "\n\n") if resp.content else ""
                    yield {"type": "final", "kwargs": dict(
                        response=note + lead + record.result.message, route="agent", intent=tool.name, status="ok",
                        tools=[r.to_dict() for r in records], metadata=self._llm_meta(resp, model_label)),
                        "records": records, "outcome": "waiting"}
                    return
                if record.result.status == "needs_confirmation":
                    timing["llm_ms"] = round(llm_ms, 1)
                    yield {"type": "final", "kwargs": dict(
                        response=note + record.result.message, route="agent", intent=tool.name, status="needs_confirmation",
                        pending=self._pending_dict(record.result), tools=[r.to_dict() for r in records],
                        metadata=self._llm_meta(resp, model_label)), "records": records}
                    return

            # Latency: a single-step request whose calls were all self-describing actions ends here, without
            # another LLM round. Multi-step tasks send the results back so the model can continue.
            if intent.single_step and round_records and all(registry.get(r.name).final for r in round_records):
                text = "\n".join(r.result.message for r in round_records if r.result.message)
                lead = (resp.content + "\n\n") if resp.content else ""
                timing["llm_ms"] = round(llm_ms, 1)
                status = "dry_run" if any(r.result.status == "dry_run" for r in round_records) else (
                    "ok" if all(r.result.success for r in round_records) else "error")
                yield {"type": "final", "kwargs": dict(
                    response=note + lead + text, route="agent", intent=round_records[-1].name, status=status,
                    tools=[r.to_dict() for r in records], metadata=self._llm_meta(resp, model_label)),
                    "records": records}
                return
        else:
            logger.warning("[Brain] Agent hit max steps")
            incomplete = bool(last and last.tool_calls)

        timing["llm_ms"] = round(llm_ms, 1)
        answer = (last.content if last else "") or (
            "I ran out of steps before finishing." if incomplete else "Done.")
        yield {"type": "final", "kwargs": dict(
            response=note + answer, route="agent", intent=records[-1].name if records else "conversation",
            status="ok", tools=[r.to_dict() for r in records], metadata=self._llm_meta(last, model_label)),
            "records": records, "outcome": "incomplete" if incomplete else None}

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

    async def _memory_context(self, query: str, is_command: bool, session_id: Optional[str] = None):
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

                # Other sessions only: this session's turns are already in the history.
                for doc in await asyncio.to_thread(vector_memory.search_similar, query, 3, None, session_id):
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
        task = loop.create_task(self._persist(result))
        self._persisting.add(task)
        task.add_done_callback(self._persisting.discard)

    async def flush_persistence(self, timeout: float = 5.0) -> None:
        """Wait for this loop's pending post-turn persistence (shutdown, tests)."""
        loop = asyncio.get_running_loop()
        pending = [t for t in self._persisting if not t.done() and t.get_loop() is loop]
        if pending:
            await asyncio.wait(pending, timeout=timeout)

    async def _persist(self, result: TurnResult) -> None:
        try:
            from app.database.connection import db_manager
            from app.database.models import ChatLogRecord
            from memory.search import index_turn

            meta = result.metadata
            usage = meta.get("usage") or {}
            record = ChatLogRecord(
                session_id=result.session_id, user_query=result.query, intent=result.intent,
                confidence=1.0 if result.route == "fast_path" else 0.9, provider=meta.get("provider", "jarvis"),
                model=meta.get("model") or "fast-path", latency_ms=meta.get("total_latency_ms", 0.0),
                prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0), response_text=result.response,
                channel=meta.get("channel", "text"),
            )
            chat_id = await db_manager.log_chat_telemetry(record)
            if chat_id:
                await index_turn(chat_id, result.query, result.response, created_at=record.created_at,
                                 channel=record.channel, session_id=result.session_id)
        except Exception as exc:  # never let bookkeeping break a turn
            logger.debug(f"[Brain] post-turn persistence skipped: {exc}")
        if result.route == "agent" and result.intent == "conversation":
            # Fact extraction may wait on an LLM: its own task, so flush_persistence() doesn't wait for it.
            task = asyncio.get_running_loop().create_task(self._extract_facts(result.query))
            self._extracting.add(task)
            task.add_done_callback(self._extracting.discard)

    async def _extract_facts(self, query: str) -> None:
        try:
            from memory.long_term import long_term_memory

            await long_term_memory.auto_extract_facts(query)
            if settings.ROUTER_USE_LLM:
                await long_term_memory.llm_extract_facts(query)
        except Exception as exc:
            logger.debug(f"[Brain] fact extraction skipped: {exc}")


brain = Brain()

__all__ = ["Brain", "TurnResult", "brain", "json"]
