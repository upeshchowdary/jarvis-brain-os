"""Productivity endpoints (reminders, notes, to-dos, routines, clipboard, briefing) and the live event stream."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core.events import bus, wire_gateway
from services import routines
from services.store import store
from services.timeparse import parse_when, strip_time_phrase

router = APIRouter(tags=["Productivity"])


class ReminderIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    when: str = Field(..., min_length=1, max_length=200)


class TextIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class RoutineRun(BaseModel):
    variables: Dict[str, Any] = Field(default_factory=dict)
    dry_run: Optional[bool] = None
    session_id: str = "dashboard"


@router.get("/reminders")
async def list_reminders() -> Dict[str, Any]:
    return {"reminders": await store.reminders()}


@router.post("/reminders")
async def add_reminder(body: ReminderIn) -> Dict[str, Any]:
    parsed = parse_when(body.when)
    if parsed is None:
        raise HTTPException(status_code=422, detail=f"Couldn't understand the time '{body.when}'.")
    rid = await store.add_reminder(strip_time_phrase(body.text, parsed) or body.text, parsed.due.timestamp(), parsed.recurrence)
    return {"id": rid, "due": parsed.due.isoformat(), "description": parsed.describe()}


@router.delete("/reminders/{rid}")
async def cancel_reminder(rid: int) -> Dict[str, Any]:
    return {"cancelled": bool(await store.mark_reminder(rid, "cancelled"))}


@router.get("/notes")
async def list_notes(search: str = "") -> Dict[str, Any]:
    return {"notes": await store.notes(search)}


@router.post("/notes")
async def add_note(body: TextIn) -> Dict[str, Any]:
    return {"id": await store.add_note(body.text)}


@router.get("/todos")
async def list_todos(include_done: bool = False) -> Dict[str, Any]:
    return {"todos": await store.todos(include_done)}


@router.post("/todos")
async def add_todo(body: TextIn) -> Dict[str, Any]:
    return {"id": await store.add_todo(body.text)}


@router.post("/todos/{tid}/done")
async def complete_todo(tid: int) -> Dict[str, Any]:
    return {"done": [t["text"] for t in await store.complete_todo(str(tid))]}


@router.get("/clipboard")
async def clipboard_history(limit: int = 20) -> Dict[str, Any]:
    return {"items": await store.clips(limit)}


@router.get("/routines")
async def list_routines() -> Dict[str, Any]:
    return {"routines": [{"name": n, "description": r.get("description", ""), "variables": r.get("variables", {})}
                         for n, r in routines.load_all().items()]}


@router.post("/routines/{name}/run")
async def run_routine(name: str, body: RoutineRun) -> Dict[str, Any]:
    result, messages = await routines.run(name, body.session_id, body.variables, dry_run=body.dry_run, source="api")
    if result is None and messages and messages[0].startswith("I don't have"):
        raise HTTPException(status_code=404, detail=messages[0])
    return {"result": result.to_dict() if result else None, "messages": messages}


@router.get("/briefing")
async def briefing() -> Dict[str, Any]:
    from tools.builtin.productivity import build_briefing

    return {"briefing": await build_briefing()}


@router.get("/events")
async def events() -> StreamingResponse:
    """Server-Sent Events: pending confirmations, audit entries, reminders, alerts, transcript."""
    wire_gateway()

    async def gen() -> AsyncIterator[str]:
        q = bus.subscribe()
        try:
            for ev in list(bus.recent)[-30:]:
                yield f"data: {json.dumps(ev, default=str)}\n\n"
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(ev, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
