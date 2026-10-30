"""Automation endpoints: execute (with dry-run), confirmations, audit log, e-stop, workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from automation.gateway import ActionRequest, gateway
from automation.workflow_recorder import workflow_recorder
from brain.pipeline import brain
from core.config import settings

router = APIRouter(prefix="/automation", tags=["Automation"])


class AutomationCommandRequest(BaseModel):
    command: str = Field(..., min_length=1, max_length=4000, description="Natural language command")
    session_id: Optional[str] = Field(default="automation")
    dry_run: Optional[bool] = Field(default=None, description="true = return the plan only, execute nothing")


class RecordStartRequest(BaseModel):
    name: str
    description: str = ""
    trigger_phrases: Optional[list] = None


class RecordStepRequest(BaseModel):
    action: str
    target: str = ""
    value: str = ""
    description: str = ""


class ReplayRequest(BaseModel):
    name: str
    params: Dict[str, str] = Field(default_factory=dict)
    dry_run: Optional[bool] = None
    session_id: str = "automation"


@router.post("/execute")
async def execute_automation(request: AutomationCommandRequest) -> Dict[str, Any]:
    """Run a command through the brain. ``dry_run`` is passed all the way to the gateway."""
    result = await brain.handle(request.command, request.session_id, channel="api", dry_run=request.dry_run)
    out = result.to_dict()
    out["success"] = result.status in ("ok", "dry_run", "needs_confirmation")
    out["summary"] = result.response
    out["dry_run"] = bool(request.dry_run if request.dry_run is not None else settings.DRY_RUN)
    return out


@router.get("/pending")
async def list_pending() -> Dict[str, Any]:
    return {"pending": [p.to_dict() for p in gateway.list_pending()]}


@router.post("/confirm/{pending_id}")
async def confirm_action(pending_id: str) -> Dict[str, Any]:
    if gateway.get_pending(pending_id) is None:
        raise HTTPException(status_code=404, detail="No such pending action (it may have expired).")
    return (await gateway.confirm(pending_id)).to_dict()


@router.post("/cancel/{pending_id}")
async def cancel_action(pending_id: str) -> Dict[str, Any]:
    return {"cancelled": gateway.cancel(pending_id, "cancelled via API")}


@router.get("/audit")
async def audit_log(limit: int = Query(default=100, ge=1, le=2000)) -> Dict[str, Any]:
    return {"entries": gateway.read_audit(limit)}


@router.get("/operations")
async def operations() -> Dict[str, Any]:
    return {"operations": gateway.list_operations()}


@router.post("/undo")
async def undo_last(session_id: str = "automation") -> Dict[str, Any]:
    return (await gateway.undo_last(session_id)).to_dict()


@router.post("/stop")
async def emergency_stop() -> Dict[str, Any]:
    gateway.emergency_stop("API call")
    return {"success": True, "message": "Emergency stop triggered. All automation halted."}


@router.post("/resume")
async def resume_automation() -> Dict[str, Any]:
    gateway.resume()
    return {"success": True, "message": "Emergency stop cleared."}


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    return {
        "success": True,
        "emergency_stopped": gateway.is_stopped,
        "dry_run": settings.DRY_RUN,
        "confirm_medium_risk": settings.CONFIRM_MEDIUM_RISK,
        "privacy_mode": settings.PRIVACY_MODE,
        "pending": [p.to_dict() for p in gateway.list_pending()],
        "is_recording": workflow_recorder.is_recording,
        "saved_workflows": workflow_recorder.list_workflows(),
    }


@router.get("/screenshot")
async def take_screenshot() -> Dict[str, Any]:
    """Describe the current screen (requires the API token; screen text is sensitive)."""
    from automation.screen_observer import screen_observer

    state = await screen_observer.analyze("Describe what is on the screen.")
    return {
        "success": True,
        "active_app": state.active_app,
        "window_title": state.window_title,
        "vision_description": state.vision_description,
        "ocr_text_preview": state.ocr_text[:500],
        "ocr_item_count": len(state.ocr_items),
    }


# ── Workflows ─────────────────────────────────────────────────────────────

@router.post("/record/start")
async def start_recording(request: RecordStartRequest) -> Dict[str, Any]:
    return workflow_recorder.start_recording(name=request.name, description=request.description, trigger_phrases=request.trigger_phrases)


@router.post("/record/step")
async def record_step(request: RecordStepRequest) -> Dict[str, Any]:
    if not workflow_recorder.is_recording:
        return {"success": False, "error": "Not currently recording."}
    workflow_recorder.record_step(action=request.action, target=request.target, value=request.value, description=request.description)
    return {"success": True, "step_recorded": request.action}


@router.post("/record/stop")
async def stop_recording() -> Dict[str, Any]:
    if not workflow_recorder.is_recording:
        return {"success": False, "error": "Not currently recording."}
    stop_result = workflow_recorder.stop_recording()
    save_result = workflow_recorder.save_workflow()
    return {**stop_result, **save_result}


@router.get("/workflows")
async def list_workflows() -> Dict[str, Any]:
    workflows = []
    for name in workflow_recorder.list_workflows():
        wf = workflow_recorder.load_workflow(name)
        if wf:
            workflows.append({"name": wf.name, "description": wf.description, "steps": len(wf.steps),
                              "trigger_phrases": wf.trigger_phrases, "run_count": wf.run_count,
                              "success_rate": round(wf.success_rate * 100, 1)})
    return {"success": True, "count": len(workflows), "workflows": workflows}


@router.post("/replay")
async def replay_workflow(request: ReplayRequest) -> Dict[str, Any]:
    """Replay a workflow — goes through the gateway (preview, confirmation, dry-run, audit)."""
    from automation.workflow_player import workflow_player

    req = workflow_player.build_request(request.name, request.params)
    if req is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{request.name}' not found.")
    return (await gateway.execute(req, request.session_id, dry_run=request.dry_run, source="api")).to_dict()


@router.delete("/workflows/{name}")
async def delete_workflow(name: str) -> Dict[str, Any]:
    return workflow_recorder.delete_workflow(name)


__all__ = ["router", "ActionRequest"]
