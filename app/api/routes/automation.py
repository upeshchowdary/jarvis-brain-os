"""JARVIS Automation REST API Endpoints.

Provides HTTP access to the automation system for testing and integration.
"""

import asyncio
from typing import Any, Dict, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from automation.orchestrator import automation_orchestrator
from automation.action_executor import trigger_emergency_stop, clear_emergency_stop, is_emergency_stopped
from automation.workflow_recorder import workflow_recorder
from automation.task_manager import task_manager
from automation.config import automation_config

router = APIRouter(prefix="/automation", tags=["Automation"])


class AutomationCommandRequest(BaseModel):
    command: str = Field(..., description="Natural language automation command")
    session_id: Optional[str] = Field(default="automation")
    dry_run: Optional[bool] = Field(default=None, description="Override dry-run mode for this request")


class RecordStartRequest(BaseModel):
    name: str = Field(..., description="Workflow name")
    description: str = Field(default="")
    trigger_phrases: Optional[list] = Field(default=None)


class RecordStepRequest(BaseModel):
    action: str
    target: str = ""
    value: str = ""
    description: str = ""


class ReplayRequest(BaseModel):
    name: str = Field(..., description="Workflow name to replay")
    adaptive: bool = Field(default=True, description="Use Vision-guided adaptive replay")


# ── Execute ──────────────────────────────────────────────────────────────────

@router.post("/execute")
async def execute_automation(request: AutomationCommandRequest) -> Dict[str, Any]:
    """Execute a natural language automation command."""
    if request.dry_run is not None:
        automation_config.DRY_RUN = request.dry_run

    result = await automation_orchestrator.execute(
        command=request.command,
        session_id=request.session_id or "automation",
    )
    return {
        "success": result.success,
        "summary": result.summary,
        "task_id": result.task_id,
        "elapsed_ms": result.elapsed_ms,
        "steps_done": result.steps_done,
        "steps_failed": result.steps_failed,
        "dry_run": result.dry_run,
        "details": result.details,
    }


# ── Emergency Stop ────────────────────────────────────────────────────────────

@router.post("/stop")
async def emergency_stop() -> Dict[str, Any]:
    """Trigger emergency stop — halts all running automation immediately."""
    trigger_emergency_stop("API call")
    return {"success": True, "message": "Emergency stop triggered. All automation halted."}


@router.post("/resume")
async def resume_automation() -> Dict[str, Any]:
    """Clear the emergency stop flag so new tasks can run."""
    clear_emergency_stop()
    return {"success": True, "message": "Emergency stop cleared. Ready to automate."}


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get current automation system status."""
    task_status = task_manager.get_status()
    return {
        "success": True,
        "is_running": task_status.get("running", False),
        "active_task": task_status if task_status.get("running") else None,
        "is_recording": workflow_recorder.is_recording,
        "emergency_stopped": is_emergency_stopped(),
        "dry_run": automation_config.DRY_RUN,
        "saved_workflows": workflow_recorder.list_workflows(),
        "config": {
            "max_retries": automation_config.MAX_RETRIES,
            "action_timeout": automation_config.ACTION_TIMEOUT,
            "task_timeout": automation_config.TASK_TIMEOUT,
            "confirmation_level": automation_config.CONFIRMATION_LEVEL,
        },
    }


# ── Screenshot ────────────────────────────────────────────────────────────────

@router.get("/screenshot")
async def take_screenshot() -> Dict[str, Any]:
    """Capture the current screen and return a vision description."""
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


# ── Workflow Recording ────────────────────────────────────────────────────────

@router.post("/record/start")
async def start_recording(request: RecordStartRequest) -> Dict[str, Any]:
    """Start recording a new workflow."""
    result = workflow_recorder.start_recording(
        name=request.name,
        description=request.description,
        trigger_phrases=request.trigger_phrases,
    )
    return result


@router.post("/record/step")
async def record_step(request: RecordStepRequest) -> Dict[str, Any]:
    """Manually add a step to the active recording."""
    if not workflow_recorder.is_recording:
        return {"success": False, "error": "Not currently recording."}
    workflow_recorder.record_step(
        action=request.action,
        target=request.target,
        value=request.value,
        description=request.description,
    )
    return {"success": True, "step_recorded": request.action}


@router.post("/record/stop")
async def stop_recording() -> Dict[str, Any]:
    """Stop recording and save the workflow."""
    if not workflow_recorder.is_recording:
        return {"success": False, "error": "Not currently recording."}
    stop_result = workflow_recorder.stop_recording()
    save_result = workflow_recorder.save_workflow()
    return {**stop_result, **save_result}


# ── Workflow Management ────────────────────────────────────────────────────────

@router.get("/workflows")
async def list_workflows() -> Dict[str, Any]:
    """List all saved workflows."""
    names = workflow_recorder.list_workflows()
    workflows = []
    for name in names:
        wf = workflow_recorder.load_workflow(name)
        if wf:
            workflows.append({
                "name": wf.name,
                "description": wf.description,
                "steps": len(wf.steps),
                "trigger_phrases": wf.trigger_phrases,
                "run_count": wf.run_count,
                "success_rate": round(wf.success_rate * 100, 1),
            })
    return {"success": True, "count": len(workflows), "workflows": workflows}


@router.post("/replay")
async def replay_workflow(request: ReplayRequest) -> Dict[str, Any]:
    """Replay a saved workflow."""
    from automation.workflow_player import workflow_player
    result = await workflow_player.replay(request.name, adaptive=request.adaptive)
    return result


@router.delete("/workflows/{name}")
async def delete_workflow(name: str) -> Dict[str, Any]:
    """Delete a saved workflow."""
    return workflow_recorder.delete_workflow(name)
