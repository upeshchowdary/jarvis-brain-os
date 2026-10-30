"""Computer-use plans: multi-step UI automation and workflow replay.

A plan is a list of step dicts::

    {"action": "navigate|open_app|click|double_click|type|fill|press|hotkey|scroll|focus_window|wait",
     "target": "...", "value": "...", "level": "L2_DOM_WEB" (optional),
     "x": 10, "y": 20 (optional), "verify_type": "...", "verify_target": "..."}

The whole plan is previewed and confirmed once (its risk is the highest step risk),
then each step is executed through the action hierarchy (DOM → UIA → keyboard →
mouse/OCR → vision) with verification and recovery. Free-form shell is never allowed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from automation.gateway import OpResult, Preview, Risk, gateway

STEP_RISK: Dict[str, Risk] = {
    "navigate": Risk.LOW,
    "open_app": Risk.LOW,
    "read_file": Risk.LOW,
    "scroll": Risk.LOW,
    "wait": Risk.LOW,
    "focus_window": Risk.LOW,
    "click": Risk.MEDIUM,
    "double_click": Risk.MEDIUM,
    "right_click": Risk.MEDIUM,
    "type": Risk.MEDIUM,
    "fill": Risk.MEDIUM,
    "fill_field": Risk.MEDIUM,
    "press": Risk.MEDIUM,
    "hotkey": Risk.MEDIUM,
}
DEFAULT_LEVEL = {
    "navigate": "L2_DOM_WEB",
    "open_app": "L1_DIRECT_API",
    "read_file": "L1_DIRECT_API",
    "click": "L3_WINDOWS_UIA",
    "double_click": "L5_MOUSE",
    "right_click": "L5_MOUSE",
    "type": "L4_KEYBOARD",
    "fill": "L2_DOM_WEB",
    "fill_field": "L2_DOM_WEB",
    "press": "L4_KEYBOARD",
    "hotkey": "L4_KEYBOARD",
}


def step_risk(step: Dict[str, Any]) -> Risk:
    return STEP_RISK.get(str(step.get("action", "")).lower(), Risk.HIGH)


def plan_risk(args: Dict[str, Any]) -> Risk:
    steps = args.get("steps") or []
    return max((step_risk(s) for s in steps), default=Risk.LOW)


def describe_step(step: Dict[str, Any]) -> str:
    if step.get("description"):
        return str(step["description"])
    action = step.get("action", "?")
    target, value = step.get("target", ""), step.get("value", "")
    return {
        "navigate": f"Go to {target}",
        "open_app": f"Open {target}",
        "click": f"Click “{target}”",
        "double_click": f"Double-click “{target}”",
        "type": f"Type “{value or target}”",
        "fill": f"Fill “{target}” with “{value}”",
        "fill_field": f"Fill “{target}” with “{value}”",
        "press": f"Press {value or target}",
        "hotkey": f"Press {value or target}",
        "scroll": f"Scroll {value or 'down'}",
        "wait": f"Wait {value or 1}s",
        "focus_window": f"Switch to {target}",
    }.get(action, f"{action} {target} {value}".strip())


def _plan_preview(args: Dict[str, Any]) -> Preview:
    steps = args.get("steps") or []
    if not steps:
        return Preview("The plan is empty.", ok=False)
    bad = [s.get("action") for s in steps if step_risk(s) >= Risk.HIGH]
    if bad:
        return Preview(f"The plan contains actions I won't run automatically: {', '.join(map(str, bad))}.", ok=False)
    lines = [f"{args.get('goal') or 'Computer task'}:"]
    lines += [f"   {i}. {describe_step(s)}" for i, s in enumerate(steps, 1)]
    return Preview("\n".join(lines))


async def _run_step(step: Dict[str, Any]) -> tuple[bool, str]:
    from automation.engine.action_router import ActionLevel, ActionSpec, action_router
    from automation.engine.recovery import recovery_engine
    from automation.engine.verification import VerificationCondition, verification_engine

    action = str(step.get("action", "")).lower()
    target = str(step.get("target", "") or "")
    value = str(step.get("value", "") or "")

    if action == "wait":
        try:
            await asyncio.sleep(min(float(value or 1), 30.0))
        except ValueError:
            await asyncio.sleep(1.0)
        return True, ""
    if action == "scroll":
        from automation.mouse_controller import mouse_controller

        amount = int(target) if target.isdigit() else 5
        res = await mouse_controller.scroll(amount, direction=value if value in ("up", "down") else "down")
        return bool(res.get("success")), res.get("error", "")
    if action == "focus_window":
        from automation.window_controller import window_controller

        res = await window_controller.focus_window(target)
        return bool(res.get("success")), res.get("error", "")
    if action in ("double_click", "right_click") or ("x" in step and "y" in step and action == "click"):
        from automation.mouse_controller import mouse_controller
        from automation.screen_observer import screen_observer

        if "x" in step and "y" in step:
            x, y = int(step["x"]), int(step["y"])
        else:
            loc = await screen_observer.find_text_location(target)
            if not loc:
                return False, f"Couldn't find “{target}” on screen."
            x, y = loc
        func = {"double_click": mouse_controller.double_click, "right_click": mouse_controller.right_click}.get(action, mouse_controller.click)
        res = await func(x, y)
        return bool(res.get("success")), res.get("error", "")

    level_name = step.get("level") or DEFAULT_LEVEL.get(action, "L3_WINDOWS_UIA")
    try:
        level = ActionLevel(level_name)
    except ValueError:
        level = ActionLevel(DEFAULT_LEVEL.get(action, "L3_WINDOWS_UIA"))
    spec = ActionSpec(level=level, action_type="fill" if action == "fill_field" else action, target=target, value=value)
    condition = None
    if step.get("verify_type") and step.get("verify_type") != "none" and step.get("verify_target"):
        condition = VerificationCondition(condition_type=step["verify_type"], target=step["verify_target"], timeout_seconds=4.0)

    res = await action_router.dispatch(spec)
    ok = res.success
    if ok and condition is not None:
        ok = (await verification_engine.verify(condition)).passed
    if not ok and action in ("click", "type", "fill", "fill_field"):
        res = await recovery_engine.escalate_and_retry(spec, condition)
        ok = res.success
    return ok, (res.error or "") if not ok else ""


@gateway.operation("computer.run_plan", plan_risk, "Run a multi-step computer task", preview=_plan_preview)
async def run_plan(goal: str = "", steps: List[Dict[str, Any]] | None = None) -> OpResult:
    steps = steps or []
    done = 0
    for i, step in enumerate(steps, 1):
        if gateway.is_stopped:
            return OpResult.fail(f"Stopped after {done}/{len(steps)} steps (emergency stop).")
        try:
            ok, err = await _run_step(step)
        except Exception as exc:  # a broken step must not crash the server
            ok, err = False, str(exc)
        if not ok:
            return OpResult.fail(f"Step {i} ({describe_step(step)}) failed: {err or 'unknown error'}. Completed {done}/{len(steps)} steps.",
                                 data={"completed": done, "total": len(steps)})
        done += 1
        await asyncio.sleep(0.05)
    return OpResult.ok(f"Done — {goal or 'task'} ({done}/{len(steps)} steps).", data={"completed": done, "total": len(steps)})
