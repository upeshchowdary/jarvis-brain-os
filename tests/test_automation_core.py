"""Validation tests for low-level automation controllers and the workflow recorder."""

import asyncio


async def run_tests():
    from automation.gateway import ActionRequest, Risk, gateway
    from automation.keyboard_controller import KeyboardController
    from automation.mouse_controller import MouseController
    from automation.workflow_recorder import workflow_recorder

    mouse = MouseController(dry_run=True)
    kbd = KeyboardController(dry_run=True)
    assert (await mouse.click(100, 200)).get("dry_run")
    assert (await mouse.scroll(3, "down")).get("dry_run")
    assert (await kbd.type_text("Hello JARVIS")).get("dry_run")
    assert (await kbd.hotkey("ctrl", "c")).get("dry_run")

    # Risk classification now lives in the gateway.
    assert gateway.classify(ActionRequest("process.kill", {"target": "x"})) == Risk.HIGH
    assert gateway.classify(ActionRequest("system.volume", {"action": "up"})) == Risk.LOW
    assert gateway.classify(ActionRequest("computer.run_plan", {"steps": [{"action": "click", "target": "OK"}]})) == Risk.MEDIUM

    workflow_recorder.start_recording("test_workflow", "Test")
    workflow_recorder.record_step("click", "Submit button", description="Click submit")
    workflow_recorder.record_step("type", value="Hello World", description="Type hello")
    assert workflow_recorder.stop_recording().get("steps") == 2
    assert workflow_recorder.save_workflow("test_workflow").get("success")
    loaded = workflow_recorder.load_workflow("test_workflow")
    assert loaded is not None and len(loaded.steps) == 2
    assert workflow_recorder.delete_workflow("test_workflow")["success"]


def test_automation_core():
    asyncio.run(run_tests())
