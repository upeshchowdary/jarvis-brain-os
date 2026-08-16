"""Quick validation test for the JARVIS Automation module."""

import asyncio

async def run_tests():
    from automation.config import automation_config
    automation_config.DRY_RUN = True

    from automation.mouse_controller import MouseController
    from automation.keyboard_controller import KeyboardController
    from automation.safety_manager import safety_manager, RiskLevel
    from automation.workflow_recorder import workflow_recorder

    mouse = MouseController(dry_run=True)
    kbd = KeyboardController(dry_run=True)

    # Test mouse dry run
    r = await mouse.click(100, 200)
    assert r.get("dry_run"), "mouse click dry_run flag missing"
    print("  [OK] mouse.click dry-run")

    r = await mouse.scroll(3, "down")
    assert r.get("dry_run"), "scroll dry_run flag missing"
    print("  [OK] mouse.scroll dry-run")

    # Test keyboard dry run
    r = await kbd.type_text("Hello JARVIS")
    assert r.get("dry_run"), "type_text dry_run flag missing"
    print("  [OK] keyboard.type_text dry-run")

    r = await kbd.hotkey("ctrl", "c")
    assert r.get("dry_run"), "hotkey dry_run flag missing"
    print("  [OK] keyboard.hotkey dry-run")

    # Test safety classification
    assert safety_manager.classify("delete_file", "/tmp/test") == RiskLevel.HIGH
    print("  [OK] safety: delete_file = HIGH")

    assert safety_manager.classify("scroll", "") == RiskLevel.LOW
    print("  [OK] safety: scroll = LOW")

    assert safety_manager.classify("click", "") == RiskLevel.MEDIUM
    print("  [OK] safety: click = MEDIUM")

    # Test safety blocking
    allowed, reason = safety_manager.is_allowed("delete_file", "/tmp/test", user_confirmed=False)
    assert not allowed, "delete_file should be blocked without confirmation"
    print("  [OK] safety: delete_file blocked without confirmation")

    # Test workflow recorder
    workflow_recorder.start_recording("test_workflow", "Test")
    workflow_recorder.record_step("click", "Submit button", description="Click submit")
    workflow_recorder.record_step("type", value="Hello World", description="Type hello")
    stop = workflow_recorder.stop_recording()
    steps = stop.get("steps", 0)
    assert steps == 2, f"Expected 2 steps, got {steps}"
    print("  [OK] workflow recorder: start/record/stop")

    # Test workflow save & load
    save = workflow_recorder.save_workflow("test_workflow")
    assert save.get("success"), f"Workflow save failed: {save}"
    print("  [OK] workflow save")

    loaded = workflow_recorder.load_workflow("test_workflow")
    assert loaded is not None, "Workflow load returned None"
    assert len(loaded.steps) == 2, f"Expected 2 steps loaded, got {len(loaded.steps)}"
    print("  [OK] workflow load")

    # Cleanup
    workflow_recorder.delete_workflow("test_workflow")
    print("  [OK] workflow delete")

    print("\nALL AUTOMATION TESTS PASSED")


asyncio.run(run_tests())
