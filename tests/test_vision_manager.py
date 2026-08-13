"""Unit tests for vision.manager VisionManager orchestrator."""

import pytest
from vision.manager import vision_manager, VisionManager
from vision.environment import EnvironmentState


@pytest.mark.asyncio
async def test_vision_manager_screen_capture_cycle():
    state = await vision_manager.capture_environment_state(source="screen", include_visual_reasoning=False)
    assert isinstance(state, EnvironmentState)
    assert state.active_application != ""
    assert isinstance(state.summary, str)
    assert state.confidence > 0.0


@pytest.mark.asyncio
async def test_vision_manager_camera_capture_cycle():
    state = await vision_manager.capture_environment_state(source="camera", include_visual_reasoning=False)
    assert isinstance(state, EnvironmentState)
    assert isinstance(state.summary, str)
