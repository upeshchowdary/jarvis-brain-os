"""Unit tests for vision.visual_reasoner and provider adapters."""

import pytest
from PIL import Image
from vision.visual_reasoner import visual_reasoner, VisualReasoner
from vision.providers.base import BaseVisionProvider
from vision.providers.openai import OpenAIVisionProvider
from vision.providers.gemini import GeminiVisionProvider
from vision.providers.qwen import QwenVisionProvider
from vision.providers.llava import LLaVAVisionProvider


class MockVisionProvider(BaseVisionProvider):
    """Mock vision provider for unit testing fallback queues."""

    def __init__(self, name: str = "mock", should_succeed: bool = True) -> None:
        super().__init__(provider_name=name, model_name="mock-v1", api_key="mock_key")
        self.should_succeed = should_succeed

    async def analyze_image(self, image: str, prompt: str, **kwargs):
        if self.should_succeed:
            return {"success": True, "provider": self.provider_name, "analysis": f"Mock visual response to: {prompt}"}
        return {"success": False, "provider": self.provider_name, "error": "Mock provider error"}

    async def health_check(self) -> bool:
        return True


def test_provider_registration_and_list():
    vr = VisualReasoner()
    vr.register_provider("mock", MockVisionProvider("mock"))

    providers = vr.list_providers()
    assert "mock" in providers
    assert vr.switch_provider("mock") is True
    assert vr.active_provider_name == "mock"


@pytest.mark.asyncio
async def test_visual_reasoner_analyze_image_with_mock():
    vr = VisualReasoner()
    vr.register_provider("mock", MockVisionProvider("mock", should_succeed=True))

    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    res = await vr.analyze_image(img, prompt="What color is this?", preferred_provider="mock")

    assert res["success"] is True
    assert res["provider"] == "mock"
    assert "Mock visual response" in res["analysis"]


@pytest.mark.asyncio
async def test_visual_reasoner_fallback_queue():
    vr = VisualReasoner()
    vr._providers = {
        "failing_mock": MockVisionProvider("failing_mock", should_succeed=False),
        "working_mock": MockVisionProvider("working_mock", should_succeed=True),
    }

    img = Image.new("RGB", (100, 100), color=(0, 255, 0))
    res = await vr.analyze_image(img, prompt="Test fallback", preferred_provider="failing_mock")

    assert res["success"] is True
    assert res["provider"] == "working_mock"


@pytest.mark.asyncio
async def test_describe_scene_helper():
    vr = VisualReasoner()
    vr.register_provider("working_mock", MockVisionProvider("working_mock", should_succeed=True))
    vr.switch_provider("working_mock")

    img = Image.new("RGB", (100, 100), color=(0, 0, 255))
    scene_text = await vr.describe_scene(img)
    assert isinstance(scene_text, str)
    assert len(scene_text) > 0
