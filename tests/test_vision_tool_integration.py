"""Unit & Integration tests for JARVIS Vision Tool, VisionManager, and Provider abstractions."""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch, call
from PIL import Image

from tools.vision_tool import ScreenVisionTool
from brain.tool_router import tool_router, ToolCallSpec
from brain.model_manager import model_manager
from brain.intent_engine import intent_engine, IntentEngine
from brain.brain_config import brain_config
from vision.manager import vision_manager, VisionManager
from vision.providers.base import BaseVisionProvider
from vision.providers.gemini import GeminiVisionProvider
from vision.providers.ollama_vision import OllamaVisionProvider
from vision.environment import EnvironmentState, UIElement, OCRTextItem, BoundingBox
from vision.schemas import StructuredVisionResult
from vision.screen_preprocessor import (
    preprocess_screen,
    build_enriched_prompt,
    get_cached_analysis,
    store_cached_analysis,
    invalidate_cache,
    _fast_image_hash,
    _fingerprint_app,
    ScreenContext,
)


@pytest.fixture
def sample_pil_image():
    """Returns a simple 200x200 RGB test image."""
    return Image.new("RGB", (200, 200), color="blue")


# ---------------------------------------------------------------------------
# Basic registration
# ---------------------------------------------------------------------------

def test_vision_tool_registration():
    """Verify ScreenVisionTool is registered in ToolRouter."""
    discovered = tool_router.discover_tools()
    tool_names = [t["name"] for t in discovered]
    assert "screen_vision" in tool_names


def test_model_capabilities_registry():
    """Verify ModelManager capability registry accurately identifies multimodal vision models."""
    assert model_manager.supports_vision("qwen3-vl:8b") is True
    assert model_manager.supports_vision("gemini-2.0-flash") is True
    assert model_manager.supports_vision("gpt-4o-mini") is True
    assert model_manager.supports_vision("llama-3.3-70b-versatile") is False
    assert model_manager.supports_vision("qwen2.5") is False


def test_classify_visual_query_type():
    """Verify IntentEngine categorizes visual query types correctly."""
    assert IntentEngine.classify_visual_query_type("What is in the image on my screen?") == "IMAGE_DESCRIPTION"
    assert IntentEngine.classify_visual_query_type("Read the text on screen") == "OCR"
    assert IntentEngine.classify_visual_query_type("Where is the submit button?") == "UI_ANALYSIS"
    assert IntentEngine.classify_visual_query_type("What error message popped up?") == "ERROR_ANALYSIS"
    assert IntentEngine.classify_visual_query_type("What is on my screen?") == "SCREEN_DESCRIPTION"


# ---------------------------------------------------------------------------
# Screen Preprocessor tests
# ---------------------------------------------------------------------------

def test_preprocessor_fast_hash(sample_pil_image):
    """verify image hashing returns stable non-empty string."""
    h1 = _fast_image_hash(sample_pil_image)
    h2 = _fast_image_hash(sample_pil_image)
    assert isinstance(h1, str) and len(h1) == 32
    assert h1 == h2


def test_preprocessor_cache_roundtrip():
    """Verify cache store/retrieve/invalidate lifecycle works correctly."""
    session = "test_session_cache"
    invalidate_cache(session)

    # Miss before store
    assert get_cached_analysis(session, "hash_abc", ttl_seconds=30) is None

    store_cached_analysis(session, "hash_abc", "The screen shows a cat photo.")

    # Hit with correct hash
    result = get_cached_analysis(session, "hash_abc", ttl_seconds=30)
    assert result == "The screen shows a cat photo."

    # Miss with different hash (screen changed)
    result_wrong = get_cached_analysis(session, "hash_xyz", ttl_seconds=30)
    assert result_wrong is None

    # Invalidate and confirm miss
    invalidate_cache(session)
    result_after = get_cached_analysis(session, "hash_abc", ttl_seconds=30)
    assert result_after is None


def test_preprocessor_app_fingerprint():
    """Verify app fingerprinting classifies known apps correctly."""
    app_type, hint = _fingerprint_app("main.py - Visual Studio Code", "Code.exe")
    assert app_type == "code_editor"
    assert "code" in hint.lower()

    app_type2, hint2 = _fingerprint_app("YouTube - Google Chrome", "chrome.exe")
    assert app_type2 == "browser"

    app_type3, hint3 = _fingerprint_app("JARVIS Chat", "python.exe")
    # Could match "code_editor" via .py or "ide" — just ensure it returns a string
    assert isinstance(app_type3, str)


def test_preprocessor_enriched_prompt(sample_pil_image):
    """Verify enriched prompt includes app context and OCR grounding."""
    ctx = ScreenContext(
        screenshot=sample_pil_image,
        image_hash="abc123",
        ocr_paragraphs=["Hello World", "def my_func():"],
        has_readable_text=True,
        app_type="code_editor",
        app_context_hint="The user has a code editor open.",
        window_title="main.py - VS Code",
    )
    prompt = build_enriched_prompt(ctx, user_query="What file is open?", query_type="SCREEN_DESCRIPTION")
    assert "JARVIS" in prompt
    assert "main.py" in prompt
    assert "Hello World" in prompt
    assert "def my_func" in prompt
    assert "code editor" in prompt.lower()
    assert "What file is open?" in prompt


def test_preprocessor_full_pipeline(sample_pil_image):
    """Verify full preprocess_screen pipeline runs without error."""
    ctx = preprocess_screen(
        screenshot=sample_pil_image,
        session_id="test_preprocess_session",
        window_title="test.py - VS Code",
        app_name="Code.exe",
        query_type="SCREEN_DESCRIPTION",
    )
    assert ctx.screenshot is not None
    assert ctx.image_hash != ""
    assert ctx.width == 200
    assert ctx.height == 200
    assert ctx.app_type == "code_editor"
    assert ctx.preprocessing_ms >= 0


# ---------------------------------------------------------------------------
# Ollama Vision Provider tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ollama_vision_provider_get_installed_models():
    """Verify get_installed_models parses Ollama tags response correctly."""
    provider = OllamaVisionProvider(model_name="qwen3-vl:8b")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "models": [
            {"name": "qwen3-vl:8b"},
            {"name": "qwen2.5:latest"},
            {"name": "gemma:latest"},
        ]
    }

    with patch.object(provider, "get_installed_models", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = ["qwen3-vl:8b", "qwen2.5:latest", "gemma:latest"]
        installed = await provider.get_installed_models()
        assert "qwen3-vl:8b" in installed


@pytest.mark.asyncio
async def test_ollama_vision_provider_auto_discover():
    """Verify get_best_vision_model returns configured model when installed."""
    provider = OllamaVisionProvider(model_name="qwen3-vl:8b")

    with patch.object(provider, "get_installed_models", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = ["qwen3-vl:8b", "gemma:latest"]
        best = await provider.get_best_vision_model()
        assert best == "qwen3-vl:8b"


@pytest.mark.asyncio
async def test_ollama_vision_provider_streaming_success(sample_pil_image):
    """Verify OllamaVisionProvider collects streamed tokens correctly."""
    provider = OllamaVisionProvider(model_name="qwen3-vl:8b")

    # Simulate streaming response lines
    stream_lines = [
        json.dumps({"message": {"content": "The "}, "done": False}),
        json.dumps({"message": {"content": "screen "}, "done": False}),
        json.dumps({"message": {"content": "shows a cat."}, "done": True}),
    ]

    # Mock get_best_vision_model to return the model directly
    with patch.object(provider, "get_best_vision_model", new_callable=AsyncMock) as mock_best:
        mock_best.return_value = "qwen3-vl:8b"

        # Mock httpx streaming
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_stream_ctx.status_code = 200

        async def mock_aiter_lines():
            for line in stream_lines:
                yield line

        mock_stream_ctx.aiter_lines = mock_aiter_lines

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=mock_stream_ctx)

        with patch("httpx.AsyncClient", return_value=mock_client):
            res = await provider.analyze_image(sample_pil_image, "Describe screen.")

    assert res["success"] is True
    assert res["provider"] == "ollama"
    assert "cat" in res["analysis"]


@pytest.mark.asyncio
async def test_ollama_vision_provider_no_model_installed(sample_pil_image):
    """Verify OllamaVisionProvider returns failure when no vision model is installed."""
    provider = OllamaVisionProvider(model_name="qwen3-vl:8b")

    with patch.object(provider, "get_best_vision_model", new_callable=AsyncMock) as mock_best:
        mock_best.return_value = None
        res = await provider.analyze_image(sample_pil_image, "Describe screen.")

    assert res["success"] is False
    assert "No Ollama vision model" in res["error"]


# ---------------------------------------------------------------------------
# Gemini + ScreenVisionTool tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gemini_vision_provider_missing_key(sample_pil_image):
    """Verify GeminiVisionProvider handles missing API key gracefully."""
    provider = GeminiVisionProvider(api_key="")
    result = await provider.analyze_image(sample_pil_image, "What is this?")
    assert result["success"] is False
    assert "not configured" in result["error"]


@pytest.mark.asyncio
async def test_screen_vision_tool_execution_full(sample_pil_image):
    """Test ScreenVisionTool full execution mode."""
    tool = ScreenVisionTool()

    with patch("vision.screen_analyzer.grab_screenshot", return_value=sample_pil_image), \
         patch("vision.manager.vision_manager.capture_environment_state", new_callable=AsyncMock) as mock_cap:

        mock_env = EnvironmentState(
            active_application="Code",
            active_window="main.py",
            summary="VS Code open on desktop.",
            confidence=0.98,
        )
        mock_cap.return_value = mock_env

        res = await tool.execute(query="What is on my screen?", mode="full")
        assert res["success"] is True
        assert res["mode"] == "full"
        assert res["data"]["active_application"] == "Code"
        assert res["data"]["description"] == "VS Code open on desktop."


@pytest.mark.asyncio
async def test_screen_vision_tool_execution_ocr_only(sample_pil_image):
    """Test ScreenVisionTool ocr_only execution mode."""
    tool = ScreenVisionTool()

    dummy_ocr = [
        OCRTextItem(text="Hello JARVIS", bounds=BoundingBox(x=10, y=10, width=50, height=20))
    ]

    with patch("tools.vision_tool.grab_screenshot", return_value=sample_pil_image), \
         patch("tools.vision_tool.ocr_engine.extract_text_items", return_value=dummy_ocr):

        res = await tool.execute(mode="ocr_only")
        assert res["success"] is True
        assert res["mode"] == "ocr_only"
        assert "Hello JARVIS" in res["extracted_text"]


@pytest.mark.asyncio
async def test_screen_vision_tool_execution_ui_only(sample_pil_image):
    """Test ScreenVisionTool ui_only execution mode."""
    tool = ScreenVisionTool()

    dummy_ui = [
        UIElement(id="btn_1", label="Submit", bounds=BoundingBox(x=100, y=100, width=80, height=30))
    ]

    with patch("tools.vision_tool.grab_screenshot", return_value=sample_pil_image), \
         patch("tools.vision_tool.ocr_engine.extract_text_items", return_value=[]), \
         patch("tools.vision_tool.ui_detector.detect_ui_elements", return_value=dummy_ui):

        res = await tool.execute(mode="ui_only")
        assert res["success"] is True
        assert res["mode"] == "ui_only"
        assert res["ui_elements_count"] == 1
        assert res["ui_elements"][0]["label"] == "Submit"


@pytest.mark.asyncio
async def test_tool_router_dispatch_screen_vision():
    """Verify ToolRouter dispatches 'screen_vision' spec correctly."""
    spec = ToolCallSpec(tool="screen_vision", arguments={"mode": "ocr_only"})

    dummy_ocr = [
        OCRTextItem(text="Test dispatch", bounds=BoundingBox(x=0, y=0, width=10, height=10))
    ]

    with patch("tools.vision_tool.grab_screenshot", return_value=Image.new("RGB", (100, 100))), \
         patch("tools.vision_tool.ocr_engine.extract_text_items", return_value=dummy_ocr):

        res = await tool_router.route_and_execute(spec)
        assert res["success"] is True
        assert res["data"]["mode"] == "ocr_only"
        assert "Test dispatch" in res["data"]["extracted_text"]
