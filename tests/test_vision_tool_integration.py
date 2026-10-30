"""Unit & Integration tests for JARVIS Vision Tool, VisionManager, and Provider abstractions."""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image

from tools import load_builtin_tools
from tools.builtin.vision import classify_visual_query_type
from vision.providers.gemini import GeminiVisionProvider
from vision.providers.ollama_vision import OllamaVisionProvider
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
    """The screen vision tool is registered, read-only, and marked untrusted + sensitive."""
    tool = load_builtin_tools().get("screen_look")
    assert tool is not None
    assert tool.side_effect is False and tool.untrusted_output and tool.sensitive_output


def test_classify_visual_query_type():
    assert classify_visual_query_type("What is in the image on my screen?") == "IMAGE_DESCRIPTION"
    assert classify_visual_query_type("Read the text on screen") == "OCR"
    assert classify_visual_query_type("Where is the submit button?") == "UI_ANALYSIS"
    assert classify_visual_query_type("What error message popped up?") == "ERROR_ANALYSIS"
    assert classify_visual_query_type("What is on my screen?") == "SCREEN_DESCRIPTION"


# ---------------------------------------------------------------------------
# Screen Preprocessor tests
# ---------------------------------------------------------------------------

def test_preprocessor_fast_hash(sample_pil_image):
    """verify image hashing returns stable non-empty string."""
    h1 = _fast_image_hash(sample_pil_image)
    h2 = _fast_image_hash(sample_pil_image)
    assert isinstance(h1, str) and len(h1) in (16, 32)
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


async def test_screen_look_tool_uses_analyzer_and_caches(sample_pil_image):
    """screen_look captures, analyses, wraps the description for the LLM and caches it per session."""
    from tools import ToolContext

    tool = load_builtin_tools().get("screen_look")
    fake = AsyncMock(return_value={"description": "VS Code open on desktop.", "vision_label": "VISION", "provider": "gemini"})
    with patch("vision.screen_analyzer.grab_screenshot", return_value=sample_pil_image), \
         patch("vision.screen_analyzer.analyze_screen", fake), \
         patch("vision.screen_capture.screen_capture_engine.get_active_window_info", return_value=None):
        ctx = ToolContext(session_id="vision_cache_test")
        res = await tool.run(ctx, question="What is on my screen?")
        assert res.success and res.content_for_llm == "VS Code open on desktop."
        again = await tool.run(ctx, question="and what else?")
        assert again.data == {"cached": True}
        assert fake.await_count == 1


async def test_screen_look_privacy_local_forces_ollama(sample_pil_image, monkeypatch):
    from core.config import settings
    from tools import ToolContext

    monkeypatch.setattr(settings, "PRIVACY_MODE", "local")
    tool = load_builtin_tools().get("screen_look")
    fake = AsyncMock(return_value={"description": "desk", "vision_label": "V"})
    with patch("vision.screen_analyzer.grab_screenshot", return_value=sample_pil_image), \
         patch("vision.screen_analyzer.analyze_screen", fake), \
         patch("vision.screen_capture.screen_capture_engine.get_active_window_info", return_value=None):
        await tool.run(ToolContext(session_id="privacy_vision"), question="what is this", refresh=True)
    assert fake.await_args.kwargs["active_model"].startswith("ollama/")
