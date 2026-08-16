"""JARVIS Screen Analyzer v3 — Real-Time Vision Pipeline.

Architecture:
  1. Ultra-fast screenshot (DXGI/mss) .............. ~5-15ms
  2. dHash change detection ........................ ~1ms
  3. Cache hit? → instant return ................... ~0ms
  4. Parallel pipeline:
     a. OCR extraction (async, WinRT/Tesseract) ... ~50-150ms
     b. OpenCV analysis (lazy, downsampled) ....... ~20-50ms
     c. Provider RACE (Gemini 2.5 + Groq) ......... ~800-1500ms
  5. First provider wins, cancel the other
  6. Cache result + return

Performance targets:
  Quick mode (chat follow-ups): < 300ms (cache hit)
  Full mode (explicit "look"):  < 1500ms (provider race)

Provider priority chain:
  P1 → Gemini 2.5 Flash        (~1-3s)   Google's fastest multimodal
  P2 → Groq Llama 4 Scout      (~1-3s)   Ultra-fast LPU inference
  P3 → OpenAI gpt-4o-mini      (~2-5s)   Reliable cloud fallback
  P4 → Ollama local             (~10-30s) Offline private fallback
  P5 → OpenCV + OCR             (< 1s)    Emergency offline

Display format:
  👁 VISION: Gemini gemini-2.5-flash [1240ms]
  👁 VISION: Groq llama-4-scout [980ms]
  👁 VISION: Cache hit [instant]
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

from loguru import logger

from brain.brain_config import brain_config

# Vision providers
from vision.providers.gemini import GeminiVisionProvider
from vision.providers.groq_vision import GroqVisionProvider
from vision.providers.ollama_vision import OllamaVisionProvider

# OpenCV preprocessing
from vision.opencv_analyzer import analyze_with_opencv, CVAnalysisResult

# Screen preprocessor (cache, change detection, WebP, app fingerprint)
from vision.screen_preprocessor import (
    preprocess_screen,
    build_enriched_prompt,
    store_cached_analysis,
    ScreenContext,
)

# OCR engine
from vision.ocr import ocr_engine

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# ---------------------------------------------------------------------------
# Fast screenshot capture
# ---------------------------------------------------------------------------

def grab_screenshot() -> Optional[Any]:
    """Capture full desktop using fastest available method."""
    try:
        from vision.fast_capture import capture_full_screen
        img = capture_full_screen()
        if img is not None:
            return img
    except Exception as e:
        logger.debug(f"[JARVIS] fast_capture failed: {e}")

    # PIL fallback
    if HAS_PIL:
        try:
            from PIL import ImageGrab
            return ImageGrab.grab(all_screens=True)
        except Exception:
            try:
                from PIL import ImageGrab
                return ImageGrab.grab()
            except Exception as e:
                logger.warning(f"[JARVIS] PIL.ImageGrab fallback failed: {e}")

    return PILImage.new("RGB", (1920, 1080), color=(30, 30, 35)) if HAS_PIL else None


# ---------------------------------------------------------------------------
# OCR-only emergency fallback
# ---------------------------------------------------------------------------

def _ocr_fallback_description(ctx: ScreenContext, cv_result: CVAnalysisResult) -> str:
    """Build structured description from OCR + OpenCV when all LLMs fail."""
    parts = []

    if ctx.ocr_paragraphs:
        text_lines = "\n".join(f"  • {p[:200]}" for p in ctx.ocr_paragraphs[:10])
        parts.append(f"Visible text on screen:\n{text_lines}")
    if ctx.ocr_code_blocks:
        parts.append(f"Visible code:\n{ctx.ocr_code_blocks[0][:400]}")
    if cv_result.cv_available and cv_result.ui_summary:
        parts.append(f"UI elements detected: {cv_result.ui_summary}")
    if cv_result.cv_available and cv_result.dominant_colors:
        top = cv_result.dominant_colors[0]
        parts.append(f"Screen dominant color: {top.hex_color} ({top.percentage:.0f}%)")

    if not parts:
        return "Screen analysis unavailable — no vision model online and no readable text detected."

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Vision display label builder
# ---------------------------------------------------------------------------

def _vision_label(provider: str, model: str, latency_ms: float, from_cache: bool = False) -> str:
    """Build a human-readable vision status label."""
    if from_cache:
        return "[VISION] Cache hit [instant]"

    provider_display = {
        "claude": "Claude",
        "gemini": "Gemini",
        "groq": "Groq",
        "openai": "OpenAI",
        "ollama": "Ollama (local)",
        "ocr-fallback": "OpenCV+OCR (offline)",
        "cache": "Cache",
    }.get(provider, provider.title())

    short_model = model.split("/")[-1] if "/" in model else model
    return f"[VISION] {provider_display} {short_model} [{latency_ms:.0f}ms]"


# ---------------------------------------------------------------------------
# Concurrent Provider Racing
# ---------------------------------------------------------------------------

async def _race_providers(
    image: Any,
    prompt: str,
    providers: list,
) -> Dict[str, Any]:
    """
    Fire multiple vision providers simultaneously.
    Return the FIRST successful result, cancel the rest.
    """

    async def _try_provider(name: str, provider: Any) -> Dict[str, Any]:
        """Wrapper that tags result with provider name."""
        try:
            result = await provider.analyze_image(image, prompt)
            result["_provider_name"] = name
            return result
        except asyncio.CancelledError:
            return {"success": False, "_provider_name": name, "error": "cancelled"}
        except Exception as e:
            return {"success": False, "_provider_name": name, "error": str(e)}

    # Create tasks for all providers
    tasks = []
    for name, provider in providers:
        task = asyncio.create_task(_try_provider(name, provider))
        tasks.append(task)

    if not tasks:
        return {"success": False, "error": "No providers available"}

    # Wait for the first successful result
    winner = None
    pending = set(tasks)

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            result = task.result()
            if result.get("success") and result.get("analysis"):
                winner = result
                # Cancel remaining tasks
                for remaining in pending:
                    remaining.cancel()
                # Wait for cancellations to complete
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                return winner

    # All failed — return last error
    if tasks:
        return tasks[-1].result()
    return {"success": False, "error": "No providers responded"}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def analyze_screen(
    image: Optional[Any] = None,
    prompt: str = "",
    user_query: Optional[str] = None,
    query_type: str = "SCREEN_DESCRIPTION",
    session_id: str = "default",
    window_title: str = "",
    app_name: str = "",
    force_ocr: bool = False,
    quick_mode: bool = False,
) -> Dict[str, Any]:
    """
    Main JARVIS vision entry point — v3 Real-Time Pipeline.

    Args:
        quick_mode: If True, skip expensive analysis for chat follow-ups.

    Returns dict with keys:
      success, provider, model, query_type, description,
      vision_label, latency_ms, preprocessing_ms, cv_analysis
    """
    total_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("[JARVIS] [VISION] PIPELINE v3 STARTED")
    logger.info(f"[JARVIS] Query type: {query_type} | quick_mode: {quick_mode}")

    # ── Step 1: Screenshot (~5-15ms) ──────────────────────────────
    if image is None:
        image = grab_screenshot()

    if image is None:
        return _error_result("Screenshot capture failed.", query_type)

    w, h = image.size if hasattr(image, "size") else (1920, 1080)
    capture_ms = (time.perf_counter() - total_start) * 1000
    logger.info(f"[JARVIS] Screenshot: {w}×{h} in {capture_ms:.0f}ms")

    # ── Step 2: Preprocessor (dHash + cache check) (~1-3ms) ─────
    ctx = preprocess_screen(
        screenshot=image,
        session_id=session_id,
        window_title=window_title,
        app_name=app_name,
        query_type=query_type,
        max_image_size=brain_config.VISION_IMAGE_SIZE,
        cache_ttl=brain_config.VISION_CACHE_TTL,
    )

    # ── Step 3: Cache hit? → instant return ──────────────────────
    if ctx.is_cache_valid and ctx.cached_analysis:
        elapsed = round((time.perf_counter() - total_start) * 1000, 1)
        label = _vision_label("cache", "cached", elapsed, from_cache=True)
        logger.info(f"[JARVIS] ✅ {label} ({elapsed:.0f}ms)")
        return {
            "success": True,
            "provider": "cache",
            "model": "cached",
            "query_type": query_type,
            "description": ctx.cached_analysis,
            "vision_label": label,
            "latency_ms": elapsed,
            "from_cache": True,
            "preprocessing_ms": ctx.preprocessing_ms,
            "cv_ui_elements": 0,
        }

    if force_ocr:
        cv_result = analyze_with_opencv(image, session_id=session_id)
        # Run OCR via engine
        ocr_raw, ocr_paras, ocr_code = ocr_engine.extract_structured(image)
        ctx.ocr_text = ocr_raw
        ctx.ocr_paragraphs = ocr_paras
        ctx.ocr_code_blocks = ocr_code
        ctx.has_readable_text = bool(ocr_paras or ocr_code)
        desc = _ocr_fallback_description(ctx, cv_result)
        elapsed = round((time.perf_counter() - total_start) * 1000, 1)
        label = _vision_label("ocr-fallback", "OCR", elapsed)
        return _build_result(False, "ocr-fallback", "OCR", query_type, desc, label, elapsed, ctx, cv_result)

    # ── Step 4: Parallel preprocessing ───────────────────────────
    # Run OCR + OpenCV concurrently while building prompt
    loop = asyncio.get_running_loop()

    async def _run_ocr():
        return await loop.run_in_executor(
            None, ocr_engine.extract_structured, image,
        )

    async def _run_cv():
        return await loop.run_in_executor(
            None, analyze_with_opencv, image, session_id, True,
        )

    ocr_task = asyncio.create_task(_run_ocr())
    cv_task = asyncio.create_task(_run_cv())

    # Wait for both to complete
    (ocr_raw, ocr_paras, ocr_code), cv_result = await asyncio.gather(ocr_task, cv_task)

    # Populate context with OCR results
    ctx.ocr_text = ocr_raw
    ctx.ocr_paragraphs = ocr_paras
    ctx.ocr_code_blocks = ocr_code
    ctx.has_readable_text = bool(ocr_paras or ocr_code)

    # Use CLAHE-enhanced image if screen is dark
    vision_image = cv_result.enhanced_image if cv_result.enhanced_image is not None else image

    # ── Step 5: Build enriched prompt ────────────────────────────
    effective_query = user_query or "Describe exactly what is visible on the screen."

    cv_context = cv_result.to_prompt_text()
    enriched_prompt = build_enriched_prompt(
        ctx=ctx,
        user_query=effective_query,
        query_type=query_type,
    )
    if cv_context:
        enriched_prompt = enriched_prompt + f"\n\n[CV PRE-ANALYSIS]:\n{cv_context}"

    # ── Step 6: Provider Racing (Concurrent) ──────────────────────
    if brain_config.VISION_RACE_PROVIDERS:
        logger.info("[JARVIS] Racing providers: Gemini 2.5 Flash (fast mode)")

        race_providers = []

        # P1: Gemini 3.5 Flash Lite — ultra-fast sub-1s multimodal vision model
        if brain_config.GEMINI_API_KEY:
            gemini = GeminiVisionProvider(
                api_key=brain_config.GEMINI_API_KEY,
                model_name="gemini-3.5-flash-lite",
                timeout=5.0,
            )
            race_providers.append(("gemini", gemini))

        # P2: Local Ollama (qwen3-vl:2b) — ultra-fast local offline backup
        try:
            ollama_local = OllamaVisionProvider(timeout=8.0)
            race_providers.append(("ollama", ollama_local))
        except Exception:
            pass

        logger.info(f"[JARVIS] Racing providers: {[name for name, _ in race_providers]}")

        if race_providers:
            res = await _race_providers(vision_image, enriched_prompt, race_providers)

            if res.get("success") and res.get("analysis"):
                analysis = res["analysis"]
                provider_name = res.get("_provider_name", res.get("provider", "unknown"))
                store_cached_analysis(session_id, ctx.image_hash, analysis)
                elapsed = round((time.perf_counter() - total_start) * 1000, 1)
                label = _vision_label(provider_name, res.get("model", ""), elapsed)
                logger.info(f"[JARVIS] [OK] RACE WINNER: {label}")
                return _build_result(
                    True, provider_name, res.get("model", ""),
                    query_type, analysis, label, elapsed, ctx, cv_result,
                )

            logger.info(f"[JARVIS] Race failed: {res.get('error', '')[:80]}")
    else:
        # Sequential fallback (legacy behavior when VISION_RACE_PROVIDERS=False)
        # P1: Gemini 3.5 Flash Lite
        if brain_config.GEMINI_API_KEY:
            logger.info("[JARVIS] P1: Trying Gemini 3.5 Flash Lite...")
            gemini = GeminiVisionProvider(
                api_key=brain_config.GEMINI_API_KEY,
                model_name="gemini-3.5-flash-lite",
            )
            res = await gemini.analyze_image(vision_image, enriched_prompt)
            if res.get("success") and res.get("analysis"):
                analysis = res["analysis"]
                store_cached_analysis(session_id, ctx.image_hash, analysis)
                elapsed = round((time.perf_counter() - total_start) * 1000, 1)
                label = _vision_label("gemini", res["model"], elapsed)
                logger.info(f"[JARVIS] [OK] {label}")
                return _build_result(True, "gemini", res["model"], query_type, analysis, label, elapsed, ctx, cv_result)

    # ── Step 7: Ollama local fallback ────────────────────────────
    logger.info("[JARVIS] P3: Trying Ollama local vision model...")
    ollama_vis = OllamaVisionProvider()
    res = await ollama_vis.analyze_image(vision_image, enriched_prompt)

    if res.get("success") and res.get("analysis"):
        analysis = res["analysis"]
        store_cached_analysis(session_id, ctx.image_hash, analysis)
        elapsed = round((time.perf_counter() - total_start) * 1000, 1)
        label = _vision_label("ollama", res.get("model", brain_config.OLLAMA_VISION_MODEL), elapsed)
        logger.info(f"[JARVIS] [OK] {label}")
        return _build_result(True, "ollama", res.get("model", "qwen3-vl:2b"), query_type, analysis, label, elapsed, ctx, cv_result)

    # ── Step 9: Emergency OCR fallback ───────────────────────────
    logger.warning("[JARVIS] P5: All vision LLMs failed. Using OpenCV+OCR fallback.")
    desc = _ocr_fallback_description(ctx, cv_result)
    elapsed = round((time.perf_counter() - total_start) * 1000, 1)
    label = _vision_label("ocr-fallback", "OCR", elapsed)
    logger.info(f"[JARVIS] [WARN] {label}")
    return _build_result(False, "ocr-fallback", "OCR", query_type, desc, label, elapsed, ctx, cv_result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error_result(msg: str, query_type: str) -> Dict[str, Any]:
    return {
        "success": False,
        "provider": "none",
        "model": "none",
        "query_type": query_type,
        "description": msg,
        "vision_label": f"👁 VISION: ERROR — {msg}",
        "latency_ms": 0,
        "preprocessing_ms": 0,
        "cv_ui_elements": 0,
    }


def _build_result(
    success: bool,
    provider: str,
    model: str,
    query_type: str,
    description: str,
    vision_label: str,
    latency_ms: float,
    ctx: ScreenContext,
    cv_result: CVAnalysisResult,
) -> Dict[str, Any]:
    return {
        "success": success,
        "provider": provider,
        "model": model,
        "query_type": query_type,
        "description": description,
        "vision_label": vision_label,
        "latency_ms": latency_ms,
        "preprocessing_ms": ctx.preprocessing_ms,
        "cv_ui_elements": len(cv_result.ui_elements),
        "cv_dominant_color": cv_result.dominant_colors[0].hex_color if cv_result.dominant_colors else "",
        "cv_brightness": round(cv_result.brightness, 1),
        "ocr_paragraphs": ctx.ocr_paragraphs[:3],
        "app_type": ctx.app_type,
    }
