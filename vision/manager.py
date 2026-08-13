"""JARVIS Central Vision Manager v3 — Lazy Perception Orchestrator.

Main entry-point for the JARVIS Vision & Environment Understanding Engine.

v3 improvements:
  - Lazy perception: only run detectors that are actually needed
  - Configurable depth: quick (screen + OCR), standard (+UI), deep (+objects, faces, gestures)
  - Parallel async preprocessing
  - Uses real-time screen_analyzer v3 with provider racing
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from vision.environment import EnvironmentState, CursorState, WindowInfo, MonitorInfo
from vision.screen_capture import screen_capture_engine
from vision.ocr import ocr_engine
from vision.screen_memory import screen_memory
from vision.diff_detector import diff_detector
from vision.screen_analyzer import analyze_screen, grab_screenshot


class VisionManager:
    """Central orchestrator for JARVIS Vision & Environment Perception Engine.

    Perception depth levels:
      quick    — screenshot + screen_analyzer only (fastest, for chat follow-ups)
      standard — + OCR + UI elements (default)
      deep     — + objects + faces + gestures + layout (full analysis)
    """

    async def capture_environment_state(
        self,
        source: str = "screen",
        include_visual_reasoning: bool = True,
        force_refresh: bool = False,
        depth: str = "standard",
    ) -> EnvironmentState:
        """
        Captures environmental perception at the requested depth.

        Args:
            source: "screen" or "camera"
            include_visual_reasoning: whether to run LLM vision analysis
            force_refresh: bypass cache
            depth: "quick", "standard", or "deep"
        """
        start_time = time.perf_counter()

        # 1. Image Acquisition
        image = None
        if source.lower() == "camera":
            try:
                from vision.camera import camera_module
                image, _ = await camera_module.capture_frame_async()
            except Exception as e:
                logger.debug(f"Camera capture failed: {e}")
        else:
            image = grab_screenshot()
            if image is None:
                image = screen_capture_engine.capture_full_desktop()

        if image is None:
            return EnvironmentState(summary="Visual acquisition unavailable.", confidence=0.0)

        # 2. Screen Memory Hash Cache
        img_hash = screen_memory.compute_image_hash(image)
        if not force_refresh and img_hash:
            cached_state = screen_memory.get_cached_state(img_hash)
            if cached_state is not None:
                logger.info(f"VisionManager returning cached state for hash '{img_hash[:8]}'.")
                return cached_state

        # 3. System telemetry (always fast)
        cursor = screen_capture_engine.get_cursor_position()
        monitors = screen_capture_engine.get_monitors_info()
        active_win = screen_capture_engine.get_active_window_info()

        # 4. Depth-based perception
        ocr_items = []
        ui_elements = []
        objects = []
        faces = []
        gestures = []
        windows = []

        if depth in ("standard", "deep"):
            # Get window list + OCR in parallel
            loop = asyncio.get_running_loop()

            async def _get_windows():
                return await loop.run_in_executor(
                    None, screen_capture_engine.get_all_open_windows
                )

            async def _get_ocr():
                return await ocr_engine.extract_text_items_async(image)

            windows_task = asyncio.create_task(_get_windows())
            ocr_task = asyncio.create_task(_get_ocr())

            windows, ocr_items = await asyncio.gather(windows_task, ocr_task)

        if depth == "deep":
            # Full analysis — run heavy detectors in parallel
            loop = asyncio.get_running_loop()

            try:
                from vision.ui_detector import ui_detector
                from vision.image_analyzer import image_analyzer
                from vision.object_detector import object_detector
                from vision.face_detector import face_detector
                from vision.gesture_detector import gesture_detector
                from vision.layout_analyzer import layout_analyzer

                ui_task = loop.run_in_executor(
                    None, lambda: ui_detector.detect_ui_elements(image, ocr_items=ocr_items)
                )
                objects_task = loop.run_in_executor(
                    None, lambda: object_detector.detect_objects(image)
                )
                faces_task = loop.run_in_executor(
                    None, lambda: face_detector.detect_faces(image)
                )
                gestures_task = loop.run_in_executor(
                    None, lambda: gesture_detector.detect_gestures(image)
                )

                ui_elements, objects, faces, gestures = await asyncio.gather(
                    ui_task, objects_task, faces_task, gestures_task,
                )
            except Exception as e:
                logger.debug(f"Deep perception detectors error: {e}")

        elif depth == "standard":
            # Standard: just UI detection
            try:
                from vision.ui_detector import ui_detector
                loop = asyncio.get_running_loop()
                ui_elements = await loop.run_in_executor(
                    None, lambda: ui_detector.detect_ui_elements(image, ocr_items=ocr_items)
                )
            except Exception as e:
                logger.debug(f"UI detection error: {e}")

        # 5. Visual reasoning via LLM
        summary_text = (
            f"Active App: {active_win.app_name} ('{active_win.title}'). "
        )

        if include_visual_reasoning:
            vision_result = await analyze_screen(
                image=image,
                window_title=active_win.title,
                app_name=active_win.app_name,
            )
            if vision_result.get("success"):
                description = vision_result.get("description", "")
                provider = vision_result.get("provider", "unknown")
                latency = vision_result.get("latency_ms", 0)
                logger.info(
                    f"VisionManager: vision via [{provider}] in {latency:.0f}ms"
                )
                summary_text = description
            else:
                logger.warning("VisionManager: vision LLM unavailable, using fallback summary.")

        # 6. Frame diff
        prev_snapshot = screen_memory.get_last_snapshot()

        # 7. Build EnvironmentState
        state = EnvironmentState(
            active_application=active_win.app_name,
            active_window=active_win.title,
            cursor=cursor,
            monitors=monitors,
            windows=windows or [active_win],
            ui=ui_elements,
            visible_text=ocr_items,
            objects=objects,
            faces=faces,
            gestures=gestures,
            summary=summary_text,
            confidence=0.98,
        )

        state.diff = diff_detector.compute_diff(prev_snapshot, state)

        # 8. Cache
        if img_hash:
            screen_memory.add_snapshot(img_hash, state)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"VisionManager perception cycle [{depth}] completed in {elapsed_ms:.1f}ms."
        )
        return state


# Global singleton instance
vision_manager = VisionManager()
