"""Screen Vision Tool for JARVIS Brain Tool System.

Allows JARVIS Brain to trigger visual perception, desktop screenshot capture,
OCR text reading, and UI element analysis as a registered tool.
"""

from typing import Dict, Any, Optional
from brain.tool_router import BaseBrainTool
from brain.logger import logger
from vision.manager import vision_manager
from vision.screen_analyzer import analyze_screen, grab_screenshot
from vision.ocr import ocr_engine
from vision.ui_detector import ui_detector


class ScreenVisionTool(BaseBrainTool):
    """Tool for capturing and analyzing desktop screen visual content, text, and UI components."""

    name: str = "screen_vision"
    description: str = "Performs visual analysis, OCR text extraction, and UI element detection on the user's active desktop screen."
    version: str = "1.0.0"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Question or prompt regarding the visual content on screen.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["full", "ocr_only", "ui_only"],
                    "default": "full",
                    "description": "Perception mode: 'full' (all components + LLM), 'ocr_only' (fast text extraction), 'ui_only' (GUI element bounds).",
                },
                "source": {
                    "type": "string",
                    "enum": ["screen", "camera"],
                    "default": "screen",
                    "description": "Visual input source.",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> Dict[str, Any]:
        query = kwargs.get("query", "Describe what is visible on the desktop screen.")
        mode = kwargs.get("mode", "full")
        source = kwargs.get("source", "screen")

        logger.info(f"ScreenVisionTool executing mode='{mode}' source='{source}' query='{query[:50]}'")

        try:
            if mode == "ocr_only":
                image = grab_screenshot()
                if image is None:
                    return {"success": False, "error": "Screen capture failed."}
                ocr_items = ocr_engine.extract_text_items(image)
                extracted_lines = [item.text for item in ocr_items]
                return {
                    "success": True,
                    "mode": "ocr_only",
                    "extracted_text": "\n".join(extracted_lines),
                    "items_count": len(ocr_items),
                    "ocr_items": [item.model_dump() for item in ocr_items[:50]],
                }

            elif mode == "ui_only":
                image = grab_screenshot()
                if image is None:
                    return {"success": False, "error": "Screen capture failed."}
                ocr_items = ocr_engine.extract_text_items(image)
                ui_elements = ui_detector.detect_ui_elements(image, ocr_items=ocr_items)
                return {
                    "success": True,
                    "mode": "ui_only",
                    "ui_elements_count": len(ui_elements),
                    "ui_elements": [elem.model_dump() for elem in ui_elements[:50]],
                }

            else:  # full mode
                env_state = await vision_manager.capture_environment_state(
                    source=source,
                    include_visual_reasoning=True,
                )
                structured_data = {
                    "source": source,
                    "timestamp": env_state.timestamp,
                    "active_application": env_state.active_application,
                    "active_window": env_state.active_window,
                    "screen": {
                        "width": env_state.monitors[0].resolution[0] if env_state.monitors else 1920,
                        "height": env_state.monitors[0].resolution[1] if env_state.monitors else 1080,
                    },
                    "text": [item.model_dump() for item in env_state.visible_text[:30]],
                    "ui_elements": [elem.model_dump() for elem in env_state.ui[:30]],
                    "objects": [obj.model_dump() for obj in env_state.objects[:20]],
                    "description": env_state.summary,
                    "confidence": env_state.confidence,
                }
                return {
                    "success": True,
                    "mode": "full",
                    "data": structured_data,
                }

        except Exception as exc:
            logger.error(f"ScreenVisionTool execution error: {exc}")
            return {"success": False, "error": str(exc)}
