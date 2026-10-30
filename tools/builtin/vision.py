"""Screen vision tool: looks at the user's screen when they ask about it."""

from __future__ import annotations

from typing import Any

from core.config import settings
from core.logging import logger
from core.session import sessions
from tools.base import Tool, ToolContext, ToolResult, obj
from tools.registry import register


def classify_visual_query_type(query: str) -> str:
    """Classify a visual question into the analyzer's sub-categories."""
    q = query.lower().strip()
    if any(w in q for w in ("read text", "read the text", "visible text", "extract text", "read paragraph")):
        return "OCR"
    if any(w in q for w in ("error", "exception", "traceback", "crash", "warning message", "stack trace")):
        return "ERROR_ANALYSIS"
    if any(w in q for w in ("button", "submit", "checkbox", "menu", "where is", "click", "text box", "input field")):
        return "UI_ANALYSIS"
    if any(w in q for w in ("image", "photo", "picture", "dog", "cat", "animal", "object", "item in picture")):
        return "IMAGE_DESCRIPTION"
    if any(w in q for w in ("document", "pdf", "table", "chart", "graph")):
        return "DOCUMENT_ANALYSIS"
    if any(w in q for w in ("what is on my screen", "describe my screen", "see my screen", "whats on screen", "entire screen")):
        return "SCREEN_DESCRIPTION"
    return "VISUAL_QUESTION"


@register
class ScreenLook(Tool):
    name = "screen_look"
    description = (
        "Look at the user's screen right now (screenshot + OCR + vision model) to answer questions about what is "
        "visible: errors, text, UI elements, images. Only use when the user asks about their screen."
    )
    parameters = obj({
        "question": {"type": "string", "description": "What the user wants to know about the screen"},
        "refresh": {"type": "boolean", "description": "Force a new capture instead of reusing the last one", "default": False},
    }, ["question"])
    untrusted_output = True
    sensitive_output = True
    final = False

    async def run(self, ctx: ToolContext, question: str = "Describe the screen.", refresh: bool = False, **_: Any) -> ToolResult:
        state = sessions.get(ctx.session_id)
        cached = state.recent_vision(max_age=60)
        if cached and not refresh:
            return ToolResult.ok("Using the screen analysis from a moment ago.", data={"cached": True}, content_for_llm=cached)
        try:
            from vision.screen_analyzer import analyze_screen, grab_screenshot
            from vision.screen_capture import screen_capture_engine

            screenshot = grab_screenshot() or screen_capture_engine.capture_full_desktop()
            if screenshot is None:
                return ToolResult.error("I couldn't capture the screen.")
            win = screen_capture_engine.get_active_window_info()
            model = state.model_override or settings.DEFAULT_MODEL
            if settings.PRIVACY_MODE == "local":
                model = f"ollama/{settings.OLLAMA_VISION_MODEL}"  # screenshots never leave the PC
            result = await analyze_screen(
                image=screenshot,
                user_query=question,
                query_type=classify_visual_query_type(question),
                session_id=ctx.session_id,
                window_title=win.title if win else "",
                app_name=win.app_name if win else "",
                active_model=model.split("/", 1)[-1] if model.startswith("gemini/") else model,
            )
        except Exception as exc:
            logger.warning(f"[screen_look] {exc}")
            return ToolResult.error(f"Screen vision is unavailable right now ({exc}).")
        description = result.get("description", "")
        if not description:
            return ToolResult.error("The vision model returned nothing.")
        state.set_vision(description)
        return ToolResult.ok(result.get("vision_label", "Analysed the screen."), data={"provider": result.get("provider")}, content_for_llm=description)
