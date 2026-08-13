"""Intent Engine for classifying structured user intents and extracted parameters."""

import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from brain.brain_config import brain_config


class StructuredIntent(BaseModel):
    intent: str = Field(..., description="High-level intent classification code")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters or target arguments")
    summary: str = Field(..., description="Human readable summary of detected intent")


class IntentEngine:
    """Analyzes user inputs to determine structured intent and arguments."""

    @staticmethod
    def classify_visual_query_type(query: str) -> str:
        """Classify visual query into sub-categories."""
        q = query.lower().strip()
        if any(w in q for w in ["read text", "read the text", "visible text", "extract text", "read paragraph"]):
            return "OCR"
        if any(w in q for w in ["error", "exception", "traceback", "crash", "warning message", "stack trace"]):
            return "ERROR_ANALYSIS"
        if any(w in q for w in ["button", "submit", "checkbox", "menu", "where is", "click", "text box", "input field"]):
            return "UI_ANALYSIS"
        if any(w in q for w in ["image", "photo", "picture", "dog", "cat", "animal", "object", "item in picture"]):
            return "IMAGE_DESCRIPTION"
        if any(w in q for w in ["what animal", "what object", "who is in", "recognize object"]):
            return "OBJECT_RECOGNITION"
        if any(w in q for w in ["document", "pdf", "table", "chart", "graph"]):
            return "DOCUMENT_ANALYSIS"
        if any(w in q for w in ["what is on my screen", "describe my screen", "see my screen", "whats on screen", "entire screen"]):
            return "SCREEN_DESCRIPTION"
        return "VISUAL_QUESTION"

    @staticmethod
    def detect_intent(query: str) -> StructuredIntent:
        """Classify intent using heuristic patterns and entity extraction."""
        q_lower = query.lower().strip()

        # 0. Model Switch / Provider Change Intent
        switch_match = re.search(
            r"\b(?:switch|change|set|use|select)\s+(?:model\s+|provider\s+)?(?:to\s+)?(ollama|gemini|groq|openai|gpt[a-zA-Z0-9_\-\.]*|llama[a-zA-Z0-9_\-\.]*|qwen[a-zA-Z0-9_\-\.]*|mixtral[a-zA-Z0-9_\-\.]*|gemma[a-zA-Z0-9_\-\.]*)",
            q_lower,
        )
        if switch_match:
            target_raw = switch_match.group(1).strip()
            target_model = target_raw
            if target_raw == "ollama":
                target_model = f"ollama/{brain_config.OLLAMA_MODEL}"
            elif target_raw == "gemini":
                target_model = "gemini-2.5-flash"
            elif target_raw == "groq":
                target_model = "llama-3.3-70b-versatile"
            elif target_raw in ("openai", "gpt"):
                target_model = "gpt-4o-mini"

            return StructuredIntent(
                intent="MODEL_SWITCH",
                confidence=0.98,
                arguments={"target_model": target_model, "raw_query": query},
                summary=f"User requested switching active model to '{target_model}'",
            )

        # 1. Refresh Vision Intent
        refresh_triggers = [
            "look again", "refresh vision", "analyze again", "analyze the screen again",
            "re-analyze screen", "reanalyze screen", "check screen again", "look at screen again",
            "what is on my screen right now", "take another look", "refresh screen"
        ]
        if any(w in q_lower for w in refresh_triggers):
            v_type = IntentEngine.classify_visual_query_type(query)
            return StructuredIntent(
                intent="SCREEN_VISION",
                confidence=0.99,
                arguments={"query": query, "visual_query_type": v_type, "force_refresh": True},
                summary=f"User explicitly requested fresh screen analysis ({v_type}).",
            )

        # 1b. Screen Vision & Visual Perception Intent
        vision_triggers = [
            # Direct screen/display references
            "screen", "display", "monitor", "desktop",
            # What-do-you-see patterns
            "can you see", "what do you see", "what can you see", "see my", "look at my",
            "whats on my", "what is on my", "what's on my", "what is on screen",
            "on my screen", "in my screen", "on the screen",
            # Image/visual content on screen
            "image on screen", "picture on screen", "photo on screen", "image on my screen",
            "picture on my screen", "what is the image", "what's the image", "what image",
            "what animal", "what object", "what is in the image", "whats in the image",
            "describe the image", "describe what you see", "describe my screen",
            # App / window detection
            "active window", "focused window", "apps opened", "open windows", "open apps",
            "taskbar", "task bar", "which app", "which application", "which window",
            "what app", "what application", "what window", "apps in taskbar",
            "apps are open", "what's open", "whats open",
            # Specific screen content
            "background colour", "background color", "background color of",
            "colour of the screen", "color of the screen",
            "what text", "text on screen", "visible text",
            "whatsapp", "chrome", "vs code", "vscode", "code editor", "terminal window",
            # Cursor / position
            "cursor position", "where is the cursor", "where is my cursor",
            # Change detection
            "what changed", "what's different", "screenshot", "screen capture",
            # General visual queries (removed 'visual'/'visually' — too broad, triggers on non-vision queries)
            "see the", "see it", "look at",
        ]
        if any(w in q_lower for w in vision_triggers):
            v_type = IntentEngine.classify_visual_query_type(query)
            return StructuredIntent(
                intent="SCREEN_VISION",
                confidence=0.98,
                arguments={"query": query, "visual_query_type": v_type, "force_refresh": False},
                summary=f"User requested visual perception ({v_type}) of active desktop.",
            )

        # 2. Real-Time / Internet Knowledge Search Intent
        realtime_triggers = [
            "today's", "right now", "latest news", "recent news",
            "search internet", "search the web", "search web", "look up online",
            "live price", "current price", "price of", "stock price",
            "weather", "who won", "score", "live update", "breaking news",
            "what is happening", "what happened today", "news today",
        ]
        if any(w in q_lower for w in realtime_triggers):
            return StructuredIntent(
                intent="REALTIME_KNOWLEDGE_SEARCH",
                confidence=0.95,
                arguments={"search_query": query},
                summary="User requested real-time live internet information or current web search.",
            )

        # 3. System Info / Telemetry Intent
        if any(w in q_lower for w in ["cpu", "ram", "memory usage", "system info", "os version", "disk space", "hardware status"]):
            return StructuredIntent(
                intent="SYSTEM_TELEMETRY",
                confidence=0.95,
                arguments={"query": query},
                summary="User requested system metrics or OS telemetry.",
            )

        # 4. File System Operation Intent
        if any(w in q_lower for w in ["read file", "write file", "list directory", "list files", "save to file"]):
            return StructuredIntent(
                intent="FILESYSTEM_OPERATION",
                confidence=0.92,
                arguments={"query": query},
                summary="User requested file or directory operation.",
            )

        # 5. Open Application Intent — ONLY for explicit launch commands, NOT screen queries
        open_app_match = re.search(r"\b(?:open|launch|start|run)\s+([a-zA-Z0-9_\-\s]+)", q_lower)
        if open_app_match and not any(w in q_lower for w in [
            "code", "script", "file", "url", "browser", "screen", "taskbar", "window", "app", "image"
        ]):
            app_name = open_app_match.group(1).strip()
            return StructuredIntent(
                intent="OPEN_APPLICATION",
                confidence=0.95,
                arguments={"application_name": app_name},
                summary=f"User requested to launch application: '{app_name}'",
            )

        # 5. Open Web / Browser Intent
        if any(w in q_lower for w in ["browser", "website", "http", "www.", ".com", ".org"]):
            url_match = re.search(r"https?://[^\s]+", query)
            target_url = url_match.group(0) if url_match else query
            return StructuredIntent(
                intent="BROWSER_NAVIGATION",
                confidence=0.90,
                arguments={"target": target_url},
                summary="User requested browser navigation or web page scraping",
            )

        # 6. Coding Request Intent
        if any(w in q_lower for w in ["def ", "class ", "write code", "fix bug", "function", "refactor", "python", "script"]):
            return StructuredIntent(
                intent="CODE_GENERATION",
                confidence=0.92,
                arguments={"task": query},
                summary="User requested software development or code generation",
            )

        # 7. Planning Request Intent
        if any(w in q_lower for w in ["plan", "roadmap", "steps", "how to build", "architecture", "workflow"]):
            return StructuredIntent(
                intent="TASK_PLANNING",
                confidence=0.88,
                arguments={"goal": query},
                summary="User requested multi-step execution plan or roadmap",
            )

        # 8. Knowledge Explanation Request
        if any(w in q_lower for w in ["explain", "what is", "why does", "how does", "compare", "define", "describe"]):
            return StructuredIntent(
                intent="KNOWLEDGE_REQUEST",
                confidence=0.90,
                arguments={"topic": query},
                summary="User requested analytical knowledge explanation",
            )

        # Default General Dialogue
        return StructuredIntent(
            intent="GENERAL_CONVERSATION",
            confidence=0.98,
            arguments={"query": query},
            summary="General conversational interaction",
        )


# Global intent engine singleton
intent_engine = IntentEngine()
