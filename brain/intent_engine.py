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
    def detect_intent(query: str) -> StructuredIntent:
        """Classify intent using heuristic patterns and entity extraction."""
        q_lower = query.lower().strip()

        # 0. Model Switch / Provider Change Intent
        switch_match = re.search(
            r"\b(?:switch|change|set|use|select)\s+(?:model\s+|provider\s+)?(?:to\s+)?(ollama|gemini|groq|llama[a-zA-Z0-9_\-\.]*|qwen[a-zA-Z0-9_\-\.]*|mixtral[a-zA-Z0-9_\-\.]*|gemma[a-zA-Z0-9_\-\.]*)",
            q_lower,
        )
        if switch_match:
            target_raw = switch_match.group(1).strip()
            target_model = target_raw
            if target_raw == "ollama":
                target_model = f"ollama/{brain_config.OLLAMA_MODEL}"
            elif target_raw == "gemini":
                target_model = "gemini-1.5-flash"
            elif target_raw == "groq":
                target_model = "llama-3.3-70b-versatile"

            return StructuredIntent(
                intent="MODEL_SWITCH",
                confidence=0.98,
                arguments={"target_model": target_model, "raw_query": query},
                summary=f"User requested switching active model to '{target_model}'",
            )

        # 1. Real-Time / Internet Knowledge Search Intent
        realtime_triggers = [
            "today", "present", "right now", "latest", "recent", "news",
            "current", "search internet", "search web", "live", "price",
            "weather", "who won", "score", "update", "2026", "2025", "2024",
            "what is happening", "what happened"
        ]
        if any(w in q_lower for w in realtime_triggers):
            return StructuredIntent(
                intent="REALTIME_KNOWLEDGE_SEARCH",
                confidence=0.95,
                arguments={"search_query": query},
                summary="User requested real-time live internet information or current web search.",
            )

        # 2. System Info / Telemetry Intent
        if any(w in q_lower for w in ["cpu", "ram", "memory usage", "system info", "os version", "disk space", "hardware status"]):
            return StructuredIntent(
                intent="SYSTEM_TELEMETRY",
                confidence=0.95,
                arguments={"query": query},
                summary="User requested system metrics or OS telemetry.",
            )

        # 3. File System Operation Intent
        if any(w in q_lower for w in ["read file", "write file", "list directory", "list files", "save to file"]):
            return StructuredIntent(
                intent="FILESYSTEM_OPERATION",
                confidence=0.92,
                arguments={"query": query},
                summary="User requested file or directory operation.",
            )

        # 4. Open Application Intent
        open_app_match = re.search(r"\b(?:open|launch|start|run)\s+([a-zA-Z0-9_\-\s]+)", q_lower)
        if open_app_match and not any(w in q_lower for w in ["code", "script", "file", "url", "browser"]):
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
