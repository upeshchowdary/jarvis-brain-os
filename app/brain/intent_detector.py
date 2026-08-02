"""Intent Detection and Query Classification Module."""

import re
from app.domain.models.intent import IntentCategory, IntentResult
from app.utils.logger import logger


class IntentDetector:
    """Classifies user queries into domain categories and confidence levels."""

    @staticmethod
    def detect_intent_heuristics(query: str) -> IntentResult:
        """Perform fast heuristic analysis on user query before or alongside LLM reasoning."""
        q_lower = query.lower().strip()

        # Code Assistant Intent
        if any(keyword in q_lower for keyword in ["def ", "class ", "function", "import ", "code", "bug", "refactor", "algorithm", "python"]):
            return IntentResult(
                category=IntentCategory.CODE_ASSISTANT,
                confidence=0.92,
                summary="User requested programming, code writing, or technical debugging.",
            )

        # Task Planning Intent
        if any(keyword in q_lower for keyword in ["plan", "steps", "workflow", "schedule", "roadmap", "how to build", "architecture"]):
            return IntentResult(
                category=IntentCategory.TASK_PLANNING,
                confidence=0.88,
                summary="User requested multi-step workflow planning or architectural design.",
            )

        # Reasoning Analysis Intent
        if any(keyword in q_lower for keyword in ["why", "explain", "analyze", "compare", "evaluate", "math", "reason"]):
            return IntentResult(
                category=IntentCategory.REASONING_ANALYSIS,
                confidence=0.85,
                summary="User requested analytical explanation or logical reasoning.",
            )

        # Tool Execution Intent
        if any(keyword in q_lower for keyword in ["run ", "search ", "fetch ", "file ", "read ", "execute "]):
            return IntentResult(
                category=IntentCategory.TOOL_EXECUTION,
                confidence=0.80,
                summary="User requested tool interaction or system command execution.",
            )

        # Default General Conversation
        return IntentResult(
            category=IntentCategory.GENERAL_CONVERSATION,
            confidence=0.95,
            summary="User engaged in general conversational dialogue.",
        )
