"""Reasoning Engine Module for breaking queries into logical steps."""

from typing import List, Dict, Any


class ReasoningEngine:
    """Processes raw LLM thinking outputs into structured step-by-step reasoning chains."""

    @staticmethod
    def extract_reasoning_chain(raw_reasoning: Any) -> List[str]:
        """Normalize raw reasoning inputs into clean string array format."""
        if isinstance(raw_reasoning, list):
            return [str(step).strip() for step in raw_reasoning if str(step).strip()]
        elif isinstance(raw_reasoning, str) and raw_reasoning.strip():
            # Split by line breaks or numbers if formatted as block text
            lines = raw_reasoning.strip().split("\n")
            return [line.strip("- *0123456789. ") for line in lines if line.strip()]
        return ["Analyzed query intent and context.", "Formulated response based on system knowledge."]
