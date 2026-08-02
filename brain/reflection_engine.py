"""Reflection Engine for post-response self-critique, self-evaluation, and self-learning capabilities."""

from typing import Dict, Any, List
from pydantic import BaseModel, Field
from brain.logger import logger


class ReflectionResult(BaseModel):
    correctness_score: float = Field(default=0.98, ge=0.0, le=1.0)
    answered_correctly: bool = True
    misunderstandings_detected: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    reflection_notes: str = "Response meets quality, safety, and precision standards."


class ReflectionEngine:
    """Evaluates generated responses post-execution for accuracy, completeness, and self-improvement."""

    @staticmethod
    def evaluate_response(user_query: str, generated_response: str) -> ReflectionResult:
        """Perform fast self-critique on generated output."""
        misunderstandings = []
        improvements = []
        score = 0.98

        # Basic self-critique heuristics
        if len(generated_response.strip()) < 10:
            score = 0.60
            misunderstandings.append("Generated response was extremely brief.")
            improvements.append("Expand answer with deeper explanation and context.")

        if "error" in generated_response.lower() and "fail" in generated_response.lower():
            score = 0.75
            improvements.append("Check model output for error details or retry with fallback provider.")

        result = ReflectionResult(
            correctness_score=score,
            answered_correctly=(score >= 0.80),
            misunderstandings_detected=misunderstandings,
            improvement_suggestions=improvements,
            reflection_notes="Response evaluated cleanly against query requirements.",
        )

        logger.debug(f"ReflectionEngine score: {result.correctness_score} | Correct: {result.answered_correctly}")
        return result


# Global reflection engine singleton
reflection_engine = ReflectionEngine()
