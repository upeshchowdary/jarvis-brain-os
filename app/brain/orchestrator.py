"""Main Brain Orchestrator Engine for JARVIS AI Operating System."""

import json
import re
from typing import Dict, Any, Optional
from app.domain.models.chat import ChatRequest, ChatMessage, MessageRole
from app.domain.models.response import BrainResponse, LLMResponseMetadata, TokenUsage
from app.brain.context_builder import ContextBuilder
from app.brain.intent_detector import IntentDetector
from app.brain.reasoning import ReasoningEngine
from app.brain.planner import Planner
from app.prompts.manager import prompt_manager
from app.llm.factory import LLMFactory
from app.utils.logger import logger
from app.utils.exceptions import ReasoningError, LLMError


class BrainOrchestrator:
    """Coordinates the end-to-end cognitive pipeline for processing inputs into structured responses."""

    def __init__(self) -> None:
        self.context_builder = ContextBuilder()
        self.intent_detector = IntentDetector()
        self.reasoning_engine = ReasoningEngine()
        self.planner = Planner()

    @staticmethod
    def _clean_json_markdown(text: str) -> str:
        """Strip markdown code block wrappers or extract JSON object substring from raw LLM output."""
        text = text.strip()
        if "```" in text:
            text = re.sub(r"^```(?:json)?\n?", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\n?```$", "", text)
            text = text.strip()
        # Find outer-most JSON object bounds { ... }
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0).strip()
        return text.strip()

    async def process_request(self, request: ChatRequest) -> BrainResponse:
        """Execute the cognitive data flow pipeline."""
        logger.info(f"Brain Pipeline initiated for query: '{request.query[:60]}...'")

        # 1. Context Building
        context = self.context_builder.build_context(request)

        # 2. Intent Detection Heuristics
        heuristic_intent = self.intent_detector.detect_intent_heuristics(request.query)

        # 3. Prompt Construction
        system_prompt_str = prompt_manager.render_prompt(
            "system/jarvis_system.jinja2",
            context["system_config"],
        )

        developer_prompt_str = prompt_manager.render_prompt(
            "developer/reasoning_developer.jinja2",
            {"user_query": request.query},
        )

        # Assemble full message sequence for LLM
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt_str),
            ChatMessage(role=MessageRole.DEVELOPER, content=developer_prompt_str),
        ]
        # Include conversation history if available
        messages.extend(request.conversation_history)
        # Add primary query
        messages.append(ChatMessage(role=MessageRole.USER, content=request.query))

        # 4. LLM Abstraction Call
        provider_name = request.override_provider
        model_name = request.override_model
        llm_provider = LLMFactory.create_provider(provider_name=provider_name, model_name=model_name)

        try:
            llm_result = await llm_provider.generate_completion(
                messages=messages,
                temperature=request.temperature or 0.2,
            )
        except LLMError as exc:
            logger.error(f"Brain Pipeline LLM execution error: {exc.message}")
            raise exc

        # 5. Parsing & Reasoning Extraction
        cleaned_content = self._clean_json_markdown(llm_result.content)
        
        parsed_json: Dict[str, Any] = {}
        try:
            parsed_json = json.loads(cleaned_content)
        except json.JSONDecodeError as exc:
            logger.warning(f"LLM output was not valid JSON. Falling back to direct formatting. Error: {exc}")
            parsed_json = {
                "intent": heuristic_intent.category.value,
                "confidence": heuristic_intent.confidence,
                "reasoning_steps": ["Direct answer generated without structured JSON wrap."],
                "plan": [],
                "required_tools": [],
                "response": llm_result.content,
            }

        # Extract components safely
        final_intent = parsed_json.get("intent", heuristic_intent.category.value)
        final_confidence = float(parsed_json.get("confidence", heuristic_intent.confidence))
        raw_reasoning = parsed_json.get("reasoning_steps", [])
        reasoning_steps = self.reasoning_engine.extract_reasoning_chain(raw_reasoning)
        raw_plan = parsed_json.get("plan", [])
        execution_plan = self.planner.parse_plan(raw_plan, request.query)
        required_tools = parsed_json.get("required_tools", [])
        final_response_text = parsed_json.get("response", llm_result.content)

        brain_response = BrainResponse(
            intent=final_intent,
            confidence=final_confidence,
            reasoning_steps=reasoning_steps,
            plan=[s.model_dump() for s in execution_plan.steps],
            required_tools=required_tools,
            response=final_response_text,
            metadata=llm_result.metadata,
        )

        logger.info(f"Brain Pipeline completed | Intent: '{final_intent}' | Latency: {llm_result.metadata.latency_ms}ms")
        return brain_response


# Global orchestrator singleton
brain_orchestrator = BrainOrchestrator()
