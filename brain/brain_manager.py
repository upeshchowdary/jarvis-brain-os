"""Brain Manager: Central lightweight controller orchestrating all cognitive sub-systems."""

import time
from typing import Dict, Any, Optional, List, AsyncGenerator
from pydantic import BaseModel, Field

from brain.brain_config import brain_config
from brain.logger import logger
from brain.model_manager import model_manager, ModelManager
from brain.personality_manager import personality_manager, PersonalityManager
from brain.prompt_manager import prompt_manager, PromptManager
from brain.context_manager import context_manager, ContextManager
from brain.conversation_manager import conversation_manager, ConversationManager
from brain.intent_engine import intent_engine, IntentEngine, StructuredIntent
from brain.reasoning_engine import reasoning_engine, ReasoningEngine
from brain.planner import planner, Planner, HierarchicalPlan
from brain.reflection_engine import reflection_engine, ReflectionEngine, ReflectionResult
from brain.tool_router import tool_router, ToolRouter, ToolCallSpec
from brain.knowledge_manager import knowledge_manager, KnowledgeManager
from memory.long_term import long_term_memory, LongTermMemoryManager

# Ensure active tools module is imported and registered
import tools  # noqa: F401


class BrainExecutionOutput(BaseModel):
    query: str
    intent: StructuredIntent
    reasoning_steps: List[str]
    plan: HierarchicalPlan
    response: str
    reflection: ReflectionResult
    tool_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BrainManager:
    """Central Controller for the JARVIS AI Operating Assistant Brain Framework."""

    def __init__(
        self,
        model_mgr: ModelManager = model_manager,
        personality_mgr: PersonalityManager = personality_manager,
        prompt_mgr: PromptManager = prompt_manager,
        context_mgr: ContextManager = context_manager,
        conversation_mgr: ConversationManager = conversation_manager,
        intent_eng: IntentEngine = intent_engine,
        reasoning_eng: ReasoningEngine = reasoning_engine,
        plan_eng: Planner = planner,
        reflection_eng: ReflectionEngine = reflection_engine,
        router: ToolRouter = tool_router,
        knowledge_mgr: KnowledgeManager = knowledge_manager,
        long_term_mem: LongTermMemoryManager = long_term_memory,
    ) -> None:
        self.model_manager = model_mgr
        self.personality_manager = personality_mgr
        self.prompt_manager = prompt_mgr
        self.context_manager = context_mgr
        self.conversation_manager = conversation_mgr
        self.intent_engine = intent_eng
        self.reasoning_engine = reasoning_eng
        self.planner = plan_eng
        self.reflection_engine = reflection_eng
        self.tool_router = router
        self.knowledge_manager = knowledge_mgr
        self.long_term_memory = long_term_mem

    async def execute_cognitive_pipeline(
        self,
        user_query: str,
        session_id: Optional[str] = None,
        personality: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> BrainExecutionOutput:
        """Lightweight coordination pipeline across all cognitive managers and tool execution."""
        start_time = time.perf_counter()
        logger.info(f"BrainManager processing user input: '{user_query[:60]}...'")

        # 1. Session & History Retrieval
        session = self.conversation_manager.get_or_create_session(session_id)
        self.conversation_manager.add_user_message(session.session_id, user_query)
        history = self.conversation_manager.get_history(session.session_id)

        # 2. Context Building & Long-term Memory Facts Lookup
        user_facts = await self.long_term_memory.get_all_facts()
        context = self.context_manager.build_context(
            user_query=user_query,
            conversation_history=history,
            extra_context={"user_facts": user_facts},
        )

        # 3. Intent Detection
        detected_intent = self.intent_engine.detect_intent(user_query)

        # 4. Tool Dispatching based on Intent (Phase 2 Active Tool Routing)
        tool_execution_data: Optional[Dict[str, Any]] = None
        tool_context_injection = ""

        if detected_intent.intent == "REALTIME_KNOWLEDGE_SEARCH":
            search_spec = ToolCallSpec(
                tool="internet_search",
                arguments={"query": user_query, "max_results": 4},
            )
            tool_res = await self.tool_router.route_and_execute(search_spec)
            tool_execution_data = tool_res
            if tool_res.get("success"):
                snippets = tool_res.get("data", {}).get("results", [])
                formatted_snippets = "\n".join([f"- [{item.get('title')}]: {item.get('snippet')} ({item.get('url')})" for item in snippets])
                tool_context_injection = f"\n\n[LIVE INTERNET SEARCH RESULTS]:\n{formatted_snippets}\nUse the live search results above to answer accurately."

        elif detected_intent.intent == "SYSTEM_TELEMETRY":
            telemetry_spec = ToolCallSpec(tool="system_info", arguments={})
            tool_res = await self.tool_router.route_and_execute(telemetry_spec)
            tool_execution_data = tool_res
            if tool_res.get("success"):
                sys_data = tool_res.get("data", {})
                tool_context_injection = f"\n\n[REAL-TIME SYSTEM METRICS]:\n{sys_data}\nUse the system metrics above to answer accurately."

        # 5. Reasoning & Multi-step Planning
        reasoning_steps = await self.reasoning_engine.generate_reasoning(user_query)
        hierarchical_plan = self.planner.create_plan(user_query, intent_code=detected_intent.intent)

        # 6. System Persona & Prompt Construction
        system_persona = self.personality_manager.get_system_prompt(personality_override=personality)
        if user_facts:
            facts_str = ", ".join([f"{k}: {v}" for k, v in user_facts.items()])
            system_persona += f"\nKnown User Facts: [{facts_str}]"
        
        system_persona += tool_context_injection

        messages = [{"role": "system", "content": system_persona}]
        messages.extend(history)

        # 7. LLM Completion via ModelManager
        generation_res = await self.model_manager.generate(
            messages=messages,
            model=model_override or self.model_manager.current_model,
        )

        final_response_text = generation_res["content"]
        self.conversation_manager.add_assistant_message(session.session_id, final_response_text)

        # 8. Post-Response Reflection Engine Self-Evaluation
        reflection_res = self.reflection_engine.evaluate_response(user_query, final_response_text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        output = BrainExecutionOutput(
            query=user_query,
            intent=detected_intent,
            reasoning_steps=reasoning_steps,
            plan=hierarchical_plan,
            response=final_response_text,
            reflection=reflection_res,
            tool_results=tool_execution_data,
            metadata={
                "session_id": session.session_id,
                "provider": generation_res.get("provider", "groq"),
                "model": generation_res.get("model", self.model_manager.current_model),
                "total_latency_ms": round(elapsed_ms, 2),
                "model_latency_ms": generation_res.get("latency_ms", 0),
                "usage": generation_res.get("usage", {}),
            },
        )

        logger.info(
            f"BrainManager pipeline finished in {elapsed_ms:.1f}ms | "
            f"Intent: {detected_intent.intent} | Tools Used: {bool(tool_execution_data)}"
        )
        return output


# Global brain manager singleton
brain_manager = BrainManager()
