"""Brain Manager: Central lightweight controller orchestrating all cognitive sub-systems.

Refactored with Fast Pipeline Routing:
    USER QUERY → FAST INTENT ROUTER → CONDITIONAL MEMORY / VISION / TOOLS → CONTEXT → MODEL → CONDITIONAL REFLECTION → RESPONSE
"""

import asyncio
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
from memory.vector_memory import vector_memory, VectorMemoryManager
from app.database.connection import db_manager

# Ensure active tools module is imported and registered
import tools  # noqa: F401


class BrainExecutionOutput(BaseModel):
    query: str
    intent: StructuredIntent
    reasoning_steps: List[str]
    plan: HierarchicalPlan
    response: str
    reflection: Optional[ReflectionResult] = None
    tool_results: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# Session-level vision context cache: maps session_id -> last vision description
_last_vision_context: Dict[str, str] = {}


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
        """Optimized Fast Coordination Pipeline across all cognitive sub-systems."""
        start_time = time.perf_counter()
        logger.info(f"BrainManager processing user input: '{user_query[:60]}...'")
        q_lower = user_query.lower().strip()

        # Telemetry timing structure
        telemetry_timing = {
            "intent_ms": 0.0,
            "memory_ms": 0.0,
            "context_ms": 0.0,
            "vision_ms": 0.0,
            "reasoning_ms": 0.0,
            "planning_ms": 0.0,
            "model_ms": 0.0,
            "reflection_ms": 0.0,
        }

        # 1. Session & History Retrieval
        session = await self.conversation_manager.get_or_create_session_async(session_id)
        self.conversation_manager.add_user_message(session.session_id, user_query)
        history = self.conversation_manager.get_history(session.session_id)

        # 2. FAST ROUTER — Classify Intent FIRST before running expensive memory or tool lookups
        t_intent_start = time.perf_counter()
        detected_intent = self.intent_engine.detect_intent(user_query)
        telemetry_timing["intent_ms"] = round((time.perf_counter() - t_intent_start) * 1000, 2)

        if detected_intent.intent == "MODEL_SWITCH":
            target_m = detected_intent.arguments.get("target_model")
            if target_m:
                logger.info(f"MODEL_SWITCH intent triggered. Switching active model to '{target_m}'")
                self.model_manager.switch_model(target_m)
                model_override = target_m

        # Upgrade follow-up queries to SCREEN_VISION if session has recent vision context
        _followup_triggers = [
            "and", "and?", "what else", "now what", "tell me more", "continue",
            "anything else", "what more", "go on", "more", "elaborate",
        ]
        if (
            detected_intent.intent == "GENERAL_CONVERSATION"
            and session.session_id in _last_vision_context
            and any(q_lower.strip() == t or q_lower.strip().startswith(t + " ") for t in _followup_triggers)
        ):
            logger.info("BrainManager: upgrading follow-up to SCREEN_VISION (reusing last vision context).")
            detected_intent = StructuredIntent(
                intent="SCREEN_VISION",
                confidence=0.85,
                arguments={"query": user_query, "reuse_cache": True},
                summary="Follow-up question reusing last vision context.",
            )

        # 3. MEMORY RETRIEVAL — Always load user facts (<1ms SQLite) + conditional vector/history search
        t_mem_start = time.perf_counter()
        user_facts: Dict[str, str] = await self.long_term_memory.get_all_facts()
        past_conversations: List[Dict[str, Any]] = []
        vector_docs: List[Any] = []

        requires_memory = (
            detected_intent.intent in ("KNOWLEDGE_REQUEST", "TASK_PLANNING", "CODE_GENERATION")
            or any(kw in q_lower for kw in [
                "remember", "recall", "my ", "about me", "who am i", "favorite", "favourite",
                "yesterday", "last time", "previous", "preference", "what did i ask", "college"
            ])
        )

        if requires_memory:
            logger.info(f"BrainManager: Intent [{detected_intent.intent}] running vector & past conversation retrieval.")
            # Fire fact extraction in background — does not block main pipeline
            asyncio.create_task(self.long_term_memory.auto_extract_facts(user_query))
            past_conversations = await db_manager.search_past_conversations(user_query, limit=5)
            vector_docs = vector_memory.search_similar(user_query, top_k=3)

        telemetry_timing["memory_ms"] = round((time.perf_counter() - t_mem_start) * 1000, 2)

        # 4. Context Building
        t_ctx_start = time.perf_counter()
        context = self.context_manager.build_context(
            user_query=user_query,
            conversation_history=history,
            extra_context={"user_facts": user_facts},
        )
        telemetry_timing["context_ms"] = round((time.perf_counter() - t_ctx_start) * 1000, 2)

        # 5. CONDITIONAL VISION / TOOL DISPATCHING
        tool_execution_data: Optional[Dict[str, Any]] = None
        tool_context_injection = ""
        t_vision_start = time.perf_counter()

        if detected_intent.intent == "REALTIME_KNOWLEDGE_SEARCH":
            search_spec = ToolCallSpec(
                tool="internet_search",
                arguments={"query": user_query, "max_results": 5},
            )
            tool_res = await self.tool_router.route_and_execute(search_spec)
            tool_execution_data = tool_res
            snippets_list = []
            if tool_res.get("success"):
                results = tool_res.get("data", {}).get("results", [])
                for item in results:
                    snippets_list.append(f"- [{item.get('title')}]: {item.get('snippet')} (URL: {item.get('url')})")
            
            snippets_str = "\n".join(snippets_list) if snippets_list else "Live web search query completed."
            tool_context_injection = (
                f"\n\n[LIVE REAL-TIME INTERNET ACCESS: ACTIVE]\n"
                f"Current Date: {context['current_date']}\n"
                f"Below are live search results fetched from the internet right now for the user's query.\n"
                f"[LIVE INTERNET SEARCH RESULTS]:\n{snippets_str}\n"
            )

        elif detected_intent.intent == "AUTOMATION_TASK":
            # Route to automation orchestrator — runs the full OBSERVE->PLAN->ACT loop
            try:
                from automation.orchestrator import automation_orchestrator
                auto_result = await automation_orchestrator.execute(
                    command=user_query,
                    session_id=session.session_id,
                )
                tool_execution_data = {
                    "automation_result": auto_result.summary,
                    "is_agent": auto_result.details.get("agent", False) if auto_result.details else False,
                }
                status_icon = "✅" if auto_result.success else "⚠️"
                is_agent_run = auto_result.details.get("agent", False) if auto_result.details else False

                # Build rich response for agent-run tasks vs simple tasks
                if is_agent_run:
                    steps_taken = auto_result.details.get("steps_taken", auto_result.steps_done)
                    stopped_reason = auto_result.details.get("stopped_reason", "")
                    final_response_text = (
                        f"{status_icon} **Task {'Completed' if auto_result.success else 'Failed'}**\n"
                        f"{auto_result.summary}\n"
                        f"*Steps executed: {steps_taken} | Time: {auto_result.elapsed_ms:.0f}ms*"
                    )
                    if not auto_result.success and stopped_reason == "max_steps":
                        final_response_text += "\n\n⚠️ Reached max step limit — the task may be partially complete."
                else:
                    final_response_text = (
                        f"{'✅' if auto_result.success else '⚠️'} {auto_result.summary}"
                        + (f"\n*Steps completed: {auto_result.steps_done}*" if auto_result.steps_done else "")
                    )

                self.conversation_manager.add_assistant_message(session.session_id, final_response_text)

                elapsed_ms = (time.perf_counter() - start_time) * 1000
                prov = "ollama" if "ollama" in self.model_manager.current_model.lower() else (
                    "gemini" if "gemini" in self.model_manager.current_model.lower() else "groq"
                )
                return BrainExecutionOutput(
                    query=user_query,
                    intent=detected_intent,
                    reasoning_steps=[f"Executed automation in {auto_result.elapsed_ms:.0f}ms"],
                    plan=HierarchicalPlan(goal=user_query, complexity="low", steps=[]),
                    response=final_response_text,
                    reflection=None,
                    tool_results=tool_execution_data,
                    metadata={
                        "session_id": session.session_id,
                        "provider": prov,
                        "model": self.model_manager.current_model,
                        "requested_model": self.model_manager.current_model,
                        "fallback_used": False,
                        "total_latency_ms": round(elapsed_ms, 2),
                        "model_latency_ms": 0.0,
                        "timing_breakdown_ms": {
                            "intent_ms": telemetry_timing.get("intent_ms", 0),
                            "automation_ms": round(auto_result.elapsed_ms, 1),
                        },
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    },
                )
            except Exception as e:
                logger.error(f"Automation execution error: {e}")
                final_response_text = f"⚠️ Automation error: {e}"
                self.conversation_manager.add_assistant_message(session.session_id, final_response_text)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return BrainExecutionOutput(
                    query=user_query,
                    intent=detected_intent,
                    reasoning_steps=[f"Automation failed: {e}"],
                    plan=HierarchicalPlan(goal=user_query, complexity="low", steps=[]),
                    response=final_response_text,
                    reflection=None,
                    tool_results={"error": str(e)},
                    metadata={
                        "session_id": session.session_id,
                        "provider": "automation",
                        "model": self.model_manager.current_model,
                        "requested_model": self.model_manager.current_model,
                        "fallback_used": False,
                        "total_latency_ms": round(elapsed_ms, 2),
                        "model_latency_ms": 0.0,
                        "timing_breakdown_ms": {},
                        "usage": {},
                    },
                )

        elif detected_intent.intent == "FILESYSTEM_OPERATION":
            from automation.system_controller import system_controller
            try:
                # ── LLM-based intent classifier ──────────────────────────────────────
                # Ask the LLM to understand what the user actually wants and
                # extract the operation + target path from natural language.
                fs_classify_prompt = (
                    "You are a filesystem intent classifier for a personal AI assistant.\n"
                    "Analyse the user's query and respond with ONLY a JSON object — no markdown, no explanation.\n\n"
                    "JSON schema:\n"
                    "{\n"
                    '  "op": "<one of: read_file | count_files | list_files | file_size | check_presence | inspect_file>",\n'
                    '  "location": "<the folder or file the user refers to, in natural language, e.g. \'desktop\', \'downloads folder\', \'pass file in my desktop\'>",\n'
                    '  "wants_summary": <true if the user wants a summary/explanation of file contents, false for raw view>\n'
                    "}\n\n"
                    "Operation definitions:\n"
                    "- read_file: user wants to see or understand the contents/text inside a specific file\n"
                    "- count_files: user asks how many files/items are in a folder\n"
                    "- list_files: user wants to see the names of files/items in a folder\n"
                    "- file_size: user asks about the size or disk space used\n"
                    "- check_presence: user asks if a specific file or folder exists\n"
                    "- inspect_file: user wants metadata about a file (type, size, lines)\n\n"
                    f"User query: \"{user_query}\"\n\n"
                    "Respond with ONLY the JSON object."
                )
                fs_intent = {}
                try:
                    cls_res = await self.model_manager.generate(
                        messages=[{"role": "user", "content": fs_classify_prompt}],
                        temperature=0.0,
                        max_tokens=120,
                    )
                    raw = (cls_res.get("content") if isinstance(cls_res, dict) else getattr(cls_res, "content", "")) or ""
                    raw = raw.strip().strip("```json").strip("```").strip()
                    import json as _json
                    fs_intent = _json.loads(raw)
                except Exception as _ce:
                    logger.warning(f"FS intent classifier fallback ({_ce}); using keyword heuristics")

                op = fs_intent.get("op", "")
                location_hint = fs_intent.get("location", "")
                wants_summary = fs_intent.get("wants_summary", True)

                # ── Keyword heuristics as fallback ───────────────────────────────────
                q_low = user_query.lower()
                if not op:
                    if any(w in q_low for w in ("content", "contents", "read", "view", "open", "cat ", "what is in", "whats in", "inside the file")):
                        op = "read_file"
                    elif any(w in q_low for w in ("how many", "count", "number of files", "total files", "how many files", "how many items", "number of items")):
                        op = "count_files"
                    elif any(w in q_low for w in ("list", "show files", "what files", "which files", "files in", "items in")):
                        op = "list_files"
                    elif any(w in q_low for w in ("size", "how big", "how much space")):
                        op = "file_size"
                    elif any(w in q_low for w in ("present", "exist", "exists", "is there", "do i have", "check if")):
                        op = "check_presence"
                    elif any(w in q_low for w in ("inspect", "what is that file", "what is the file", "file info", "details of")):
                        op = "inspect_file"
                    else:
                        op = "list_files"

                # ── Smart location extraction fallback ───────────────────────────────
                # If LLM didn't give a clean location, extract it from known folder keywords.
                if not location_hint or len(location_hint.strip()) > 80:
                    _loc_map = {
                        "desktop": "desktop", "downloads": "downloads",
                        "documents": "documents", "pictures": "pictures",
                        "photos": "pictures", "music": "music",
                        "videos": "videos", "temp": "temp",
                        "appdata": "appdata", "onedrive": "onedrive",
                    }
                    location_hint = user_query  # default: let system_controller parse it
                    for kw, loc in _loc_map.items():
                        if kw in q_low:
                            location_hint = loc
                            break

                # ── Route to the correct operation ───────────────────────────────────
                if op == "read_file":
                    read_res = await system_controller.read_file(location_hint)
                    tool_execution_data = read_res
                    if read_res.get("success"):
                        path = read_res.get("path")
                        name = read_res.get("name")
                        lines = read_res.get("line_count", 0)
                        size_b = read_res.get("size_bytes", 0)
                        content = read_res.get("content", "")

                        if wants_summary and content and len(content.strip()) > 0 and not content.startswith("[Binary File:"):
                            try:
                                llm_prompt = (
                                    f"The user asked about a local file on their system.\n\n"
                                    f"File: {name} ({path}) — {size_b} bytes, {lines} lines\n\n"
                                    f"Extracted Content:\n```\n{content[:7500]}\n```\n\n"
                                    f"User Question: \"{user_query}\"\n\n"
                                    f"Analyse the file content and provide a direct, accurate, concise response. "
                                    f"Explain what the file is and answer the user's question precisely."
                                )
                                llm_res = await self.model_manager.generate(
                                    messages=[{"role": "user", "content": llm_prompt}],
                                    temperature=0.2,
                                    max_tokens=700,
                                )
                                ans_text = (llm_res.get("content") if isinstance(llm_res, dict) else getattr(llm_res, "content", "")) or ""
                                if ans_text and len(ans_text.strip()) > 10:
                                    final_response_text = f"📄 **{name}** (`{path}`)\n\n{ans_text.strip()}"
                                else:
                                    final_response_text = f"📄 **{name}** (`{path}` — {size_b} bytes, {lines} lines):\n```\n{content}\n```"
                            except Exception as e:
                                logger.warning(f"LLM file summarization fallback: {e}")
                                final_response_text = f"📄 **{name}** (`{path}` — {size_b} bytes, {lines} lines):\n```\n{content}\n```"
                        else:
                            final_response_text = f"📄 **{name}** (`{path}` — {size_b} bytes, {lines} lines):\n```\n{content}\n```"
                    else:
                        final_response_text = f"⚠️ Could not read file: {read_res.get('error')}"

                elif op == "count_files":
                    list_res = await system_controller.list_directory(location_hint)
                    tool_execution_data = list_res
                    if list_res.get("success"):
                        items = list_res.get("items", [])
                        path = list_res.get("path")
                        total = list_res.get("total_items", len(items))
                        file_count = sum(1 for i in items if not i.get("is_dir"))
                        dir_count = sum(1 for i in items if i.get("is_dir"))
                        final_response_text = (
                            f"📁 **{path}** contains **{total} items total** "
                            f"({file_count} files, {dir_count} folders)."
                        )
                    else:
                        final_response_text = f"⚠️ Could not count files: {list_res.get('error')}"

                elif op == "list_files":
                    list_res = await system_controller.list_directory(location_hint)
                    tool_execution_data = list_res
                    if list_res.get("success"):
                        items = list_res.get("items", [])
                        path = list_res.get("path")
                        total = list_res.get("total_items", len(items))
                        file_lines = []
                        for item in items[:40]:
                            prefix = "📁" if item.get("is_dir") else "📄"
                            file_lines.append(f"  {prefix} {item.get('name')}")
                        items_str = "\n".join(file_lines) if file_lines else "  (empty directory)"
                        final_response_text = f"Contents of `{path}` ({total} items):\n{items_str}"
                    else:
                        final_response_text = f"⚠️ Could not list directory: {list_res.get('error')}"

                elif op == "file_size":
                    size_res = await system_controller.get_path_size(location_hint)
                    tool_execution_data = size_res
                    if size_res.get("success"):
                        name = size_res.get("name")
                        is_dir = size_res.get("is_dir")
                        item_type = "folder" if is_dir else "file"
                        formatted = size_res.get("formatted_size")
                        num_files = size_res.get("num_files")
                        path = size_res.get("path")
                        if is_dir:
                            final_response_text = f"The {item_type} \"{name}\" at `{path}` has a total size of **{formatted}** ({num_files} files)."
                        else:
                            final_response_text = f"The {item_type} \"{name}\" at `{path}` has a size of **{formatted}**."
                    else:
                        final_response_text = f"⚠️ Could not retrieve size: {size_res.get('error')}"

                elif op == "check_presence":
                    pres_res = await system_controller.check_presence(location_hint)
                    tool_execution_data = pres_res
                    if pres_res.get("exists"):
                        name = pres_res.get("name")
                        path = pres_res.get("path")
                        item_type = "folder" if pres_res.get("is_dir") else "file"
                        f_type = pres_res.get("file_type", "")
                        size_fmt = pres_res.get("formatted_size", "")
                        preview = pres_res.get("content_preview", "")
                        preview_str = f"\n\n**Content Preview:**\n```\n{preview[:500]}\n```" if preview else ""
                        final_response_text = f"✅ **Yes, the {item_type} `{name}` is present.**\n- **Location:** `{path}`\n- **Type:** {f_type}\n- **Size:** {size_fmt}{preview_str}"
                    else:
                        final_response_text = f"❌ **Not Found:** {pres_res.get('error', 'Item is not present in the system.')}"

                elif op == "inspect_file":
                    insp_res = await system_controller.inspect_file(location_hint)
                    tool_execution_data = insp_res
                    if insp_res.get("exists"):
                        name = insp_res.get("name")
                        path = insp_res.get("path")
                        item_type = "Folder" if insp_res.get("is_dir") else "File"
                        f_type = insp_res.get("file_type", "")
                        size_fmt = insp_res.get("formatted_size", "")
                        lines = insp_res.get("line_count", 0)
                        preview = insp_res.get("content_preview", "")

                        if preview and len(preview.strip()) > 0:
                            try:
                                llm_prompt = (
                                    f"The user asked to inspect a local file.\n\n"
                                    f"File: {name} ({f_type}, {size_fmt}, {lines} lines)\nPath: {path}\n\n"
                                    f"Content Preview:\n```\n{preview[:5000]}\n```\n\n"
                                    f"User Prompt: \"{user_query}\"\n\n"
                                    f"Explain what this file is and provide a clear, accurate summary of its contents."
                                )
                                llm_res = await self.model_manager.generate(
                                    messages=[{"role": "user", "content": llm_prompt}],
                                    temperature=0.2,
                                    max_tokens=500,
                                )
                                ans_text = (llm_res.get("content") if isinstance(llm_res, dict) else getattr(llm_res, "content", "")) or ""
                                if ans_text and len(ans_text.strip()) > 10:
                                    final_response_text = f"📄 **{name}** ({f_type} — `{path}`, {size_fmt})\n\n{ans_text.strip()}"
                                else:
                                    final_response_text = f"📄 **{name}** ({item_type})\n- **Type:** {f_type}\n- **Path:** `{path}`\n- **Size:** {size_fmt}\n- **Lines:** {lines}\n\n```\n{preview}\n```"
                            except Exception:
                                final_response_text = f"📄 **{name}** ({item_type})\n- **Type:** {f_type}\n- **Path:** `{path}`\n- **Size:** {size_fmt}\n- **Lines:** {lines}"
                        else:
                            final_response_text = f"📄 **{name}** ({item_type})\n- **Type:** {f_type}\n- **Path:** `{path}`\n- **Size:** {size_fmt}\n- **Lines:** {lines}"
                    else:
                        final_response_text = f"⚠️ Could not inspect item: {insp_res.get('error')}"

                else:
                    final_response_text = f"I understood your filesystem request but couldn't determine the exact operation. Could you clarify what you'd like to do?"

                self.conversation_manager.add_assistant_message(session.session_id, final_response_text)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                prov = "ollama" if "ollama" in self.model_manager.current_model.lower() else (
                    "gemini" if "gemini" in self.model_manager.current_model.lower() else "groq"
                )
                return BrainExecutionOutput(
                    query=user_query,
                    intent=detected_intent,
                    reasoning_steps=[f"Filesystem operation executed in {elapsed_ms:.1f}ms"],
                    plan=HierarchicalPlan(goal=user_query, complexity="low", steps=[]),
                    response=final_response_text,
                    reflection=None,
                    tool_results=tool_execution_data,
                    metadata={
                        "session_id": session.session_id,
                        "provider": prov,
                        "model": self.model_manager.current_model,
                        "requested_model": self.model_manager.current_model,
                        "fallback_used": False,
                        "total_latency_ms": round(elapsed_ms, 2),
                        "model_latency_ms": 0.0,
                        "timing_breakdown_ms": {
                            "intent_ms": telemetry_timing.get("intent_ms", 0),
                            "fs_ms": round(elapsed_ms, 1),
                        },
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    },
                )
            except Exception as e:
                logger.error(f"Filesystem execution error: {e}")
                final_response_text = f"⚠️ Filesystem error: {e}"
                self.conversation_manager.add_assistant_message(session.session_id, final_response_text)
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                return BrainExecutionOutput(
                    query=user_query,
                    intent=detected_intent,
                    reasoning_steps=[f"Filesystem operation failed: {e}"],
                    plan=HierarchicalPlan(goal=user_query, complexity="low", steps=[]),
                    response=final_response_text,
                    reflection=None,
                    tool_results={"error": str(e)},
                    metadata={
                        "session_id": session.session_id,
                        "provider": "system",
                        "model": self.model_manager.current_model,
                        "requested_model": self.model_manager.current_model,
                        "fallback_used": False,
                        "total_latency_ms": round(elapsed_ms, 2),
                        "model_latency_ms": 0.0,
                        "timing_breakdown_ms": {},
                        "usage": {},
                    },
                )

        elif detected_intent.intent == "SYSTEM_TELEMETRY":
            telemetry_spec = ToolCallSpec(tool="system_info", arguments={})
            tool_res = await self.tool_router.route_and_execute(telemetry_spec)
            tool_execution_data = tool_res
            if tool_res.get("success"):
                sys_data = tool_res.get("data", {})
                tool_context_injection = (
                    f"\n\n[REAL-TIME SYSTEM METRICS]:\n{sys_data}\n"
                    "Use the live system metrics above to provide precise hardware and OS information to the user."
                )

        elif detected_intent.intent == "SCREEN_VISION":
            query_type = detected_intent.arguments.get("visual_query_type", "SCREEN_DESCRIPTION")
            reuse_cache = detected_intent.arguments.get("reuse_cache", False)
            force_refresh = detected_intent.arguments.get("force_refresh", False)
            cached_ctx = _last_vision_context.get(session.session_id)

            if reuse_cache and cached_ctx and not force_refresh:
                logger.info("BrainManager SCREEN_VISION: reusing cached vision context for follow-up.")
                tool_execution_data = {"cached": True, "description": cached_ctx}
                tool_context_injection = (
                    f"\n\n[LIVE SCREEN VISION — CACHED FROM PREVIOUS ANALYSIS]\n"
                    f"You are JARVIS with real-time visual perception.\n\n"
                    f"[LAST SCREEN ANALYSIS]:\n{cached_ctx}\n\n"
                    f"The user is asking a follow-up: \"{user_query}\"\n"
                    f"Answer using the screen analysis above."
                )
            else:
                try:
                    from vision.screen_analyzer import analyze_screen, grab_screenshot
                    from vision.screen_capture import screen_capture_engine

                    screenshot = grab_screenshot() or screen_capture_engine.capture_full_desktop()
                    active_win = screen_capture_engine.get_active_window_info()

                    vision_result = await analyze_screen(
                        image=screenshot,
                        user_query=user_query,
                        query_type=query_type,
                        session_id=session.session_id,
                        window_title=active_win.title if active_win else "",
                        app_name=active_win.app_name if active_win else "",
                        active_model=self.model_manager.current_model,
                    )
                    tool_execution_data = vision_result
                    vision_description = vision_result.get("description", "")
                    vision_label = vision_result.get("vision_label", "👁 VISION: Active")
                    if vision_description:
                        _last_vision_context[session.session_id] = vision_description

                    tool_context_injection = (
                        f"\n\n[LIVE SCREEN VISION ENGINE ACTIVE]\n"
                        f"{vision_label}\n"
                        f"[SCREEN ANALYSIS RESULT]:\n{vision_description}\n\n"
                        f"Answer using the screen analysis above."
                    )
                except Exception as exc:
                    logger.error(f"Vision analysis failed: {exc}")
                    tool_execution_data = {"success": False, "error": str(exc)}
                    tool_context_injection = (
                        f"\n\n[VISION ERROR]: Vision perception is currently unavailable ({exc})."
                    )

        telemetry_timing["vision_ms"] = round((time.perf_counter() - t_vision_start) * 1000, 2)

        # 6. CONDITIONAL REASONING & PLANNING — Only run heavy planning for multi-step goals or complex queries
        t_reason_start = time.perf_counter()
        requires_complex_reasoning = (
            detected_intent.intent in ("TASK_PLANNING", "CODE_GENERATION")
            or len(user_query) > 150
            or any(kw in q_lower for kw in ["how to build", "architecture", "plan", "steps to", "refactor"])
        )

        if requires_complex_reasoning:
            reasoning_steps = await self.reasoning_engine.generate_reasoning(user_query)
            hierarchical_plan = self.planner.create_plan(user_query, intent_code=detected_intent.intent)
        else:
            reasoning_steps = [f"Processed intent: '{detected_intent.intent}' cleanly."]
            hierarchical_plan = HierarchicalPlan(goal=user_query, complexity="low", steps=[])

        telemetry_timing["reasoning_ms"] = round((time.perf_counter() - t_reason_start) * 1000, 2)

        # 7. System Persona & Live Date/Time Context Construction
        system_persona = self.personality_manager.get_system_prompt(personality_override=personality)

        # Inject Live Local System Date & Time Context
        live_time_str = context.get("current_formatted_local", "")
        time_12h = context.get("current_time_full", "")
        date_str = context.get("current_date", "")
        
        system_persona += (
            f"\n\n[LIVE SYSTEM DATE & TIME CONTEXT]:\n"
            f"- Current Local Date & Time: {live_time_str}\n"
            f"- Current Local Time: {time_12h}\n"
            f"- Current Date: {date_str}\n"
            f"MANDATE: Only mention or state the date or time if the user explicitly asks for the current time, date, day, or schedule. Do NOT include the date or time in normal greetings, pleasantries, or general conversations.\n"
        )
        
        memory_block = []
        if user_facts:
            facts_str = ", ".join([f"{k}: {v}" for k, v in user_facts.items()])
            memory_block.append(f"[STORED USER FACTS & PREFERENCES]: {facts_str}")

        if vector_docs:
            v_str_list = [f"- {d.content}" for d in vector_docs]
            memory_block.append(f"[RELEVANT SEMANTIC MEMORIES]:\n" + "\n".join(v_str_list))

        if past_conversations:
            past_str_list = []
            for p in past_conversations:
                dt_str = str(p.get("created_at", ""))[:10]
                q_text = str(p.get("user_query", ""))
                r_text = str(p.get("response_text", ""))[:200]
                past_str_list.append(f"- [{dt_str}]: User: '{q_text}' -> JARVIS: '{r_text}'")
            if past_str_list:
                memory_block.append(f"[HISTORICAL PAST CHATS]:\n" + "\n".join(past_str_list))

        if memory_block:
            system_persona += "\n\n" + "\n\n".join(memory_block)

        system_persona += tool_context_injection
        system_persona += f"\n\n[ACTIVE RUNNING MODEL]: You are currently running on '{self.model_manager.current_model}'."

        messages = [{"role": "system", "content": system_persona}]
        messages.extend(history)

        # 8. LLM Completion via ModelManager
        t_model_start = time.perf_counter()
        is_realtime = (detected_intent.intent == "REALTIME_KNOWLEDGE_SEARCH")
        generation_res = await self.model_manager.generate(
            messages=messages,
            model=model_override or self.model_manager.current_model,
            is_realtime_query=is_realtime,
            intent_code=detected_intent.intent,
        )
        telemetry_timing["model_ms"] = round((time.perf_counter() - t_model_start) * 1000, 2)

        final_response_text = generation_res["content"]
        self.conversation_manager.add_assistant_message(session.session_id, final_response_text)

        # Auto-index turn into vector memory ONLY for substantial non-trivial turns
        # Fire as background task so it doesn't delay the response to the user
        if len(user_query) > 20 and detected_intent.intent != "GENERAL_CONVERSATION":
            import uuid
            doc_id = f"chat_{session.session_id}_{uuid.uuid4().hex[:6]}"
            chat_turn_summary = f"User asked: {user_query} | JARVIS answered: {final_response_text[:300]}"
            asyncio.create_task(
                vector_memory.add_document_async(doc_id=doc_id, content=chat_turn_summary, metadata={"session_id": session.session_id})
            )

        # 9. CONDITIONAL REFLECTION ENGINE EVALUATION — Only reflect for complex tasks or tool runs
        t_refl_start = time.perf_counter()
        if requires_complex_reasoning or tool_execution_data:
            reflection_res = self.reflection_engine.evaluate_response(user_query, final_response_text)
        else:
            reflection_res = ReflectionResult(
                correctness_score=0.99,
                answered_correctly=True,
                reflection_notes="Direct conversation turn processed.",
            )
        telemetry_timing["reflection_ms"] = round((time.perf_counter() - t_refl_start) * 1000, 2)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        req_clean = self.model_manager.current_model.replace("ollama/", "").lower().strip()
        res_clean = generation_res.get("model", "").replace("ollama/", "").lower().strip()
        fallback_used = bool(req_clean and res_clean and req_clean != res_clean)

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
                "requested_model": self.model_manager.current_model,
                "fallback_used": fallback_used,
                "total_latency_ms": round(elapsed_ms, 2),
                "model_latency_ms": generation_res.get("latency_ms", 0),
                "timing_breakdown_ms": telemetry_timing,
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
