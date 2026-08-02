"""Unit tests for Brain components (IntentDetector, ContextBuilder, ReasoningEngine, Planner)."""

from app.brain.intent_detector import IntentDetector
from app.brain.context_builder import ContextBuilder
from app.brain.reasoning import ReasoningEngine
from app.brain.planner import Planner
from app.domain.models.chat import ChatRequest
from app.domain.models.intent import IntentCategory


def test_intent_detector_heuristics():
    code_result = IntentDetector.detect_intent_heuristics("def hello_world(): return 42")
    assert code_result.category == IntentCategory.CODE_ASSISTANT
    assert code_result.confidence > 0.8

    plan_result = IntentDetector.detect_intent_heuristics("What is the roadmap for building a web server?")
    assert plan_result.category == IntentCategory.TASK_PLANNING

    chat_result = IntentDetector.detect_intent_heuristics("Hello JARVIS, how are you?")
    assert chat_result.category == IntentCategory.GENERAL_CONVERSATION


def test_context_builder():
    builder = ContextBuilder()
    req = ChatRequest(query="Test Query", session_id="test_session_123")
    ctx = builder.build_context(req)
    assert ctx["user_query"] == "Test Query"
    assert ctx["session_id"] == "test_session_123"
    assert "system_config" in ctx


def test_reasoning_engine_extraction():
    engine = ReasoningEngine()
    steps = engine.extract_reasoning_chain(["Step 1: Parse query", "Step 2: Generate response"])
    assert len(steps) == 2
    assert steps[0] == "Step 1: Parse query"


def test_planner_plan_generation():
    planner = Planner()
    plan = planner.parse_plan([{"step_number": 1, "action": "search", "description": "Search internet"}], "Search query")
    assert len(plan.steps) == 1
    assert plan.steps[0].action == "search"
