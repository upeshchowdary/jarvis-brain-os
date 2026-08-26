"""JARVIS Advanced Computer Automation Engine.

Implements the 7-subsystem closed-loop architecture:
1. Context Memory & Reference Resolution (context_memory.py)
2. Multi-Modal Perception Engine (perception.py)
3. Action Router — 6-Level Hierarchy (action_router.py)
4. Verification Engine (verification.py)
5. Recovery Engine (recovery.py)
6. Task Planner & Graph (planner.py)
7. Advanced Orchestrator (engine_orchestrator.py)
"""

from automation.engine.context_memory import execution_memory, ExecutionMemory, Entity
from automation.engine.perception import perception_engine, PerceptionEngine, ScreenState
from automation.engine.verification import verification_engine, VerificationEngine, VerificationResult
from automation.engine.recovery import recovery_engine, RecoveryEngine
from automation.engine.action_router import action_router, ActionRouter, ActionSpec, ActionLevel
from automation.engine.planner import task_planner, TaskPlanner, TaskGraph, TaskNode
from automation.engine.engine_orchestrator import advanced_orchestrator, AdvancedOrchestrator

__all__ = [
    "execution_memory",
    "ExecutionMemory",
    "Entity",
    "perception_engine",
    "PerceptionEngine",
    "ScreenState",
    "verification_engine",
    "VerificationEngine",
    "VerificationResult",
    "recovery_engine",
    "RecoveryEngine",
    "action_router",
    "ActionRouter",
    "ActionSpec",
    "ActionLevel",
    "task_planner",
    "TaskPlanner",
    "TaskGraph",
    "TaskNode",
    "advanced_orchestrator",
    "AdvancedOrchestrator",
]
