"""FastAPI Dependency Injection Providers."""

from app.brain.orchestrator import BrainOrchestrator, brain_orchestrator
from app.database.connection import DatabaseManager, db_manager


def get_brain_orchestrator() -> BrainOrchestrator:
    """Dependency injector for Brain Orchestrator."""
    return brain_orchestrator


def get_database_manager() -> DatabaseManager:
    """Dependency injector for Database Manager."""
    return db_manager
