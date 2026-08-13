"""FastAPI Dependency Injection Providers."""

from brain.brain_manager import BrainManager, brain_manager
from app.database.connection import DatabaseManager, db_manager


def get_brain_orchestrator() -> BrainManager:
    """Dependency injector for Brain Manager."""
    return brain_manager


def get_database_manager() -> DatabaseManager:
    """Dependency injector for Database Manager."""
    return db_manager
