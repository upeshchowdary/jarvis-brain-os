"""FastAPI dependency providers."""

from app.database.connection import DatabaseManager, db_manager


def get_database_manager() -> DatabaseManager:
    return db_manager
