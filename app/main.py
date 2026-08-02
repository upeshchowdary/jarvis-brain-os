"""FastAPI Application Entry Point for JARVIS AI Operating System."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config.settings import settings
from app.api.middleware import jarvis_exception_handler, log_requests_middleware
from app.api.routes import chat_router, health_router, config_router, status_router
from app.database.connection import db_manager
from app.utils.exceptions import JarvisError
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle context manager."""
    logger.info("Initializing JARVIS Brain Engine resources...")
    # Initialize SQLite tables asynchronously
    await db_manager.init_db()
    logger.info("JARVIS Brain Engine startup complete.")
    yield
    logger.info("Shutting down JARVIS Brain Engine resources...")


def create_app() -> FastAPI:
    """Construct and configure FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        description="Modular AI Operating System - Phase 1: Brain Engine",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Middleware
    app.middleware("http")(log_requests_middleware)

    # Global Exception Handlers
    app.add_exception_handler(JarvisError, jarvis_exception_handler)

    # Include API Routers
    app.include_router(chat_router)
    app.include_router(health_router)
    app.include_router(config_router)
    app.include_router(status_router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
