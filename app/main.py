"""FastAPI application for JARVIS.

Security defaults: binds to 127.0.0.1, every route except /health requires the
bearer token in .env (JARVIS_API_TOKEN, generated on first run), CORS is closed
unless CORS_ORIGINS is set, and requests are rate limited per client.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware import jarvis_exception_handler, log_requests_middleware
from app.api.routes import (
    automation_router,
    chat_router,
    config_router,
    health_router,
    status_router,
    voice_router,
)
from app.api.security import rate_limit, request_id_middleware, require_auth
from app.database.connection import db_manager
from app.utils.exceptions import JarvisError
from core.config import ensure_api_token, settings
from core.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    from core import connectivity
    from core.llm import llm
    from memory.long_term import long_term_memory
    from memory.vector_memory import vector_memory

    ensure_api_token()
    await db_manager.init_db()
    await long_term_memory.init_db()
    await vector_memory.load_from_db()
    if settings.BACKGROUND_SERVICES:
        connectivity.start_background_refresh()
        from services import start_background_services

        await start_background_services()
    logger.info(f"JARVIS ready on http://{settings.HOST}:{settings.PORT} (token in .env as JARVIS_API_TOKEN)")
    yield
    if settings.BACKGROUND_SERVICES:
        from services import stop_background_services

        await stop_background_services()
    await llm.aclose()
    logger.info("JARVIS shut down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description="JARVIS — personal AI assistant & automation for Windows",
        version=settings.SYSTEM_VERSION,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )
    if settings.CORS_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.CORS_ORIGINS,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        )
    app.middleware("http")(log_requests_middleware)
    app.middleware("http")(request_id_middleware)
    app.add_exception_handler(JarvisError, jarvis_exception_handler)

    protected = [Depends(rate_limit), Depends(require_auth)]
    app.include_router(health_router)
    for r in (chat_router, config_router, status_router, automation_router, voice_router):
        app.include_router(r, dependencies=protected)

    from app.api.routes.extras import router as extras_router
    from app.api.routes.web import router as web_router

    app.include_router(extras_router, dependencies=protected)
    app.include_router(web_router)  # static site + console; every data call they make needs the token
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
