"""FastAPI Exception & Request Middleware."""

import time
from typing import Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from app.utils.exceptions import JarvisError
from app.utils.logger import logger


async def jarvis_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler converting custom JarvisExceptions into standardized HTTP error responses."""
    if isinstance(exc, JarvisError):
        logger.error(f"Handled JarvisError during {request.method} {request.url.path}: {exc.message}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": True,
                "error_type": exc.__class__.__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    logger.critical(f"Unhandled exception during {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "error_type": "InternalServerError",
            "message": "An unexpected internal server error occurred in JARVIS Brain Engine.",
            "details": str(exc) if request.app.debug else None,
        },
    )


async def log_requests_middleware(request: Request, call_next: Any) -> Any:
    """Middleware for measuring API request latency and logging inbound requests."""
    start_time = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        f"API Request | {request.method} {request.url.path} | "
        f"Status: {response.status_code} | Latency: {elapsed_ms:.1f}ms"
    )
    return response
