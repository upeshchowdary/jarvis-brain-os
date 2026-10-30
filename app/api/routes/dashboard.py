"""Local web dashboard (static page; all data calls use the API token)."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Dashboard"])


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> str:
    return "<!doctype html><title>JARVIS</title><p>Dashboard coming soon.</p>"
