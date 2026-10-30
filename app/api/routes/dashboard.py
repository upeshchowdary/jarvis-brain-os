"""Local web dashboard. The page itself is static; every data call it makes needs the API token."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["Dashboard"])
_PAGE = Path(__file__).resolve().parents[2] / "static" / "dashboard.html"


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"), headers={
        "Content-Security-Policy": "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
    })
