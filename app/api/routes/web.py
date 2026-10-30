"""The web interface: the project site at ``/`` and the console at ``/console``.

Both pages and their assets are static and public; every data call the console makes needs the
API token. Pages are served with a strict Content-Security-Policy: scripts, styles and fonts only
from this origin, no inline script, no eval, no framing.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

router = APIRouter(tags=["Web"], include_in_schema=False)

WEB_DIR = Path(__file__).resolve().parents[2] / "static" / "web"

CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; media-src 'self'; manifest-src 'self'; base-uri 'none'; form-action 'self'; "
    "frame-ancestors 'none'"
)
_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    # The console may use the microphone (speech recognition); nothing else.
    "Permissions-Policy": "microphone=(self), camera=(), geolocation=(), payment=(), usb=()",
}
_PAGE_HEADERS = {**_SECURITY_HEADERS, "Content-Security-Policy": CSP, "X-Frame-Options": "DENY", "Cache-Control": "no-cache"}
_ASSET_TYPES = {
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
}


def _page(name: str) -> HTMLResponse:
    return HTMLResponse((WEB_DIR / name).read_text(encoding="utf-8"), headers=_PAGE_HEADERS)


@router.get("/", response_class=HTMLResponse)
async def site() -> HTMLResponse:
    return _page("index.html")


@router.get("/console", response_class=HTMLResponse)
async def console() -> HTMLResponse:
    return _page("console.html")


@router.get("/dashboard")
async def dashboard() -> RedirectResponse:
    """The old dashboard address keeps working."""
    return RedirectResponse("/console", status_code=308)


@router.get("/web/{asset:path}")
async def asset(asset: str) -> FileResponse:
    path = (WEB_DIR / asset).resolve()
    if not path.is_relative_to(WEB_DIR) or not path.is_file() or path.suffix not in _ASSET_TYPES:
        raise HTTPException(status_code=404, detail="Not found")
    headers = {**_SECURITY_HEADERS, "Cache-Control": "no-cache" if path.suffix in (".css", ".js") else "public, max-age=604800"}
    return FileResponse(path, media_type=_ASSET_TYPES[path.suffix], headers=headers)
