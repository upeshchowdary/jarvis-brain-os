"""Web interface: pages and assets are served safely, and every API call the UI makes exists."""

import re

import pytest
from fastapi.routing import APIRoute

from app.api.routes.web import WEB_DIR
from app.main import app

PAGES = ["/", "/console"]
SCRIPTS = sorted((WEB_DIR / "js").glob("*.js"))


def _routes():
    def walk(routes):
        for r in routes:
            if isinstance(r, APIRoute):
                yield r
            elif hasattr(r, "original_router"):
                yield from walk(r.original_router.routes)
            elif hasattr(r, "routes"):
                yield from walk(r.routes)

    return list(walk(app.routes))


@pytest.mark.parametrize("path", PAGES)
def test_pages_are_public_with_strict_csp(anon_client, path):
    res = anon_client.get(path)
    assert res.status_code == 200
    csp = res.headers["Content-Security-Policy"]
    assert "script-src 'self'" in csp and "unsafe-inline" not in csp and "unsafe-eval" not in csp
    assert "frame-ancestors 'none'" in csp and res.headers["X-Frame-Options"] == "DENY"
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert "<h1" in res.text and "JARVIS" in res.text


def test_old_dashboard_redirects_to_console(anon_client):
    res = anon_client.get("/dashboard", follow_redirects=False)
    assert res.status_code == 308 and res.headers["location"] == "/console"


@pytest.mark.parametrize("path", PAGES)
def test_pages_have_no_inline_code(anon_client, path):
    html = anon_client.get(path).text
    for tag in re.findall(r"<script\b[^>]*>(.*?)</script>", html, flags=re.S):
        assert tag.strip() == "", "inline script would be blocked by the CSP"
    assert "<style" not in html and not re.search(r"\sstyle=", html)
    assert not re.search(r"\son[a-z]+=", html), "inline event handlers would be blocked by the CSP"


@pytest.mark.parametrize("path", PAGES)
def test_every_referenced_asset_is_served(anon_client, path):
    html = anon_client.get(path).text
    assets = set(re.findall(r'(?:src|href)="(/web/[^"]+)"', html))
    assert assets, "pages should load their CSS/JS from /web/"
    for asset in assets:
        res = anon_client.get(asset)
        assert res.status_code == 200, asset
        assert res.headers["X-Content-Type-Options"] == "nosniff"


def test_asset_types_and_font_licence(anon_client):
    assert anon_client.get("/web/css/base.css").headers["content-type"].startswith("text/css")
    assert anon_client.get("/web/js/console.js").headers["content-type"].startswith("text/javascript")
    assert anon_client.get("/web/fonts/hubot-sans-latin-wdth-normal.woff2").headers["content-type"] == "font/woff2"
    assert "SIL Open Font License" in anon_client.get("/web/fonts/OFL.txt").text


@pytest.mark.parametrize("bad", ["/web/../main.py", "/web/..%2F..%2Fmain.py", "/web/%2e%2e/%2e%2e/core/config.py",
                                 "/web/index.html", "/web/nope.css", "/web/js"])
def test_assets_cannot_escape_or_serve_pages(anon_client, bad):
    assert anon_client.get(bad).status_code == 404


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_scripts_never_inject_html(script):
    src = script.read_text(encoding="utf-8")
    for banned in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert banned not in src, f"{script.name} uses {banned}"


def _called_paths():
    calls = set()
    for script in SCRIPTS:
        src = script.read_text(encoding="utf-8")
        for m in re.finditer(r'(?:api|post|readSSE|fetch)\(\s*([`"])(/[^`"]*)\1', src):
            path = re.sub(r"\$\{[^}]+\}", "X", m.group(2))
            path = path.split("${")[0].split("?")[0]  # an unclosed ${ is a query-string template
            calls.add(path)
    return sorted(calls)


def test_ui_calls_found():
    assert len(_called_paths()) >= 20


@pytest.mark.parametrize("path", _called_paths())
def test_every_ui_call_has_a_route(path):
    patterns = [re.compile("^" + re.sub(r"\{[^}]+\}", "[^/]+", r.path) + "$") for r in _routes()]
    assert any(p.match(path) for p in patterns), f"the UI calls {path}, which the API doesn't have"


def test_console_round_trip(sync_client):
    """The calls the console makes on start-up all succeed with the token."""
    for path in ("/automation/status", "/automation/pending", "/config", "/metrics", "/status", "/reminders",
                 "/todos", "/notes", "/routines", "/automation/workflows", "/automation/audit?limit=300"):
        assert sync_client.get(path).status_code == 200, path
