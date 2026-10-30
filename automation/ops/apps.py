"""Application and browser operations."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import webbrowser

from automation.gateway import OpResult, Preview, Risk, gateway
from automation.web_utils import search_url, site_url


@gateway.operation("app.open", Risk.LOW, "Launch an installed application")
async def open_app(name: str, args: str = "") -> OpResult:
    from automation.application_controller import application_controller

    res = await application_controller.open_app(name, args)
    if res.get("success"):
        return OpResult.ok(f"Opened {res.get('app', name)}.", data=res)
    return OpResult.fail(res.get("error") or f"Couldn't open {name}.", data=res)


def _close_preview(args):
    return Preview(f"Close {args.get('name')} (unsaved work in it may be lost).")


@gateway.operation("app.close", Risk.MEDIUM, "Close an application window", preview=_close_preview)
async def close_app(name: str) -> OpResult:
    from automation.application_controller import application_controller

    res = await application_controller.close_app(name, force=False)
    return OpResult(bool(res.get("success")), f"Closed {res.get('app', name)}." if res.get("success") else f"Couldn't find an open {name} window.")


_BROWSERS = {
    "chrome": ["chrome", "chrome.exe", "google-chrome"],
    "edge": ["msedge", "msedge.exe", "microsoft-edge"],
    "firefox": ["firefox", "firefox.exe"],
    "brave": ["brave", "brave.exe", "brave-browser"],
}


def _launch_url(url: str, browser: str = "") -> bool:
    browser = (browser or "").lower().strip()
    if browser in _BROWSERS:
        if sys.platform == "win32":
            from automation.application_controller import dynamic_app_manager

            info = dynamic_app_manager.find_app(browser)
            if info and info.get("type") == "exe":
                subprocess.Popen([info["launch_cmd"], url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        for exe in _BROWSERS[browser]:
            path = shutil.which(exe)
            if path:
                subprocess.Popen([path, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
    return bool(webbrowser.open(url, new=2))


@gateway.operation("browser.open_url", Risk.LOW, "Open a website in the browser")
async def open_url(url: str, browser: str = "") -> OpResult:
    resolved = site_url(url) or url
    if not resolved.startswith(("http://", "https://")):
        return OpResult.fail(f"'{url}' doesn't look like a website.")
    ok = await asyncio.to_thread(_launch_url, resolved, browser)
    return OpResult(ok, f"Opened {resolved}." if ok else "Couldn't open the browser.", data={"url": resolved})


@gateway.operation("browser.search", Risk.LOW, "Search the web (Google, YouTube, …) in the browser")
async def search(query: str, engine: str = "google", browser: str = "") -> OpResult:
    url = search_url(query, engine)
    ok = await asyncio.to_thread(_launch_url, url, browser)
    label = engine.title() if engine else "Google"
    return OpResult(ok, f"Searching {label} for “{query}”." if ok else "Couldn't open the browser.", data={"url": url})
