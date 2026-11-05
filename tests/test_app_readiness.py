"""P2 item 1: open_app is ready within seconds for an app that is already running, and never waits 45 s.

Root cause of the 45 s: a stale (> 24 h) app index was rebuilt synchronously on the event loop by the first
``find_app`` of the day, and that rebuild walked the whole of Program Files (the depth limit skipped files but
never pruned the walk). Readiness then matched the user's existing Notepad window by title.

Every window / launch helper is mocked here: no real desktop, no real Program Files scan.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from unittest.mock import patch

import pytest

import automation.application_controller as ac
import automation.window_controller as wc


class FakeDesktop:
    """Windows ({hwnd: title}) + the foreground window, changed by scheduled events on a fake clock."""

    def __init__(self, windows, foreground, events=()):
        self.windows = dict(windows)
        self.fg = foreground
        self.events = sorted(events, key=lambda e: e[0])  # (at_seconds, callable(desktop))
        self.now = 0.0

    def tick(self):
        while self.events and self.events[0][0] <= self.now:
            self.events.pop(0)[1](self)

    def snapshot(self):
        self.tick()
        return dict(self.windows)

    def foreground(self):
        self.tick()
        return self.fg, self.windows.get(self.fg, "")


@pytest.fixture
def desktop(monkeypatch):
    holder = {}

    def install(windows, foreground, events=()):
        d = FakeDesktop(windows, foreground, events)

        async def fake_sleep(seconds):
            d.now += seconds

        monkeypatch.setattr(wc, "_clock", lambda: d.now)
        monkeypatch.setattr(wc, "_sleep", fake_sleep)
        monkeypatch.setattr(wc.window_controller, "window_snapshot", d.snapshot)
        monkeypatch.setattr(wc.window_controller, "foreground_window", d.foreground)
        holder["d"] = d
        return d

    return install


NOTEPAD = {"name": "Notepad", "exe": "notepad.exe", "type": "exe", "launch_cmd": "notepad.exe"}
JARVIS_CONSOLE = (1, "JARVIS console - Google Chrome")


@pytest.fixture
def launch(monkeypatch):
    monkeypatch.setattr(ac.dynamic_app_manager, "find_app", lambda q: dict(NOTEPAD) if "notepad" in q.lower() else None)
    monkeypatch.setattr(ac.dynamic_app_manager, "refresh_index", lambda: 0)
    with patch.object(ac.subprocess, "Popen") as popen, patch.object(ac.os, "startfile", create=True):
        yield popen


# ── readiness ────────────────────────────────────────────────────────────

async def test_existing_single_window_app_brought_to_front_is_ready_fast(desktop, launch):
    """Notepad already open with an unsaved document; the launch just brings that window to the front."""
    doc = "*YOUR TURN - Kaggle - Notepad"
    d = desktop({1: JARVIS_CONSOLE[1], 42: doc}, 1, events=[(0.3, lambda d: setattr(d, "fg", 42))])
    res = await ac.application_controller.open_app("notepad")
    assert res["success"] and res["ready"] is True
    assert res["window"] == {"hwnd": 42, "title": doc, "kind": "existing"}
    assert d.now <= 3.0, d.now


async def test_existing_window_whose_title_changes_is_a_new_tab(desktop, launch):
    """Windows 11 Notepad opens a new tab in the existing window: same hwnd, new title."""
    def new_tab(d):
        d.windows[42] = "Untitled - Notepad"
        d.fg = 42

    d = desktop({1: JARVIS_CONSOLE[1], 42: "*notes - Notepad"}, 1, events=[(0.4, new_tab)])
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is True and res["window"] == {"hwnd": 42, "title": "Untitled - Notepad", "kind": "new_tab"}
    assert d.now <= 3.0


async def test_new_window_app_is_ready_when_its_window_appears(desktop, launch):
    def appear(d):
        d.windows[77] = "Untitled - Notepad"
        d.fg = 77

    d = desktop({1: JARVIS_CONSOLE[1]}, 1, events=[(1.2, appear)])
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is True and res["window"] == {"hwnd": 77, "title": "Untitled - Notepad", "kind": "new_window"}
    assert 1.2 <= d.now <= 3.0


async def test_new_window_counts_even_when_another_app_window_was_in_front(desktop, launch):
    """An old Notepad window in front must not be mistaken for the new one when a new window appears."""
    d = desktop({42: "*old - Notepad"}, 42, events=[(0.2, lambda d: d.windows.update({77: "Untitled - Notepad"}))])
    res = await ac.application_controller.open_app("notepad")
    assert res["window"]["hwnd"] == 77 and res["window"]["kind"] == "new_window"


async def test_new_foreground_window_without_the_app_name_counts(desktop, launch):
    """Some apps' titles don't contain their name (a document title); a brand-new foreground window is the app."""
    def appear(d):
        d.windows[90] = "Shopping list"
        d.fg = 90

    desktop({1: JARVIS_CONSOLE[1]}, 1, events=[(0.5, appear)])
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is True and res["window"] == {"hwnd": 90, "title": "Shopping list", "kind": "new_window"}


async def test_app_that_never_shows_fails_within_the_bound(desktop, launch):
    d = desktop({1: JARVIS_CONSOLE[1]}, 1)
    res = await ac.application_controller.open_app("notepad")
    assert res["success"] is True and res["ready"] is False and "window" not in res
    assert d.now <= 5.0, d.now
    assert "no window" in res["note"].lower()


async def test_existing_window_that_never_comes_to_front_is_not_ready(desktop, launch):
    """An old window that merely exists (behind the console) is not proof the launch worked."""
    d = desktop({1: JARVIS_CONSOLE[1], 42: "*doc - Notepad"}, 1)
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is False and d.now <= 5.0


async def test_op_message_says_when_no_window_appeared(desktop, launch):
    from automation.gateway import ActionRequest, gateway

    desktop({1: JARVIS_CONSOLE[1]}, 1)
    res = await gateway.execute(ActionRequest("app.open", {"name": "notepad"}), "p2-ready")
    assert res.status == "executed"
    assert "no window" in res.message.lower()
    data = res.steps[0]["data"]
    assert data["ready"] is False


# ── the app index never blocks a launch for long ─────────────────────────

def _write_index(path, apps, age_seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"indexed_at": time.time() - age_seconds, "apps": apps}), encoding="utf-8")


def test_stale_index_is_used_at_once_and_refreshed_in_the_background(monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    _write_index(mgr.cache_file, {"notepad": NOTEPAD}, age_seconds=3 * 24 * 3600)
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    def slow_refresh():
        started.set()
        release.wait(5)
        done.set()
        return 1

    monkeypatch.setattr(mgr, "refresh_index", slow_refresh)
    t = time.perf_counter()
    found = mgr.find_app("notepad")
    elapsed = time.perf_counter() - t
    assert found == NOTEPAD and elapsed < 0.5, elapsed
    assert started.wait(2), "a background refresh should have started"
    assert mgr.find_app("notepad") == NOTEPAD  # no second refresh while one runs
    release.set()
    assert done.wait(2)


def test_fresh_index_starts_no_refresh(monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    _write_index(mgr.cache_file, {"notepad": NOTEPAD}, age_seconds=60)
    monkeypatch.setattr(mgr, "refresh_index", lambda: pytest.fail("fresh cache must not be rescanned"))
    assert mgr.find_app("notepad") == NOTEPAD


def test_program_files_scan_is_pruned_below_the_depth_limit(tmp_path, monkeypatch):
    deep = tmp_path / "a" / "b" / "c" / "d" / "e"
    deep.mkdir(parents=True)
    (tmp_path / "a" / "b" / "c" / "tool.exe").write_text("")
    (deep / "deep.exe").write_text("")
    visited = []
    real_walk = os.walk

    def recording_walk(top, *a, **kw):
        for root, dirs, files in real_walk(top, *a, **kw):
            visited.append(root)
            yield root, dirs, files

    monkeypatch.setattr(ac.os, "walk", recording_walk)
    found = ac._scan_exes(str(tmp_path))
    assert "tool" in found and "deep" not in found
    assert not any(str(tmp_path / "a" / "b" / "c" / "d") in v for v in visited), visited


async def test_open_app_looks_up_the_app_off_the_event_loop(monkeypatch, desktop):
    """A cold (first-run) index scan must not freeze every other request while it runs."""
    desktop({1: JARVIS_CONSOLE[1]}, 1, events=[(0.1, lambda d: (d.windows.update({77: "Untitled - Notepad"}),
                                                                 setattr(d, "fg", 77)))])

    def slow_find(q):
        time.sleep(0.3)
        return dict(NOTEPAD)

    monkeypatch.setattr(ac.dynamic_app_manager, "find_app", slow_find)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        with patch.object(ac.subprocess, "Popen"):
            res = await ac.application_controller.open_app("notepad")
    finally:
        task.cancel()
    assert res["success"] and ticks >= 5, ticks
