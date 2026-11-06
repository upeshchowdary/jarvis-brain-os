"""P2 item 1 / fix round 1 item 5: the app index never makes a request wait on a disk scan.

Root cause of the 45 s: a stale (> 24 h) index was rebuilt synchronously on the event loop by the first ``find_app``
of the day, walking the whole of Program Files. Now: the walk is pruned, a stale index answers at once and is
refreshed in the background, the index is prewarmed at startup, synchronous callers on the loop never scan, every
refresh is single-flight and app_index.json is written atomically.

No real registry / Program Files / Start menu scan: ``_scan_system`` is replaced in every test.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest

import automation.application_controller as ac
from tests.test_app_readiness import CONSOLE, NOTEPAD, desktop, np  # noqa: F401  (desktop is a fixture)


# ── the app index never blocks a launch for long ─────────────────────────

@pytest.fixture
def index_file(tmp_path, monkeypatch):
    path = tmp_path / "app_index.json"
    monkeypatch.setattr(ac.DynamicSystemAppManager, "cache_file", property(lambda self: path))
    return path


def _write_index(path, apps, age_seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"indexed_at": time.time() - age_seconds, "apps": apps}), encoding="utf-8")


class GatedScan:
    """Stands in for the real system scan: counts calls, optionally waits for ``release``."""

    def __init__(self, apps=None, gated=False):
        self.apps = apps if apps is not None else {"notepad": dict(NOTEPAD)}
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        if not gated:
            self.release.set()

    def __call__(self):
        self.calls += 1
        self.started.set()
        self.release.wait(5)
        return dict(self.apps)


def _wait_until(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_stale_index_is_used_at_once_and_refreshed_in_the_background(index_file, monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    _write_index(index_file, {"notepad": NOTEPAD}, age_seconds=3 * 24 * 3600)
    scan = GatedScan(gated=True)
    monkeypatch.setattr(mgr, "_scan_system", scan)
    t = time.perf_counter()
    found = mgr.find_app("notepad")
    elapsed = time.perf_counter() - t
    assert found == NOTEPAD and elapsed < 0.5, elapsed
    assert scan.started.wait(2), "a background refresh should have started"
    assert mgr.find_app("notepad") == NOTEPAD  # no second refresh while one runs
    scan.release.set()
    assert _wait_until(lambda: not mgr._bg_running)
    assert scan.calls == 1


def test_fresh_index_starts_no_refresh(index_file, monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    _write_index(index_file, {"notepad": NOTEPAD}, age_seconds=60)
    monkeypatch.setattr(mgr, "_scan_system", lambda: pytest.fail("fresh cache must not be rescanned"))
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


def test_prewarm_builds_the_index_in_the_background(index_file, monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    scan = GatedScan(gated=True)
    monkeypatch.setattr(mgr, "_scan_system", scan)
    t = time.perf_counter()
    mgr.prewarm()
    assert time.perf_counter() - t < 0.2
    assert scan.started.wait(2)
    scan.release.set()
    assert _wait_until(lambda: mgr._loaded)
    assert mgr.find_app("notepad") == NOTEPAD and scan.calls == 1


def test_prewarm_loads_a_saved_index_without_scanning(index_file, monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    _write_index(index_file, {"notepad": NOTEPAD}, age_seconds=60)
    monkeypatch.setattr(mgr, "_scan_system", lambda: pytest.fail("a fresh saved index must not be rescanned"))
    mgr.prewarm()
    assert _wait_until(lambda: mgr._loaded) and mgr.find_app("notepad", scan=False) == NOTEPAD


def test_lookup_without_scan_never_scans_on_the_caller(index_file, monkeypatch):
    """Synchronous callers on the event loop (the router) use whatever is loaded; a scan runs in the background."""
    mgr = ac.DynamicSystemAppManager()
    scan = GatedScan(gated=True)
    monkeypatch.setattr(mgr, "_scan_system", scan)
    t = time.perf_counter()
    assert mgr.find_app("notepad", scan=False) is None
    assert time.perf_counter() - t < 0.2
    assert scan.started.wait(2)
    scan.release.set()
    assert _wait_until(lambda: mgr._loaded)
    assert mgr.find_app("notepad", scan=False) == NOTEPAD and scan.calls == 1


def test_concurrent_lookups_share_one_scan(index_file, monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    scan = GatedScan(gated=True)
    monkeypatch.setattr(mgr, "_scan_system", scan)
    results = []
    threads = [threading.Thread(target=lambda: results.append(mgr.find_app("notepad"))) for _ in range(3)]
    for th in threads:
        th.start()
    assert scan.started.wait(2)
    time.sleep(0.05)  # the other lookups are now waiting on the running scan
    scan.release.set()
    for th in threads:
        th.join(3)
    assert results == [NOTEPAD] * 3 and scan.calls == 1


def test_refresh_during_a_background_refresh_waits_for_it_instead_of_rescanning(index_file, monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    _write_index(index_file, {"notepad": NOTEPAD}, age_seconds=3 * 24 * 3600)
    scan = GatedScan(gated=True)
    monkeypatch.setattr(mgr, "_scan_system", scan)
    mgr.find_app("notepad")  # stale: background refresh starts
    assert scan.started.wait(2)
    counts = []
    th = threading.Thread(target=lambda: counts.append(mgr.refresh_index()))
    th.start()
    time.sleep(0.05)
    scan.release.set()
    th.join(3)
    assert counts == [1] and scan.calls == 1


def test_index_file_is_written_atomically(index_file, monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    _write_index(index_file, {"old": NOTEPAD}, age_seconds=60)
    monkeypatch.setattr(mgr, "_scan_system", GatedScan())

    def broken_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(ac.os, "replace", broken_replace)
    mgr.refresh_index()
    assert set(json.loads(index_file.read_text(encoding="utf-8"))["apps"]) == {"old"}  # old file intact


def test_index_write_leaves_no_temp_file(index_file, monkeypatch):
    mgr = ac.DynamicSystemAppManager()
    monkeypatch.setattr(mgr, "_scan_system", GatedScan())
    mgr.refresh_index()
    assert set(json.loads(index_file.read_text(encoding="utf-8"))["apps"]) == {"notepad"}
    assert [p.name for p in index_file.parent.iterdir()] == ["app_index.json"]


def test_router_known_app_check_never_scans_on_the_loop(index_file, monkeypatch):
    from brain import router

    mgr = ac.dynamic_app_manager
    scan = GatedScan(apps={"obscureapp": dict(NOTEPAD)}, gated=True)
    monkeypatch.setattr(mgr, "_loaded", False)
    monkeypatch.setattr(mgr, "_apps_cache", {})
    monkeypatch.setattr(mgr, "_scan_system", scan)
    t = time.perf_counter()
    assert router._is_known_app("obscureapp") is False  # not loaded yet: no answer, no blocking scan
    assert time.perf_counter() - t < 0.2
    scan.release.set()
    assert _wait_until(lambda: mgr._loaded and not mgr._bg_running)
    assert router._is_known_app("obscureapp") is True


def test_lifespan_prewarms_the_app_index(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    prewarm = []
    monkeypatch.setattr(ac.dynamic_app_manager, "prewarm", lambda: prewarm.append(True))
    with TestClient(app):
        pass
    assert prewarm == [True]


async def _ticks_while(coro):
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.02)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        res = await coro
    finally:
        task.cancel()
    return res, ticks


def _slow_find(q, *a, **kw):
    time.sleep(0.3)
    return dict(NOTEPAD)


async def test_open_app_looks_up_the_app_off_the_event_loop(monkeypatch, desktop):
    """A cold (first-run) index scan must not freeze every other request while it runs."""
    desktop({1: CONSOLE}, 1, events=[(0.1, lambda d: (d.windows.update({77: np("Untitled - Notepad")}),
                                                       setattr(d, "fg", 77)))])
    monkeypatch.setattr(ac.dynamic_app_manager, "find_app", _slow_find)
    with patch.object(ac.subprocess, "Popen"):
        res, ticks = await _ticks_while(ac.application_controller.open_app("notepad"))
    assert res["success"] and ticks >= 5, ticks


async def test_close_app_looks_up_the_app_off_the_event_loop(monkeypatch):
    monkeypatch.setattr(ac.dynamic_app_manager, "find_app", _slow_find)
    monkeypatch.setattr(ac.window_controller, "close_window", AsyncMock(return_value={"success": True}))
    res, ticks = await _ticks_while(ac.application_controller.close_app("notepad"))
    assert res["success"] and ticks >= 5, ticks
