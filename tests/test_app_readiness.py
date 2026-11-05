"""P2 item 1: open_app is ready within seconds for an app that is already running, and never waits 45 s.

Root cause of the 45 s: a stale (> 24 h) app index was rebuilt synchronously on the event loop by the first
``find_app`` of the day, and that rebuild walked the whole of Program Files (the depth limit skipped files but
never pruned the walk). Readiness then matched the user's existing Notepad window by title.

Fix round 1: windows are identified by their owning process (title only as a hint for apps whose process isn't
known), so a popup / Notepad++ / a browser tab is never taken for the app. Index tests: tests/test_app_index.py.

Every window / launch helper is mocked here: no real desktop, no real Program Files scan.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

import automation.application_controller as ac
import automation.window_controller as wc


class FakeDesktop:
    """Windows ({hwnd: (title, process)}) + the foreground window, changed by scheduled events on a fake clock."""

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
        return self.fg, self.windows.get(self.fg, ("", ""))[0]


@pytest.fixture
def desktop(monkeypatch):
    def install(windows, foreground, events=()):
        d = FakeDesktop(windows, foreground, events)

        async def fake_sleep(seconds):
            d.now += seconds

        monkeypatch.setattr(wc, "_clock", lambda: d.now)
        monkeypatch.setattr(wc, "_sleep", fake_sleep)
        monkeypatch.setattr(wc.window_controller, "window_snapshot", d.snapshot)
        monkeypatch.setattr(wc.window_controller, "foreground_window", d.foreground)
        return d

    return install


NOTEPAD = {"name": "Notepad", "exe": "notepad.exe", "type": "exe", "launch_cmd": "notepad.exe"}
SPOTIFY_UWP = {"name": "Spotify", "exe": "SpotifyAB.SpotifyMusic_x!Spotify", "type": "uwp",
               "launch_cmd": "shell:AppsFolder\\SpotifyAB.SpotifyMusic_x!Spotify"}
CONSOLE = ("JARVIS console - Google Chrome", "chrome.exe")


def np(title):
    return title, "notepad.exe"


@pytest.fixture
def launch(monkeypatch):
    apps = {"notepad": NOTEPAD, "spotify": SPOTIFY_UWP}
    monkeypatch.setattr(ac.dynamic_app_manager, "find_app",
                        lambda q, *a, **kw: dict(apps[q.lower()]) if q.lower() in apps else None)
    monkeypatch.setattr(ac.dynamic_app_manager, "refresh_index", lambda: 0)
    with patch.object(ac.subprocess, "Popen") as popen, patch.object(ac.os, "startfile", create=True):
        yield popen


# ── readiness ────────────────────────────────────────────────────────────

async def test_existing_single_window_app_brought_to_front_is_ready_fast(desktop, launch):
    """Notepad already open with an unsaved document; the launch just brings that window to the front."""
    doc = "*YOUR TURN - Kaggle - Notepad"
    d = desktop({1: CONSOLE, 42: np(doc)}, 1, events=[(0.3, lambda d: setattr(d, "fg", 42))])
    res = await ac.application_controller.open_app("notepad")
    assert res["success"] and res["ready"] is True
    assert res["window"] == {"hwnd": 42, "title": doc, "kind": "existing"}
    assert d.now <= 3.0, d.now


async def test_existing_window_whose_title_changes_is_a_new_tab(desktop, launch):
    """Windows 11 Notepad opens a new tab in the existing window: same hwnd, new title."""
    def new_tab(d):
        d.windows[42] = np("Untitled - Notepad")
        d.fg = 42

    d = desktop({1: CONSOLE, 42: np("*notes - Notepad")}, 1, events=[(0.4, new_tab)])
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is True and res["window"] == {"hwnd": 42, "title": "Untitled - Notepad", "kind": "new_tab"}
    assert d.now <= 3.0


async def test_new_window_app_is_ready_when_its_window_appears(desktop, launch):
    def appear(d):
        d.windows[77] = np("Untitled - Notepad")
        d.fg = 77

    d = desktop({1: CONSOLE}, 1, events=[(1.2, appear)])
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is True and res["window"] == {"hwnd": 77, "title": "Untitled - Notepad", "kind": "new_window"}
    assert 1.2 <= d.now <= 3.0


async def test_new_window_counts_even_when_another_app_window_was_in_front(desktop, launch):
    """An old Notepad window in front must not be mistaken for the new one when a new window appears."""
    d = desktop({42: np("*old - Notepad")}, 42,
                events=[(0.2, lambda d: d.windows.update({77: np("Untitled - Notepad")}))])
    res = await ac.application_controller.open_app("notepad")
    assert res["window"]["hwnd"] == 77 and res["window"]["kind"] == "new_window"


async def test_same_process_window_without_the_app_name_counts(desktop, launch):
    """Identity is the owning process: a new notepad.exe window titled after its document is the app."""
    def appear(d):
        d.windows[90] = np("Shopping list")
        d.fg = 90

    desktop({1: CONSOLE}, 1, events=[(0.5, appear)])
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is True and res["window"] == {"hwnd": 90, "title": "Shopping list", "kind": "new_window"}


@pytest.mark.parametrize("popup", [
    ("New notification", "shellexperiencehost.exe"),
    ("User Account Control", "consent.exe"),
    ("Notepad - update available", "updater.exe"),  # even with the app's name in its title
])
async def test_foreign_process_popup_is_not_accepted(desktop, launch, popup):
    def appear(d):
        d.windows[66] = popup
        d.fg = 66

    d = desktop({1: CONSOLE}, 1, events=[(0.2, appear)])
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is False and "window" not in res and d.now <= 5.0


@pytest.mark.parametrize("lookalike", [
    ("new 1 - Notepad++", "notepad++.exe"),
    ("notepad tips - Google Chrome", "chrome.exe"),
])
async def test_title_lookalikes_from_other_apps_are_not_notepad(desktop, launch, lookalike):
    """Process identity first: Notepad++ or a browser tab mentioning "notepad" is not Notepad."""
    def appear(d):
        d.windows[55] = lookalike
        d.fg = 55

    desktop({1: CONSOLE}, 1, events=[(0.2, appear)])
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is False and "window" not in res


async def test_lookalike_in_front_is_not_an_existing_notepad(desktop, launch):
    desktop({55: ("notepad tips - Google Chrome", "chrome.exe")}, 55)
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is False


async def test_unknown_process_app_uses_its_title_as_a_hint_but_never_a_browser(desktop, launch):
    """A Store app's process isn't known from its launch command: its title is the hint; browsers never count."""
    def appear(d):
        d.windows[30] = ("Spotify tips - Google Chrome", "chrome.exe")
        d.windows[31] = ("Spotify Premium", "spotify.exe")
        d.fg = 31

    desktop({1: CONSOLE}, 1, events=[(0.3, appear)])
    res = await ac.application_controller.open_app("spotify")
    assert res["ready"] is True and res["window"]["hwnd"] == 31


async def test_unknown_process_app_ignores_a_browser_tab_with_its_name(desktop, launch):
    def appear(d):
        d.windows[30] = ("Spotify tips - Google Chrome", "chrome.exe")
        d.fg = 30

    desktop({1: CONSOLE}, 1, events=[(0.3, appear)])
    res = await ac.application_controller.open_app("spotify")
    assert res["ready"] is False


async def test_app_that_never_shows_fails_within_the_bound(desktop, launch):
    d = desktop({1: CONSOLE}, 1)
    res = await ac.application_controller.open_app("notepad")
    assert res["success"] is True and res["ready"] is False and "window" not in res
    assert d.now <= 5.0, d.now
    assert "no window" in res["note"].lower()


async def test_existing_window_that_never_comes_to_front_is_not_ready(desktop, launch):
    """An old window that merely exists (behind the console) is not proof the launch worked."""
    d = desktop({1: CONSOLE, 42: np("*doc - Notepad")}, 1)
    res = await ac.application_controller.open_app("notepad")
    assert res["ready"] is False and d.now <= 5.0


async def test_op_message_says_when_no_window_appeared(desktop, launch):
    from automation.gateway import ActionRequest, gateway

    desktop({1: CONSOLE}, 1)
    res = await gateway.execute(ActionRequest("app.open", {"name": "notepad"}), "p2-ready")
    assert res.status == "executed"
    assert "no window" in res.message.lower()
    data = res.steps[0]["data"]
    assert data["ready"] is False


@pytest.mark.parametrize("launch_cmd,app_type,expected", [
    ("notepad.exe", "exe", {"notepad.exe"}),
    (r"C:\Program Files\Notepad++\notepad++.exe", "exe", {"notepad++.exe"}),
    ("wt.exe", "exe", {"wt.exe", "windowsterminal.exe"}),
    ("shell:AppsFolder\\X!App", "uwp", set()),
    ("ms-settings:", "protocol", set()),
])
def test_app_processes_come_from_the_launch_command(launch_cmd, app_type, expected):
    assert ac.app_processes({"launch_cmd": launch_cmd, "type": app_type}) == expected


def test_window_snapshot_names_each_windows_process(monkeypatch):
    class W:
        def __init__(self, hwnd, title):
            self._hWnd, self.title = hwnd, title

    monkeypatch.setattr(wc, "HAS_GW", True)
    monkeypatch.setattr(wc.gw, "getAllWindows", lambda: [W(42, "Untitled - Notepad"), W(43, "  ")], raising=False)
    monkeypatch.setattr(wc, "_process_name", lambda hwnd: {42: "notepad.exe"}.get(hwnd, ""))
    assert wc.WindowController().window_snapshot() == {42: ("Untitled - Notepad", "notepad.exe")}
