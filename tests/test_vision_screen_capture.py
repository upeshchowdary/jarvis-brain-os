"""Unit tests for vision.screen_capture module."""

from vision.screen_capture import screen_capture_engine, ScreenCaptureEngine
from vision.environment import BoundingBox, CursorState, MonitorInfo, WindowInfo


def test_cursor_position_retrieval():
    cursor = screen_capture_engine.get_cursor_position()
    assert isinstance(cursor, CursorState)
    assert isinstance(cursor.x, int)
    assert isinstance(cursor.y, int)


def test_monitors_info_retrieval():
    monitors = screen_capture_engine.get_monitors_info()
    assert len(monitors) > 0
    assert isinstance(monitors[0], MonitorInfo)
    assert monitors[0].resolution[0] > 0
    assert monitors[0].resolution[1] > 0


def test_active_window_info():
    win_info = screen_capture_engine.get_active_window_info()
    assert isinstance(win_info, WindowInfo)
    assert win_info.bounds.width > 0
    assert win_info.bounds.height > 0


def test_all_open_windows():
    windows = screen_capture_engine.get_all_open_windows()
    assert len(windows) > 0
    assert isinstance(windows[0], WindowInfo)


def test_capture_full_desktop():
    img = screen_capture_engine.capture_full_desktop()
    assert img is not None
    assert img.size[0] > 0
    assert img.size[1] > 0


def test_capture_region():
    bbox = BoundingBox(x=10, y=10, width=100, height=80)
    img = screen_capture_engine.capture_region(bbox)
    assert img is not None
    assert img.size[0] == 100
    assert img.size[1] == 80


def test_capture_window():
    img, win = screen_capture_engine.capture_window("Desktop")
    assert img is not None
    assert win is not None
