"""Unit tests for vision.screen_memory and vision.diff_detector modules."""

from PIL import Image
from vision.screen_memory import screen_memory
from vision.diff_detector import diff_detector
from vision.environment import EnvironmentState, WindowInfo, CursorState, BoundingBox


def test_screen_memory_caching():
    img = Image.new("RGB", (50, 50), color=(100, 100, 100))
    h_str = screen_memory.compute_image_hash(img)
    assert isinstance(h_str, str)
    assert len(h_str) > 0

    state = EnvironmentState(active_application="VS Code", summary="Code editor open.")
    screen_memory.add_snapshot(h_str, state)

    cached = screen_memory.get_cached_state(h_str)
    assert cached is not None
    assert cached.active_application == "VS Code"


def test_diff_detector_computation():
    s1 = EnvironmentState(
        cursor=CursorState(x=10, y=10),
        windows=[WindowInfo(title="Chrome", app_name="Chrome")]
    )
    s2 = EnvironmentState(
        cursor=CursorState(x=500, y=500),
        windows=[WindowInfo(title="Chrome", app_name="Chrome"), WindowInfo(title="Terminal", app_name="Terminal")]
    )

    diff = diff_detector.compute_diff(s1, s2)
    assert diff.has_changes is True
    assert "Terminal" in diff.new_windows
    assert diff.cursor_moved is True
