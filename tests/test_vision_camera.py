"""Unit tests for vision.camera module."""

import pytest
from vision.camera import camera_module, CameraModule


def test_is_camera_available_bool_return():
    is_avail = camera_module.is_camera_available(0)
    assert isinstance(is_avail, bool)


def test_list_available_cameras():
    cams = camera_module.list_available_cameras(max_devices_to_check=2)
    assert isinstance(cams, list)


def test_capture_frame_sync():
    img, is_hw = camera_module.capture_frame(target_resolution=(320, 240))
    assert img is not None
    assert img.size == (320, 240)
    assert isinstance(is_hw, bool)


@pytest.mark.asyncio
async def test_capture_frame_async():
    img, is_hw = await camera_module.capture_frame_async(target_resolution=(640, 480))
    assert img is not None
    assert img.size == (640, 480)
    assert isinstance(is_hw, bool)


def test_get_last_cached_frame():
    camera_module.capture_frame(target_resolution=(100, 100))
    cached = camera_module.get_last_cached_frame()
    assert cached is not None
    assert cached.size == (100, 100)
