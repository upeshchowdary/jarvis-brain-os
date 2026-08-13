"""JARVIS Camera Ingestion & Perception Engine.

NOTE: Camera hardware access is DISABLED.
JARVIS uses screen capture (PIL ImageGrab) for visual perception — not the webcam.
All methods return synthetic placeholder frames to preserve API compatibility.
"""

import asyncio
from typing import List, Optional, Tuple
from loguru import logger

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class CameraModule:
    """Camera module — DISABLED. JARVIS uses screen capture only."""

    def __init__(self, default_device_index: int = 0) -> None:
        self.default_device_index = default_device_index
        self._cached_frame: Optional[object] = None

    def is_camera_available(self, device_index: Optional[int] = None) -> bool:
        """Camera is DISABLED. Always returns False."""
        return False

    def list_available_cameras(self, max_devices_to_check: int = 4) -> List[int]:
        """Camera is DISABLED. Always returns empty list."""
        return []

    def capture_frame(
        self,
        device_index: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[object], bool]:
        """Camera is DISABLED. Returns a synthetic black placeholder frame.

        Returns:
            Tuple[PIL.Image, is_hardware_frame=False]
        """
        logger.debug("Camera is disabled. Returning synthetic frame (use screen capture instead).")
        res = target_resolution or (640, 480)
        if HAS_PIL:
            img = Image.new("RGB", res, color=(20, 20, 24))
        else:
            img = None
        self._cached_frame = img
        return img, False

    async def capture_frame_async(
        self,
        device_index: Optional[int] = None,
        target_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[object], bool]:
        """Camera is DISABLED. Returns synthetic frame asynchronously."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.capture_frame(device_index=device_index, target_resolution=target_resolution),
        )

    def get_last_cached_frame(self) -> Optional[object]:
        """Returns the most recently captured (or synthetic) frame from cache."""
        return self._cached_frame


# Global camera module singleton
camera_module = CameraModule()
