"""JARVIS Fast Screen Capture v3 — Ultra-Low-Latency Pipeline.

Capture priority (fastest first):
  1. d3dshot (DXGI Desktop Duplication)  — ~3-5ms, zero-copy GPU path
  2. mss (native ctypes)                — ~15-25ms, no disk I/O
  3. PIL.ImageGrab                       — ~50-100ms, legacy fallback

All methods return PIL.Image.Image in RGB mode for downstream compatibility.
Thread-safe via thread-local singleton instances.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional, Tuple

from loguru import logger

# ---------------------------------------------------------------------------
# Optional dependency detection
# ---------------------------------------------------------------------------

try:
    import d3dshot
    HAS_D3DSHOT = True
except ImportError:
    HAS_D3DSHOT = False

try:
    import mss
    import mss.tools
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Thread-local singleton managers
# ---------------------------------------------------------------------------
_local = threading.local()


def _get_d3dshot_instance() -> Optional[Any]:
    """Get or create a thread-local d3dshot instance (DXGI capture)."""
    if not HAS_D3DSHOT:
        return None
    if not hasattr(_local, "_d3d"):
        try:
            _local._d3d = d3dshot.create(capture_output="numpy")
            logger.info("[JARVIS][FastCapture] DXGI capture initialized via d3dshot")
        except Exception as e:
            logger.debug(f"[JARVIS][FastCapture] d3dshot init failed: {e}")
            _local._d3d = None
    return _local._d3d


def _get_mss_instance() -> Optional[Any]:
    """Get or create a thread-local mss instance."""
    if not HAS_MSS:
        return None
    if not hasattr(_local, "_mss"):
        try:
            _local._mss = mss.mss()
        except Exception as e:
            logger.debug(f"[JARVIS][FastCapture] mss init failed: {e}")
            _local._mss = None
    return _local._mss


# ---------------------------------------------------------------------------
# Capture methods (ordered by speed)
# ---------------------------------------------------------------------------

def _capture_d3dshot(monitor_idx: int = 0) -> Optional[Any]:
    """Capture via DXGI Desktop Duplication (~3-5ms)."""
    d3d = _get_d3dshot_instance()
    if d3d is None:
        return None
    try:
        frame = d3d.screenshot()  # returns numpy RGB array
        if frame is not None and HAS_PIL:
            return Image.fromarray(frame)
    except Exception as e:
        logger.debug(f"[JARVIS][FastCapture] d3dshot capture failed: {e}")
    return None


def _capture_mss(monitor_idx: int = 0) -> Optional[Any]:
    """Capture via mss native ctypes (~15-25ms)."""
    sct = _get_mss_instance()
    if sct is None or not HAS_PIL:
        return None
    try:
        # monitor 0 = all monitors combined, 1+ = individual
        monitors = sct.monitors
        target = monitors[monitor_idx] if monitor_idx < len(monitors) else monitors[0]
        sct_img = sct.grab(target)
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    except Exception as e:
        logger.debug(f"[JARVIS][FastCapture] mss capture failed: {e}")
    return None


def _capture_pil() -> Optional[Any]:
    """Capture via PIL ImageGrab (~50-100ms fallback)."""
    if not HAS_PIL:
        return None
    try:
        from PIL import ImageGrab
        return ImageGrab.grab(all_screens=True)
    except Exception:
        try:
            from PIL import ImageGrab
            return ImageGrab.grab()
        except Exception as e:
            logger.warning(f"[JARVIS][FastCapture] PIL.ImageGrab failed: {e}")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def capture_full_screen(monitor_idx: int = 0) -> Optional[Any]:
    """
    Capture entire desktop as PIL Image (RGB).

    Tries capture methods in speed order:
      1. d3dshot (DXGI)  — ~3-5ms
      2. mss             — ~15-25ms
      3. PIL.ImageGrab   — ~50-100ms

    Returns None only if all methods fail.
    """
    t0 = time.perf_counter()

    # Priority 1: DXGI
    img = _capture_d3dshot(monitor_idx)
    if img is not None:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"[JARVIS][FastCapture] DXGI capture: {img.size[0]}×{img.size[1]} in {elapsed:.1f}ms")
        return img

    # Priority 2: mss
    img = _capture_mss(monitor_idx)
    if img is not None:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"[JARVIS][FastCapture] mss capture: {img.size[0]}×{img.size[1]} in {elapsed:.1f}ms")
        return img

    # Priority 3: PIL
    img = _capture_pil()
    if img is not None:
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"[JARVIS][FastCapture] PIL capture: {img.size[0]}×{img.size[1]} in {elapsed:.1f}ms")
        return img

    logger.error("[JARVIS][FastCapture] All capture methods failed")
    return None


def capture_region(x: int, y: int, width: int, height: int) -> Optional[Any]:
    """
    Capture a specific screen region as PIL Image.
    Uses mss for precision region capture, PIL as fallback.
    """
    t0 = time.perf_counter()

    # mss region capture (precise and fast)
    if HAS_MSS and HAS_PIL:
        try:
            sct = _get_mss_instance()
            if sct:
                region = {"left": x, "top": y, "width": width, "height": height}
                sct_img = sct.grab(region)
                img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                elapsed = (time.perf_counter() - t0) * 1000
                logger.debug(f"[JARVIS][FastCapture] Region capture: {width}×{height} in {elapsed:.1f}ms")
                return img
        except Exception as e:
            logger.debug(f"[JARVIS][FastCapture] mss region capture failed: {e}")

    # PIL fallback
    if HAS_PIL:
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab(bbox=(x, y, x + width, y + height))
            return img
        except Exception as e:
            logger.warning(f"[JARVIS][FastCapture] PIL region capture failed: {e}")

    return None


def capture_as_numpy() -> Optional[Any]:
    """
    Capture screen directly as numpy array (RGB, uint8).
    Avoids PIL conversion overhead for OpenCV processing.
    """
    if not HAS_NUMPY:
        img = capture_full_screen()
        return np.array(img) if img and HAS_NUMPY else None

    # d3dshot returns numpy directly — zero-copy
    d3d = _get_d3dshot_instance()
    if d3d is not None:
        try:
            frame = d3d.screenshot()
            if frame is not None:
                return frame  # already numpy RGB
        except Exception:
            pass

    # Fallback: capture PIL then convert
    img = capture_full_screen()
    if img is not None:
        try:
            return np.array(img.convert("RGB"))
        except Exception:
            pass
    return None


def get_screen_size() -> Tuple[int, int]:
    """Return desktop resolution (width, height)."""
    if HAS_MSS:
        try:
            sct = _get_mss_instance()
            if sct:
                m = sct.monitors[0]
                return m["width"], m["height"]
        except Exception:
            pass

    if HAS_D3DSHOT:
        try:
            d3d = _get_d3dshot_instance()
            if d3d and d3d.displays:
                disp = d3d.displays[0]
                return disp.resolution[0], disp.resolution[1]
        except Exception:
            pass

    return 1920, 1080


def is_mss_available() -> bool:
    """Return True if mss is installed and working."""
    return HAS_MSS


def is_dxgi_available() -> bool:
    """Return True if d3dshot (DXGI) is available."""
    return HAS_D3DSHOT


def get_capture_backend() -> str:
    """Return name of the fastest available capture backend."""
    if HAS_D3DSHOT:
        return "dxgi"
    if HAS_MSS:
        return "mss"
    if HAS_PIL:
        return "pil"
    return "none"
