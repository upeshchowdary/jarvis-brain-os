"""JARVIS Temporal Screen Memory & Frame Hash Cache Module.

Maintains frame history buffer, image perceptual hashes, and cached EnvironmentState
snapshots to avoid re-analyzing identical frames.
"""

import hashlib
from typing import Dict, Any, List, Optional
from loguru import logger

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from vision.environment import EnvironmentState


class ScreenMemory:
    """Buffer and cache manager for temporal screenshot history and image hashing."""

    def __init__(self, max_history: int = 20) -> None:
        self.max_history = max_history
        self._history: List[EnvironmentState] = []
        self._hash_cache: Dict[str, EnvironmentState] = {}
        self._last_image_hash: Optional[str] = None

    def compute_image_hash(self, image: Any) -> str:
        """Computes fast MD5 perceptual hash of image payload."""
        if image is None or not HAS_PIL:
            return ""

        try:
            small = image.resize((64, 64)).convert("L")
            raw_bytes = small.tobytes()
            return hashlib.md5(raw_bytes).hexdigest()
        except Exception as e:
            logger.debug(f"Image hash exception: {e}")
            return ""

    def get_cached_state(self, hash_str: str) -> Optional[EnvironmentState]:
        """Returns cached EnvironmentState if frame hash matches."""
        if not hash_str:
            return None
        return self._hash_cache.get(hash_str)

    def add_snapshot(self, hash_str: str, state: EnvironmentState) -> None:
        """Stores a snapshot in history buffer and hash cache."""
        self._history.append(state)
        if len(self._history) > self.max_history:
            self._history.pop(0)

        if hash_str:
            self._hash_cache[hash_str] = state
            self._last_image_hash = hash_str

    def get_last_snapshot(self) -> Optional[EnvironmentState]:
        """Returns the most recent EnvironmentState snapshot from memory."""
        if not self._history:
            return None
        return self._history[-1]


# Global singleton instance
screen_memory = ScreenMemory()
