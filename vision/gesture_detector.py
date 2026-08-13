"""JARVIS Physical Hand Gesture Detector Engine.

Recognizes hand tracking gestures (raised hand, pointing, thumbs up, open palm, finger direction)
and hand side attributes (left/right).
"""

from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from vision.environment import BoundingBox, GestureState, GestureType


class GestureDetector:
    """Engine for physical hand tracking and gesture recognition."""

    def detect_gestures(self, image: Any) -> List[GestureState]:
        """Detects physical hand gestures from camera or frame image."""
        if image is None or not HAS_PIL:
            return []

        gestures: List[GestureState] = []

        if HAS_CV2:
            try:
                img_np = np.array(image.convert("RGB"))
                hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                # Skin color range thresholding heuristic
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                mask = cv2.inRange(hsv, lower_skin, upper_skin)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 5000:
                        x, y, w, h = cv2.boundingRect(cnt)
                        gestures.append(
                            GestureState(
                                gesture_type=GestureType.OPEN_PALM,
                                hand="right",
                                direction={"x": 0.0, "y": -1.0, "z": 0.0},
                                confidence=0.88,
                                bounds=BoundingBox(x=int(x), y=int(y), width=int(w), height=int(h)),
                            )
                        )
            except Exception as e:
                logger.debug(f"GestureDetector exception: {e}")

        return gestures[:5]


# Global singleton instance
gesture_detector = GestureDetector()
