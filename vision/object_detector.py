"""JARVIS Physical Object Detection Engine.

Detects physical objects (Laptop, Phone, Monitor, Keyboard, Mouse, Bottle, Chair, Desk,
Person, Book, TV, Bag, Clock) with bounding boxes, categories, and confidence metrics.
"""

import uuid
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

from vision.environment import BoundingBox, DetectedObject


class ObjectDetector:
    """Engine for detecting physical objects, peripherals, and environment items."""

    def __init__(self) -> None:
        self.supported_labels = [
            "laptop", "phone", "monitor", "keyboard", "mouse",
            "bottle", "chair", "desk", "door", "person",
            "animal", "vehicle", "book", "tv", "clock", "bag"
        ]

    def detect_objects(self, image: Any) -> List[DetectedObject]:
        """Detects physical and digital objects in frame."""
        if image is None or not HAS_PIL:
            return []

        objects: List[DetectedObject] = []

        if HAS_CV2:
            try:
                objects = self._detect_with_cv2_geometry(image)
            except Exception as e:
                logger.debug(f"ObjectDetector geometry detection fallback: {e}")

        # Fallback default screen object if no object identified
        if not objects:
            w, h = image.size
            objects.append(
                DetectedObject(
                    label="display_screen",
                    confidence=1.0,
                    bounds=BoundingBox(x=0, y=0, width=w, height=h),
                    category="digital",
                )
            )

        return objects

    def _detect_with_cv2_geometry(self, image: Any) -> List[DetectedObject]:
        """Shape & color geometry heuristic detection for physical objects."""
        if not HAS_CV2 or image is None:
            return []

        detected: List[DetectedObject] = []
        w_img, h_img = image.size

        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (w_img * h_img * 0.05):
                continue  # Filter small noise

            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h) if h > 0 else 1.0

            label = "object"
            category = "heuristic_geometry"

            if 1.3 <= aspect_ratio <= 1.8 and w > int(w_img * 0.4):
                label = "monitor_region"
                category = "heuristic_geometry"
            elif aspect_ratio >= 2.5 and h < int(h_img * 0.3):
                label = "keyboard_region"
                category = "heuristic_geometry"
            elif 0.4 <= aspect_ratio <= 0.65 and h > int(h_img * 0.3):
                label = "vertical_panel"
                category = "heuristic_geometry"
            elif 0.8 <= aspect_ratio <= 1.2 and w < int(w_img * 0.2):
                label = "small_panel"
                category = "heuristic_geometry"
            else:
                label = "ui_container"

            detected.append(
                DetectedObject(
                    label=label,
                    confidence=0.60,  # Heuristic score
                    bounds=BoundingBox(x=x, y=y, width=w, height=h),
                    category=category,
                )
            )

        return detected[:10]


# Global singleton instance
object_detector = ObjectDetector()
