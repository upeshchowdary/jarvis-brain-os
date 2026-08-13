"""JARVIS GUI & UI Element Detection Module.

Detects interactive user interface components (Buttons, Textboxes, Checkboxes, Dropdowns,
Menus, Tabs, Taskbars, Address bars, Toolbars, Popups) with bounding boxes and semantic labels.
"""

import uuid
from typing import List, Dict, Any, Optional, Tuple
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

from vision.environment import BoundingBox, UIElement, UIElementType, OCRTextItem


class UIDetector:
    """Engine for detecting and classifying GUI interaction elements on screens."""

    def __init__(self) -> None:
        self._button_keywords = {"submit", "ok", "cancel", "save", "delete", "apply", "close", "search", "login", "send", "confirm"}
        self._checkbox_keywords = {"remember", "agree", "enable", "disable", "keep", "select", "check"}

    def detect_ui_elements(
        self,
        image: Any,
        ocr_items: Optional[List[OCRTextItem]] = None,
    ) -> List[UIElement]:
        """Detects interactive UI elements from screen image and OCR text items.

        Returns a list of structured UIElement Pydantic models.
        """
        if image is None or not HAS_PIL:
            return []

        elements: List[UIElement] = []
        ocr_list = ocr_items or []

        # 1. Structural Contour Analysis using OpenCV
        contours_found = []
        if HAS_CV2:
            try:
                contours_found = self._detect_rect_contours(image)
            except Exception as e:
                logger.debug(f"UIDetector contour detection fallback: {e}")

        # 2. Map OCR text items to detected contours or create text-bound UI elements
        used_ocr_indices = set()
        for idx, ocr in enumerate(ocr_list):
            txt_lower = ocr.text.lower().strip()
            if not txt_lower:
                continue

            elem_type = UIElementType.OTHER
            is_clickable = True

            if any(k in txt_lower for k in self._button_keywords):
                elem_type = UIElementType.BUTTON
            elif any(k in txt_lower for k in self._checkbox_keywords):
                elem_type = UIElementType.CHECKBOX
            elif "http" in txt_lower or "www." in txt_lower or "/" in txt_lower:
                elem_type = UIElementType.ADDRESS_BAR
            elif len(txt_lower) < 25 and not txt_lower.endswith("."):
                elem_type = UIElementType.BUTTON

            if elem_type != UIElementType.OTHER:
                used_ocr_indices.add(idx)
                elem_id = f"ui_{elem_type.value}_{uuid.uuid4().hex[:6]}"
                elements.append(
                    UIElement(
                        id=elem_id,
                        element_type=elem_type,
                        label=ocr.text,
                        value=None,
                        bounds=ocr.bounds,
                        confidence=round(ocr.confidence * 0.95, 2),
                        is_clickable=is_clickable,
                        is_focused=False,
                    )
                )

        # 3. Process remaining contours as unlabeled textboxes, buttons, or checkboxes
        for box in contours_found:
            # Avoid duplicate bounds
            if not any(self._bounds_overlap(box, el.bounds) for el in elements):
                aspect_ratio = box.width / float(box.height) if box.height > 0 else 1.0
                elem_type = UIElementType.TEXTBOX if aspect_ratio > 3.0 else (
                    UIElementType.CHECKBOX if 0.8 <= aspect_ratio <= 1.2 and box.width < 35 else UIElementType.BUTTON
                )

                elem_id = f"ui_{elem_type.value}_{uuid.uuid4().hex[:6]}"
                elements.append(
                    UIElement(
                        id=elem_id,
                        element_type=elem_type,
                        label=f"Unlabeled {elem_type.value.capitalize()}",
                        value=None,
                        bounds=box,
                        confidence=0.85,
                        is_clickable=True,
                        is_focused=False,
                    )
                )

        return elements

    def _detect_rect_contours(self, image: Any) -> List[BoundingBox]:
        """Detects rectangular UI component boundaries using OpenCV contour analysis."""
        boxes: List[BoundingBox] = []
        if not HAS_CV2 or image is None:
            return boxes

        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:  # Quadrilateral / Rectangular bounding box
                x, y, w, h = cv2.boundingRect(cnt)
                if 20 <= w <= 800 and 15 <= h <= 200:
                    boxes.append(BoundingBox(x=x, y=y, width=w, height=h))

        return boxes[:30]

    @staticmethod
    def _bounds_overlap(b1: BoundingBox, b2: BoundingBox, threshold: float = 0.5) -> bool:
        """Determines whether two bounding boxes significantly overlap."""
        x_left = max(b1.x, b2.x)
        y_top = max(b1.y, b2.y)
        x_right = min(b1.x + b1.width, b2.x + b2.width)
        y_bottom = min(b1.y + b1.height, b2.y + b2.height)

        if x_right < x_left or y_bottom < y_top:
            return False

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = b1.width * b1.height
        area2 = b2.width * b2.height

        min_area = min(area1, area2)
        if min_area == 0:
            return False

        return (intersection / float(min_area)) >= threshold


# Global singleton instance
ui_detector = UIDetector()
