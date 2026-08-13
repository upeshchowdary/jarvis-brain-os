"""Unit tests for vision.ui_detector module."""

from PIL import Image, ImageDraw
from vision.ui_detector import ui_detector, UIDetector
from vision.environment import BoundingBox, UIElement, UIElementType, OCRTextItem


def test_ui_detector_with_ocr_items():
    ocr1 = OCRTextItem(text="Submit", bounds=BoundingBox(x=100, y=100, width=80, height=30))
    ocr2 = OCRTextItem(text="Remember me", bounds=BoundingBox(x=100, y=150, width=120, height=20))
    ocr3 = OCRTextItem(text="https://github.com", bounds=BoundingBox(x=50, y=10, width=400, height=25))

    img = Image.new("RGB", (600, 400), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 180, 130], outline=(0, 0, 0))

    elements = ui_detector.detect_ui_elements(img, ocr_items=[ocr1, ocr2, ocr3])
    assert isinstance(elements, list)
    assert len(elements) >= 3

    types = [el.element_type for el in elements]
    assert UIElementType.BUTTON in types
    assert UIElementType.CHECKBOX in types
    assert UIElementType.ADDRESS_BAR in types


def test_bounds_overlap():
    b1 = BoundingBox(x=10, y=10, width=100, height=100)
    b2 = BoundingBox(x=20, y=20, width=80, height=80)
    b3 = BoundingBox(x=300, y=300, width=50, height=50)

    assert UIDetector._bounds_overlap(b1, b2) is True
    assert UIDetector._bounds_overlap(b1, b3) is False
