"""Unit tests for vision.ocr module."""

from PIL import Image, ImageDraw, ImageFont
from vision.ocr import ocr_engine, OCREngine
from vision.environment import BoundingBox, OCRTextItem


def test_classify_text_type():
    assert ocr_engine._classify_text_type("def main():") == "code"
    assert ocr_engine._classify_text_type("2026-08-03 INFO: System start") == "log"
    assert ocr_engine._classify_text_type("x + y = 10") == "math"
    assert ocr_engine._classify_text_type("Hello World") == "printed"


def test_sort_reading_order():
    item1 = OCRTextItem(text="Second Row", bounds=BoundingBox(x=10, y=100, width=50, height=20))
    item2 = OCRTextItem(text="First Row Left", bounds=BoundingBox(x=10, y=10, width=50, height=20))
    item3 = OCRTextItem(text="First Row Right", bounds=BoundingBox(x=100, y=10, width=50, height=20))

    sorted_items = ocr_engine._sort_reading_order([item1, item2, item3])
    assert sorted_items[0].text == "First Row Left"
    assert sorted_items[1].text == "First Row Right"
    assert sorted_items[2].text == "Second Row"


def test_extract_text_items_synthetic_image():
    # Create synthetic image with text drawing
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "Hello JARVIS", fill=(0, 0, 0))

    items = ocr_engine.extract_text_items(img)
    assert isinstance(items, list)
    full_text = ocr_engine.extract_full_text(img)
    assert isinstance(full_text, str)
