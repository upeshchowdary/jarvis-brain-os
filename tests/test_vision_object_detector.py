"""Unit tests for vision.object_detector module."""

from PIL import Image, ImageDraw
from vision.object_detector import object_detector, ObjectDetector
from vision.environment import DetectedObject


def test_object_detector_synthetic_image():
    img = Image.new("RGB", (800, 600), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    # Draw simulated monitor rectangle
    draw.rectangle([100, 100, 700, 500], fill=(30, 30, 30), outline=(0, 0, 0))

    objects = object_detector.detect_objects(img)
    assert isinstance(objects, list)
    assert len(objects) > 0
    assert isinstance(objects[0], DetectedObject)
