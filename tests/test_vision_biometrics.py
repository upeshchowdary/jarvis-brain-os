"""Unit tests for vision.face_detector and vision.gesture_detector modules."""

from PIL import Image
from vision.face_detector import face_detector
from vision.gesture_detector import gesture_detector


def test_face_detector_synthetic_image():
    img = Image.new("RGB", (300, 300), color=(200, 200, 200))
    faces = face_detector.detect_faces(img)
    assert isinstance(faces, list)


def test_gesture_detector_synthetic_image():
    img = Image.new("RGB", (300, 300), color=(200, 200, 200))
    gestures = gesture_detector.detect_gestures(img)
    assert isinstance(gestures, list)
