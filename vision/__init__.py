"""JARVIS Vision & Environment Understanding Package v3."""

from vision.environment import (
    BoundingBox,
    CursorState,
    MonitorInfo,
    WindowInfo,
    UIElementType,
    UIElement,
    OCRTextItem,
    DetectedObject,
    FaceState,
    GestureType,
    GestureState,
    TableCell,
    DocumentStructure,
    ChartData,
    DiffSummary,
    EnvironmentState,
)

from vision.screen_capture import ScreenCaptureEngine, screen_capture_engine
from vision.camera import CameraModule, camera_module
from vision.ocr import OCREngine, ocr_engine
from vision.visual_reasoner import VisualReasoner, visual_reasoner
from vision.ui_detector import UIDetector, ui_detector
from vision.layout_analyzer import LayoutAnalyzer, layout_analyzer
from vision.image_analyzer import ImageAnalyzer, image_analyzer
from vision.object_detector import ObjectDetector, object_detector
from vision.face_detector import FaceDetector, face_detector
from vision.gesture_detector import GestureDetector, gesture_detector
from vision.document_reader import DocumentReader, document_reader
from vision.chart_reader import ChartReader, chart_reader
from vision.screen_memory import ScreenMemory, screen_memory
from vision.diff_detector import DiffDetector, diff_detector
from vision.manager import VisionManager, vision_manager

__all__ = [
    "BoundingBox",
    "CursorState",
    "MonitorInfo",
    "WindowInfo",
    "UIElementType",
    "UIElement",
    "OCRTextItem",
    "DetectedObject",
    "FaceState",
    "GestureType",
    "GestureState",
    "TableCell",
    "DocumentStructure",
    "ChartData",
    "DiffSummary",
    "EnvironmentState",
    "ScreenCaptureEngine",
    "screen_capture_engine",
    "CameraModule",
    "camera_module",
    "OCREngine",
    "ocr_engine",
    "VisualReasoner",
    "visual_reasoner",
    "UIDetector",
    "ui_detector",
    "LayoutAnalyzer",
    "layout_analyzer",
    "ImageAnalyzer",
    "image_analyzer",
    "ObjectDetector",
    "object_detector",
    "FaceDetector",
    "face_detector",
    "GestureDetector",
    "gesture_detector",
    "DocumentReader",
    "document_reader",
    "ChartReader",
    "chart_reader",
    "ScreenMemory",
    "screen_memory",
    "DiffDetector",
    "diff_detector",
    "VisionManager",
    "vision_manager",
]
