"""JARVIS Vision Engine Environment Models & Schemas.

Defines Pydantic models for structured visual perception outputs, UI elements,
objects, faces, gestures, documents, charts, and screen diff states.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box in pixel coordinates [x, y, width, height]."""

    x: int = Field(..., description="Top-left X coordinate in pixels")
    y: int = Field(..., description="Top-left Y coordinate in pixels")
    width: int = Field(..., description="Width of bounding box in pixels")
    height: int = Field(..., description="Height of bounding box in pixels")

    @property
    def center(self) -> Tuple[int, int]:
        """Calculates center coordinate (x, y)."""
        return (self.x + self.width // 2, self.y + self.height // 2)


class CursorState(BaseModel):
    """Cursor state and position."""

    x: int = Field(default=0, description="Current cursor X coordinate")
    y: int = Field(default=0, description="Current cursor Y coordinate")
    visible: bool = Field(default=True, description="Whether cursor is visible")
    cursor_type: str = Field(default="arrow", description="Type of cursor icon (arrow, beam, pointer, wait, etc.)")


class MonitorInfo(BaseModel):
    """Monitor display properties."""

    index: int = Field(default=0, description="Monitor index (0 is primary)")
    name: str = Field(default="Primary Monitor", description="Monitor display name")
    resolution: Tuple[int, int] = Field(default=(1920, 1080), description="(width, height) in pixels")
    scaling: float = Field(default=1.0, description="DPI scale factor")
    is_primary: bool = Field(default=True, description="Whether this is primary display")


class WindowInfo(BaseModel):
    """Information about an open application window."""

    window_id: int = Field(default=0, description="Native window handle ID")
    title: str = Field(default="", description="Window title text")
    app_name: str = Field(default="", description="Application executable or process name")
    is_active: bool = Field(default=False, description="Whether window is currently focused/active")
    bounds: BoundingBox = Field(default_factory=lambda: BoundingBox(x=0, y=0, width=1920, height=1080))
    monitor_index: int = Field(default=0, description="Index of monitor containing window")
    z_index: int = Field(default=0, description="Layer depth index")


class UIElementType(str, Enum):
    BUTTON = "button"
    TEXTBOX = "textbox"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    LIST = "list"
    MENU = "menu"
    TAB = "tab"
    NAVBAR = "navbar"
    ICON = "icon"
    TASKBAR_ITEM = "taskbar_item"
    ADDRESS_BAR = "address_bar"
    CONTEXT_MENU = "context_menu"
    TOOLBAR = "toolbar"
    STATUS_BAR = "status_bar"
    POPUP = "popup"
    OTHER = "other"


class UIElement(BaseModel):
    """Detected GUI interaction element."""

    id: str = Field(..., description="Unique identifier for element")
    element_type: UIElementType = Field(default=UIElementType.OTHER, description="UI component classification")
    label: str = Field(default="", description="Text label or name of element")
    value: Optional[str] = Field(default=None, description="Current value/text inside element")
    bounds: BoundingBox = Field(..., description="Element screen coordinates")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score")
    is_clickable: bool = Field(default=True, description="Whether element can be clicked")
    is_focused: bool = Field(default=False, description="Whether element is currently focused")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional properties (checked, active, etc.)")


class OCRTextItem(BaseModel):
    """Extracted text item with layout coordinates."""

    text: str = Field(..., description="Extracted text string")
    bounds: BoundingBox = Field(..., description="Bounding box surrounding text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="OCR confidence score")
    text_type: str = Field(default="printed", description="Type of text: printed, handwritten, code, log, math, table_cell")


class DetectedObject(BaseModel):
    """Physical real-world or digital object detected in frame."""

    label: str = Field(..., description="Object classification name (e.g. laptop, bottle, person)")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence score")
    bounds: BoundingBox = Field(..., description="Object bounding box")
    category: str = Field(default="physical", description="Category: physical, digital, peripheral")


class FaceState(BaseModel):
    """Detected human face properties (No identity PII)."""

    bounds: BoundingBox = Field(..., description="Face bounding box")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    head_orientation: Dict[str, float] = Field(
        default_factory=lambda: {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
        description="Head orientation angles in degrees"
    )
    eye_gaze: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0},
        description="Eye gaze direction vector"
    )
    attention_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Attention level toward screen/camera")
    expression: str = Field(default="neutral", description="Face expression summary")


class GestureType(str, Enum):
    RAISED_HAND = "raised_hand"
    POINTING = "pointing"
    THUMBS_UP = "thumbs_up"
    OPEN_PALM = "open_palm"
    FINGER_DIRECTION = "finger_direction"
    NONE = "none"


class GestureState(BaseModel):
    """Hand tracking and gesture state."""

    gesture_type: GestureType = Field(default=GestureType.NONE, description="Gesture classification")
    hand: str = Field(default="right", description="Hand side: left, right")
    direction: Dict[str, float] = Field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    bounds: Optional[BoundingBox] = Field(default=None, description="Hand bounding box")


class TableCell(BaseModel):
    row: int
    col: int
    text: str
    bounds: Optional[BoundingBox] = None


class DocumentStructure(BaseModel):
    """Parsed document / PDF structure."""

    title: str = Field(default="", description="Document title")
    sections: List[str] = Field(default_factory=list, description="Section headings")
    paragraphs: List[str] = Field(default_factory=list, description="Extracted paragraph texts")
    tables: List[List[TableCell]] = Field(default_factory=list, description="Extracted table grids")
    footnotes: List[str] = Field(default_factory=list, description="Footnotes or citations")


class ChartData(BaseModel):
    """Extracted chart or graph structure."""

    chart_type: str = Field(default="bar", description="Chart type: pie, line, bar, scatter, heatmap")
    title: str = Field(default="", description="Chart title")
    x_label: str = Field(default="", description="X-axis label")
    y_label: str = Field(default="", description="Y-axis label")
    legend: List[str] = Field(default_factory=list, description="Legend items")
    estimated_values: Dict[str, Any] = Field(default_factory=dict, description="Parsed numerical series or data points")


class DiffSummary(BaseModel):
    """Delta/differences between consecutive screen states."""

    has_changes: bool = Field(default=False, description="Whether any meaningful visual changes occurred")
    new_windows: List[str] = Field(default_factory=list, description="Titles of newly opened windows")
    closed_windows: List[str] = Field(default_factory=list, description="Titles of closed windows")
    popup_appeared: bool = Field(default=False, description="Whether a new popup or modal appeared")
    cursor_moved: bool = Field(default=False, description="Whether cursor position shifted significantly")
    changed_text: List[str] = Field(default_factory=list, description="New or altered text lines")
    changed_elements: List[str] = Field(default_factory=list, description="Altered UI element IDs")


class EnvironmentState(BaseModel):
    """Complete internal structured environment representation for JARVIS Brain."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp of perception capture"
    )
    active_application: str = Field(default="Unknown", description="Name of currently focused application")
    active_window: str = Field(default="Unknown", description="Title of currently focused window")
    cursor: CursorState = Field(default_factory=CursorState)
    monitors: List[MonitorInfo] = Field(default_factory=list)
    windows: List[WindowInfo] = Field(default_factory=list)
    ui: List[UIElement] = Field(default_factory=list)
    visible_text: List[OCRTextItem] = Field(default_factory=list)
    objects: List[DetectedObject] = Field(default_factory=list)
    faces: List[FaceState] = Field(default_factory=list)
    gestures: List[GestureState] = Field(default_factory=list)
    document: Optional[DocumentStructure] = Field(default=None)
    chart: Optional[ChartData] = Field(default=None)
    diff: Optional[DiffSummary] = Field(default=None)
    summary: str = Field(default="", description="Semantic summary description of visible screen and environment")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall environment perception confidence")
