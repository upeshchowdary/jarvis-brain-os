"""Unit tests for vision.environment domain models and Pydantic validation."""

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
    EnvironmentState,
)


def test_bounding_box():
    bbox = BoundingBox(x=100, y=200, width=400, height=300)
    assert bbox.x == 100
    assert bbox.y == 200
    assert bbox.width == 400
    assert bbox.height == 300
    assert bbox.center == (300, 350)


def test_environment_state_serialization():
    cursor = CursorState(x=500, y=300, cursor_type="pointer")
    monitor = MonitorInfo(index=0, name="Main Display", resolution=(1920, 1080))
    window = WindowInfo(
        window_id=101,
        title="main.py - Visual Studio Code",
        app_name="Visual Studio Code",
        is_active=True,
        bounds=BoundingBox(x=0, y=0, width=1920, height=1080)
    )
    ui_elem = UIElement(
        id="btn_submit",
        element_type=UIElementType.BUTTON,
        label="Submit Query",
        bounds=BoundingBox(x=100, y=100, width=80, height=30)
    )
    ocr_item = OCRTextItem(
        text="def main():",
        bounds=BoundingBox(x=20, y=40, width=150, height=20),
        text_type="code"
    )

    env_state = EnvironmentState(
        active_application=window.app_name,
        active_window=window.title,
        cursor=cursor,
        monitors=[monitor],
        windows=[window],
        ui=[ui_elem],
        visible_text=[ocr_item],
        summary="VS Code is active with main.py open.",
        confidence=0.99
    )

    data = env_state.model_dump()
    assert data["active_application"] == "Visual Studio Code"
    assert data["cursor"]["x"] == 500
    assert data["ui"][0]["element_type"] == "button"
    assert data["visible_text"][0]["text"] == "def main():"
    assert data["confidence"] == 0.99
