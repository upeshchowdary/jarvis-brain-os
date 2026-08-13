"""Unit tests for vision.layout_analyzer and vision.image_analyzer."""

from PIL import Image
from vision.layout_analyzer import layout_analyzer
from vision.image_analyzer import image_analyzer
from vision.environment import WindowInfo, MonitorInfo, BoundingBox, UIElement, UIElementType


def test_layout_analyzer_maximized():
    win = WindowInfo(
        window_id=1,
        title="Visual Studio Code",
        app_name="Code",
        is_active=True,
        bounds=BoundingBox(x=0, y=0, width=1920, height=1080)
    )
    mon = MonitorInfo(index=0, resolution=(1920, 1080))

    layout = layout_analyzer.analyze_window_layout([win], [mon])
    assert layout["layout_style"] == "maximized"
    assert layout["active_application"] == "Code"


def test_group_ui_elements_by_region():
    btn_header = UIElement(id="b1", element_type=UIElementType.BUTTON, label="File", bounds=BoundingBox(x=10, y=20, width=40, height=20))
    btn_taskbar = UIElement(id="b2", element_type=UIElementType.BUTTON, label="Start", bounds=BoundingBox(x=10, y=1050, width=40, height=30))

    grouped = layout_analyzer.group_ui_elements_by_region([btn_header, btn_taskbar], screen_resolution=(1920, 1080))
    assert len(grouped["header_navbar"]) == 1
    assert len(grouped["taskbar"]) == 1


def test_image_analyzer_properties():
    img = Image.new("RGB", (200, 200), color=(255, 0, 0))  # Red image
    props = image_analyzer.analyze_image_properties(img)

    assert props["resolution"] == (200, 200)
    assert props["dominant_color_hex"] == "#ff0000"
    assert props["corner_pixels_hex"]["top_left"] == "#ff0000"
    assert props["visual_warning_detected"] is True
