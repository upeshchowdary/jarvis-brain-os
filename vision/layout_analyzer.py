"""JARVIS Spatial Layout & Window Hierarchy Analyzer.

Parses spatial distribution of desktop windows, window overlaps, screen grid alignment
(maximized, split-screen left/right, multi-monitor placement), and UI element region grouping.
"""

from typing import List, Dict, Any, Tuple
from vision.environment import WindowInfo, MonitorInfo, UIElement, BoundingBox


class LayoutAnalyzer:
    """Analyzer for spatial screen layout, window placement hierarchy, and UI grouping."""

    def analyze_window_layout(
        self,
        windows: List[WindowInfo],
        monitors: List[MonitorInfo],
    ) -> Dict[str, Any]:
        """Analyzes spatial layout of open windows across monitors."""
        if not windows:
            return {
                "layout_style": "empty",
                "active_monitor": 0,
                "window_count": 0,
                "overlap_detected": False,
            }

        primary_mon = monitors[0] if monitors else MonitorInfo()
        screen_w, screen_h = primary_mon.resolution

        active_win = next((w for w in windows if w.is_active), windows[0])
        wb = active_win.bounds

        # Determine alignment pattern
        layout_style = "floating"
        if wb.width >= int(screen_w * 0.95) and wb.height >= int(screen_h * 0.95):
            layout_style = "maximized"
        elif wb.width <= int(screen_w * 0.55) and wb.x < 50:
            layout_style = "split_left"
        elif wb.width <= int(screen_w * 0.55) and wb.x >= int(screen_w * 0.45):
            layout_style = "split_right"

        overlap_detected = len(windows) > 1

        return {
            "layout_style": layout_style,
            "active_window_title": active_win.title,
            "active_application": active_win.app_name,
            "active_monitor": active_win.monitor_index,
            "window_count": len(windows),
            "overlap_detected": overlap_detected,
            "active_bounds": {"x": wb.x, "y": wb.y, "width": wb.width, "height": wb.height},
        }

    def group_ui_elements_by_region(
        self,
        ui_elements: List[UIElement],
        screen_resolution: Tuple[int, int] = (1920, 1080),
    ) -> Dict[str, List[UIElement]]:
        """Groups UI interaction elements into spatial semantic regions (Header, Main, Sidebar, Footer, Taskbar)."""
        sw, sh = screen_resolution
        grouped: Dict[str, List[UIElement]] = {
            "header_navbar": [],
            "sidebar": [],
            "main_content": [],
            "footer_status": [],
            "taskbar": [],
        }

        for el in ui_elements:
            cy = el.bounds.y + (el.bounds.height // 2)
            cx = el.bounds.x + (el.bounds.width // 2)

            if cy >= int(sh * 0.93):
                grouped["taskbar"].append(el)
            elif cy <= int(sh * 0.12):
                grouped["header_navbar"].append(el)
            elif cy >= int(sh * 0.85):
                grouped["footer_status"].append(el)
            elif cx <= int(sw * 0.20):
                grouped["sidebar"].append(el)
            else:
                grouped["main_content"].append(el)

        return grouped


# Global singleton instance
layout_analyzer = LayoutAnalyzer()
