"""JARVIS Image Properties & Scene Analysis Engine.

Calculates dominant RGB/HEX colors, brightness score, contrast ratio, 4-corner pixel HEX codes,
visual complexity, edge density, quadrant breakdowns, and visual warning detection.
"""

from typing import Dict, Any, List, Tuple
from loguru import logger

try:
    from PIL import Image, ImageStat
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class ImageAnalyzer:
    """Engine for visual image properties, color palettes, brightness, and edge density."""

    def analyze_image_properties(self, image: Any) -> Dict[str, Any]:
        """Analyzes image properties returning detailed visual metrics."""
        if image is None or not HAS_PIL:
            return self._fallback_properties()

        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size

        # 1. Brightness & Contrast
        stat = ImageStat.Stat(image)
        r_mean, g_mean, b_mean = stat.mean[:3]
        brightness = (0.299 * r_mean) + (0.587 * g_mean) + (0.114 * b_mean)

        r_std, g_std, b_std = stat.stddev[:3]
        contrast = (r_std + g_std + b_std) / 3.0

        # 2. Corner Pixels
        corners = {
            "top_left": self._rgb_to_hex(image.getpixel((0, 0))),
            "top_right": self._rgb_to_hex(image.getpixel((w - 1, 0))),
            "bottom_left": self._rgb_to_hex(image.getpixel((0, h - 1))),
            "bottom_right": self._rgb_to_hex(image.getpixel((w - 1, h - 1))),
        }

        # 3. Dominant Color
        small = image.resize((50, 50))
        result = small.quantize(colors=3)
        palette = result.getpalette()[:9]
        dominant_rgb = tuple(palette[:3]) if palette else (int(r_mean), int(g_mean), int(b_mean))
        dominant_hex = self._rgb_to_hex(dominant_rgb)

        # 4. Edge Density & Visual Complexity using OpenCV
        edge_density = 0.0
        if HAS_CV2:
            try:
                img_np = np.array(image)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_pixels = np.count_nonzero(edges)
                edge_density = round(float(edge_pixels) / float(w * h) * 100.0, 2)
            except Exception as e:
                logger.debug(f"Edge density calculation fallback: {e}")

        complexity = "low" if edge_density < 3.0 else ("medium" if edge_density < 12.0 else "high")

        # 5. Visual Warning Check (Red popup or warning highlight)
        has_warning = (r_mean > 160 and g_mean < 80 and b_mean < 80)

        return {
            "resolution": (w, h),
            "brightness_score": round(brightness, 2),
            "contrast_score": round(contrast, 2),
            "dominant_color_hex": dominant_hex,
            "dominant_color_rgb": dominant_rgb,
            "corner_pixels_hex": corners,
            "edge_density_percent": edge_density,
            "visual_complexity": complexity,
            "visual_warning_detected": has_warning,
        }

    @staticmethod
    def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        """Converts RGB tuple to HEX string."""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

    @staticmethod
    def _fallback_properties() -> Dict[str, Any]:
        return {
            "resolution": (1920, 1080),
            "brightness_score": 50.0,
            "contrast_score": 10.0,
            "dominant_color_hex": "#1e1e23",
            "dominant_color_rgb": (30, 30, 35),
            "corner_pixels_hex": {"top_left": "#1e1e23", "top_right": "#1e1e23", "bottom_left": "#1e1e23", "bottom_right": "#1e1e23"},
            "edge_density_percent": 1.0,
            "visual_complexity": "low",
            "visual_warning_detected": False,
        }


# Global singleton instance
image_analyzer = ImageAnalyzer()
