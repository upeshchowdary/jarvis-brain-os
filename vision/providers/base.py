"""Base abstract protocol for Vision LLM Providers.

Supports: Gemini, Groq, OpenAI, Ollama.
v3: WebP compression support, connection pooling helpers.
"""

import base64
import io
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from loguru import logger

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class BaseVisionProvider(ABC):
    """Abstract Base Class for Vision Model Providers."""

    def __init__(self, provider_name: str, model_name: str, api_key: Optional[str] = None) -> None:
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_key = api_key

    @abstractmethod
    async def analyze_image(self, image: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """Analyzes image with prompt and returns structured dictionary result."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Verifies provider API key and endpoint availability."""
        pass

    @staticmethod
    def image_to_base64_jpeg(image: Any, max_dimension: int = 768, quality: int = 75) -> str:
        """Convert PIL Image to base64 JPEG data URI string."""
        if not HAS_PIL or image is None:
            return ""

        w, h = image.size
        if max(w, h) > max_dimension:
            scale = max_dimension / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    @staticmethod
    def image_to_base64_webp(image: Any, max_dimension: int = 768, quality: int = 70) -> str:
        """Convert PIL Image to base64 WebP data URI string. ~40% smaller than JPEG."""
        if not HAS_PIL or image is None:
            return ""

        w, h = image.size
        if max(w, h) > max_dimension:
            scale = max_dimension / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="WEBP", quality=quality, method=4)
        img_bytes = buffer.getvalue()
        base64_str = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/webp;base64,{base64_str}"

    @staticmethod
    def image_to_raw_base64(
        image: Any,
        max_dimension: int = 768,
        quality: int = 70,
        use_webp: bool = True,
    ) -> tuple:
        """
        Convert PIL Image to raw base64 string (no data-URI prefix).
        Returns (base64_string, mime_type).
        """
        if not HAS_PIL or image is None:
            return "", ""

        w, h = image.size
        if max(w, h) > max_dimension:
            scale = max_dimension / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = io.BytesIO()
        if use_webp:
            try:
                image.save(buffer, format="WEBP", quality=quality, method=4)
                mime = "image/webp"
            except Exception:
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
                mime = "image/jpeg"
        else:
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            mime = "image/jpeg"

        return base64.b64encode(buffer.getvalue()).decode("utf-8"), mime
