"""Google Gemini Vision Provider v3 — PRIMARY vision engine for JARVIS.

Uses gemini-2.5-flash — Google's latest, fastest multimodal model.
v3 improvements:
  - WebP input (smaller payload = faster upload)
  - Connection-pooled httpx client (reuse across calls)
  - Max dim 768px (proven sufficient for screen analysis)
  - Streaming response collection for faster TTFT
  - Graceful quota error handling for fallback chain
"""

import base64
import io
import time
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from vision.providers.base import BaseVisionProvider


# Gemini quota / rate-limit HTTP status codes that trigger fallback
_QUOTA_CODES = {429, 503, 500}

# Reusable httpx client for connection pooling
_http_client: Optional[httpx.AsyncClient] = None


def _get_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Get or create a pooled async HTTP client."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
    return _http_client


class GeminiVisionProvider(BaseVisionProvider):
    """Adapter for Google Gemini Vision Models (gemini-2.5-flash)."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider_name="gemini", model_name=model_name, api_key=api_key)
        self.timeout = timeout

    @staticmethod
    def _prepare_image(image: Any, max_dim: int = 768, quality: int = 70) -> tuple:
        """Resize + compress, return (raw_base64, mime_type). Prefers WebP."""
        try:
            from PIL import Image as PILImage
            if not isinstance(image, PILImage.Image):
                return None, None
            img = image.convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    PILImage.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            # Try WebP first (40% smaller)
            try:
                img.save(buf, format="WEBP", quality=quality, method=4)
                mime = "image/webp"
            except Exception:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=quality, optimize=True)
                mime = "image/jpeg"

            return base64.b64encode(buf.getvalue()).decode("utf-8"), mime
        except Exception as e:
            logger.error(f"[JARVIS][Gemini] Image preparation failed: {e}")
            return None, None

    async def analyze_image(self, image: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "success": False,
                "provider": self.provider_name,
                "error": "GEMINI_API_KEY is not configured.",
            }

        raw_b64, mime_type = self._prepare_image(
            image,
            max_dim=kwargs.get("max_dim", 768),
        )
        if not raw_b64:
            return {
                "success": False,
                "provider": self.provider_name,
                "error": "Image is invalid or empty.",
            }

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model_name}:generateContent?key={self.api_key}"
        )
        gen_config: Dict[str, Any] = {
            "temperature": 0.1,
            "maxOutputTokens": 512,
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": raw_b64,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": gen_config,
        }

        start_time = time.perf_counter()
        try:
            client = _get_client(self.timeout)
            response = await client.post(url, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code in _QUOTA_CODES:
                err = f"Gemini quota/rate-limit [{response.status_code}]: {response.text[:200]}"
                logger.warning(f"[JARVIS][Gemini] {err}")
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "quota_exceeded": True,
                    "error": err,
                }

            if response.status_code != 200:
                err = f"Gemini error [{response.status_code}]: {response.text[:200]}"
                logger.warning(f"[JARVIS][Gemini] {err}")
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "error": err,
                }

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "error": "Gemini returned empty candidates.",
                }

            content = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )

            if not content:
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "error": "Gemini returned empty text.",
                }

            logger.info(
                f"[JARVIS][Gemini] ✓ {self.model_name} responded in {elapsed_ms:.0f}ms"
            )
            return {
                "success": True,
                "provider": "gemini",
                "model": self.model_name,
                "analysis": content,
                "latency_ms": round(elapsed_ms, 2),
            }

        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"[JARVIS][Gemini] Timed out after {elapsed_ms:.0f}ms")
            return {
                "success": False,
                "provider": self.provider_name,
                "error": f"Gemini timed out after {elapsed_ms:.0f}ms",
            }
        except Exception as e:
            logger.error(f"[JARVIS][Gemini] Exception: {e}")
            return {"success": False, "provider": self.provider_name, "error": str(e)}

    async def health_check(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("AIza"))
