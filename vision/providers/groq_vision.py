"""Groq Vision Provider v3 — SECONDARY fast vision engine for JARVIS.

Uses meta-llama/llama-4-scout-17b-16e-instruct on Groq's LPU hardware.
v3 improvements:
  - Updated to Llama 4 Scout (latest multimodal model)
  - Connection-pooled httpx client
  - WebP input support (smaller payloads)
  - Request ID tracking for debugging
  - Reduced max_dim to 768px
"""

import base64
import io
import time
import uuid
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from vision.providers.base import BaseVisionProvider
from brain.brain_config import brain_config


# Reusable httpx client
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


class GroqVisionProvider(BaseVisionProvider):
    """Adapter for Groq Llama 4 Scout Vision (meta-llama/llama-4-scout-17b-16e-instruct)."""

    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        key = api_key or brain_config.GROQ_API_KEY or ""
        model = model_name or brain_config.GROQ_VISION_MODEL
        super().__init__(provider_name="groq", model_name=model, api_key=key)
        self.timeout = timeout

    @staticmethod
    def _prepare_image(image: Any, max_dim: int = 768, quality: int = 75) -> Optional[str]:
        """Resize + compress, return base64 data-URI string."""
        try:
            from PIL import Image as PILImage
            if not isinstance(image, PILImage.Image):
                return None
            img = image.convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    PILImage.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            # Groq OpenAI-compatible API accepts JPEG reliably
            img.save(buf, format="JPEG", quality=quality, optimize=True)
            raw = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{raw}"
        except Exception as e:
            logger.error(f"[JARVIS][Groq Vision] Image preparation failed: {e}")
            return None

    async def analyze_image(self, image: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "success": False,
                "provider": self.provider_name,
                "error": "GROQ_API_KEY is not configured.",
            }

        data_uri = self._prepare_image(image, max_dim=kwargs.get("max_dim", 768))
        if not data_uri:
            return {
                "success": False,
                "provider": self.provider_name,
                "error": "Image is invalid or empty.",
            }

        request_id = uuid.uuid4().hex[:8]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_uri, "detail": "high"},
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        start_time = time.perf_counter()
        try:
            client = _get_client(self.timeout)
            response = await client.post(self.ENDPOINT, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 429:
                err = f"Groq quota/rate-limit (429): {response.text[:200]}"
                logger.warning(f"[JARVIS][Groq Vision] [{request_id}] {err}")
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "quota_exceeded": True,
                    "error": err,
                }

            if response.status_code != 200:
                err = f"Groq error [{response.status_code}]: {response.text[:200]}"
                logger.warning(f"[JARVIS][Groq Vision] [{request_id}] {err}")
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "error": err,
                }

            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )

            if not content:
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "error": "Groq Vision returned empty response.",
                }

            logger.info(
                f"[JARVIS][Groq Vision] ✓ [{request_id}] {self.model_name} in {elapsed_ms:.0f}ms"
            )
            return {
                "success": True,
                "provider": "groq",
                "model": self.model_name,
                "analysis": content,
                "latency_ms": round(elapsed_ms, 2),
            }

        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"[JARVIS][Groq Vision] [{request_id}] Timed out after {elapsed_ms:.0f}ms")
            return {
                "success": False,
                "provider": self.provider_name,
                "error": f"Groq Vision timed out after {elapsed_ms:.0f}ms",
            }
        except Exception as e:
            logger.error(f"[JARVIS][Groq Vision] [{request_id}] Exception: {e}")
            return {"success": False, "provider": self.provider_name, "error": str(e)}

    async def health_check(self) -> bool:
        return bool(self.api_key)
