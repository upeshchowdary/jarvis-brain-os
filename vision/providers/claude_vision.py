"""Anthropic Claude Vision Provider for JARVIS.

Uses claude-3-5-sonnet-20241022 for high-precision, low-latency screen analysis.
"""

import time
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from brain.brain_config import brain_config
from vision.providers.base import BaseVisionProvider

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


class ClaudeVisionProvider(BaseVisionProvider):
    """Adapter for Anthropic Claude Vision Models (claude-3-5-sonnet-20241022)."""

    ENDPOINT = "https://api.anthropic.com/v1/messages"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        key = api_key or brain_config.ANTHROPIC_API_KEY or ""
        model = model_name or brain_config.CLAUDE_VISION_MODEL
        super().__init__(provider_name="claude", model_name=model, api_key=key)
        self.timeout = timeout

    async def analyze_image(self, image: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "success": False,
                "provider": self.provider_name,
                "error": "ANTHROPIC_API_KEY is not configured.",
            }

        raw_b64, mime_type = self.image_to_raw_base64(
            image,
            max_dimension=kwargs.get("max_dim", 768),
            use_webp=False,  # Anthropic prefers jpeg or png
        )
        if not raw_b64:
            return {
                "success": False,
                "provider": self.provider_name,
                "error": "Image is invalid or empty.",
            }

        # Ensure supported mime_type for Anthropic (image/jpeg, image/png, image/gif, image/webp)
        if mime_type not in ("image/jpeg", "image/png", "image/gif", "image/webp"):
            mime_type = "image/jpeg"

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": raw_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        }

        start_time = time.perf_counter()
        try:
            client = _get_client(self.timeout)
            response = await client.post(self.ENDPOINT, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code in (429, 529):
                err = f"Claude rate limit/overloaded [{response.status_code}]: {response.text[:200]}"
                logger.warning(f"[JARVIS][Claude Vision] {err}")
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "quota_exceeded": True,
                    "error": err,
                }

            if response.status_code != 200:
                err = f"Claude error [{response.status_code}]: {response.text[:200]}"
                logger.warning(f"[JARVIS][Claude Vision] {err}")
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "error": err,
                }

            data = response.json()
            content_blocks = data.get("content", [])
            content = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    content += block.get("text", "")
            content = content.strip()

            if not content:
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "error": "Claude Vision returned empty content.",
                }

            logger.info(
                f"[JARVIS][Claude Vision] ✓ {self.model_name} responded in {elapsed_ms:.0f}ms"
            )
            return {
                "success": True,
                "provider": "claude",
                "model": self.model_name,
                "analysis": content,
                "latency_ms": round(elapsed_ms, 2),
            }

        except httpx.TimeoutException:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.warning(f"[JARVIS][Claude Vision] Timed out after {elapsed_ms:.0f}ms")
            return {
                "success": False,
                "provider": self.provider_name,
                "error": f"Claude Vision timed out after {elapsed_ms:.0f}ms",
            }
        except Exception as e:
            logger.error(f"[JARVIS][Claude Vision] Exception: {e}")
            return {"success": False, "provider": self.provider_name, "error": str(e)}

    async def health_check(self) -> bool:
        return bool(self.api_key and self.api_key.startswith("sk-ant-"))
