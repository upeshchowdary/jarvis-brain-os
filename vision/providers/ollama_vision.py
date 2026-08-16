"""Ollama Local Vision Provider v3.

Supports local multimodal models served via Ollama:
  - qwen3-vl:8b, gemma3, llava, moondream, llama3.2-vision, pixtral

v3 improvements:
  - Updated preferred model list (Gemma 3, Pixtral added)
  - Reduced timeout to 60s (from 180s)
  - Streaming optimizations
  - Consolidates LLaVA + Qwen adapters (deleted separate files)
"""

import base64
import io
import time
from typing import Dict, Any, Optional, List

import httpx
from loguru import logger

from vision.providers.base import BaseVisionProvider
from brain.brain_config import brain_config


# Model names that indicate multimodal support
_VISION_MODEL_KEYWORDS = [
    "vl", "vision", "llava", "bakllava", "moondream",
    "minicpm-v", "qwen3", "pixtral", "gemma3",
]

# Preferred order when auto-discovering
_PREFERRED_VISION_ORDER = [
    "qwen3-vl:2b", "gemma3", "qwen2.5-vl", "pixtral",
    "llava:13b", "llava:7b", "llava",
    "moondream", "minicpm-v", "bakllava", "llama3.2-vision",
]


class OllamaVisionProvider(BaseVisionProvider):
    """Adapter for Ollama local vision models."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        target_base = (base_url or brain_config.OLLAMA_BASE_URL).rstrip("/")
        target_model = model_name or brain_config.OLLAMA_VISION_MODEL
        super().__init__(provider_name="ollama", model_name=target_model, api_key=None)
        self.base_url = target_base
        self.timeout = timeout or min(brain_config.VISION_TIMEOUT, 60.0)

    # ------------------------------------------------------------------
    # Model discovery
    # ------------------------------------------------------------------

    async def get_installed_models(self) -> List[str]:
        """Return list of all model names installed in local Ollama."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    return [m.get("name", "") for m in res.json().get("models", [])]
        except Exception as e:
            logger.debug(f"[JARVIS] OllamaVisionProvider.get_installed_models error: {e}")
        return []

    async def get_best_vision_model(self) -> Optional[str]:
        """Return the best available vision model installed locally."""
        installed = await self.get_installed_models()
        if not installed:
            return None

        installed_lower = {m.lower(): m for m in installed}

        # 1. Exact configured model
        cfg = brain_config.OLLAMA_VISION_MODEL.lower()
        for inst_low, inst_orig in installed_lower.items():
            if cfg in inst_low or inst_low in cfg:
                return inst_orig

        # 2. Preferred order
        for preferred in _PREFERRED_VISION_ORDER:
            p_lower = preferred.lower()
            for inst_low, inst_orig in installed_lower.items():
                if p_lower in inst_low or inst_low in p_lower:
                    return inst_orig

        # 3. Any vision-keyword model
        for inst_low, inst_orig in installed_lower.items():
            if any(kw in inst_low for kw in _VISION_MODEL_KEYWORDS):
                return inst_orig

        return None

    async def verify_model_installed(self, model_name: Optional[str] = None) -> bool:
        """Check if a specific model is installed."""
        target = (model_name or self.model_name).lower()
        installed = await self.get_installed_models()
        return any(target in m.lower() or m.lower() in target for m in installed)

    # ------------------------------------------------------------------
    # Image preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_image(image: Any, max_dim: int = 768) -> Optional[str]:
        """Resize + JPEG-compress, return raw base64 string."""
        try:
            from PIL import Image
            if not isinstance(image, Image.Image):
                return None
            img = image.convert("RGB")
            w, h = img.size
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))
                img = img.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))),
                    Image.Resampling.LANCZOS,
                )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=78, optimize=True)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error(f"[JARVIS] Image preparation failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    async def analyze_image(self, image: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Send image + prompt to Ollama vision model.
        Uses streaming to avoid timeout on large responses.
        """
        max_dim = kwargs.get("max_dim", brain_config.VISION_IMAGE_SIZE)
        raw_b64 = kwargs.get("raw_b64") or self._prepare_image(image, max_dim=max_dim)

        if not raw_b64:
            return {
                "success": False,
                "provider": self.provider_name,
                "error": "Invalid or empty image.",
            }

        # Auto-discover best vision model if not explicitly specified
        if "model" in kwargs and kwargs["model"]:
            target_model = kwargs["model"]
        else:
            best_model = await self.get_best_vision_model()
            if best_model and best_model != self.model_name:
                logger.info(f"[JARVIS] Auto-selected: {best_model} (configured: {self.model_name})")
                target_model = best_model
            elif best_model:
                target_model = best_model
            else:
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "error": "No Ollama vision model found. Run: ollama pull qwen3-vl:2b",
                }

        logger.info(f"[JARVIS] Ollama vision: {target_model} (streaming)")

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": target_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [raw_b64],
                }
            ],
            "stream": True,
            "options": {
                "temperature": 0.1,
                "num_predict": 1024,
                "think": False,
            },
        }

        start_time = time.perf_counter()
        full_text = []
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        err_body = await response.aread()
                        err_msg = f"Ollama [{response.status_code}]: {err_body.decode()[:300]}"
                        return {
                            "success": False,
                            "provider": self.provider_name,
                            "model": target_model,
                            "error": err_msg,
                        }

                    import json as _json
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            chunk = _json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                full_text.append(token)
                            if chunk.get("done"):
                                break
                        except Exception:
                            continue

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            content = "".join(full_text).strip()

            logger.info(f"[JARVIS] Ollama response: {elapsed_ms:.0f}ms ({len(content)} chars)")

            if not content:
                return {
                    "success": False,
                    "provider": self.provider_name,
                    "model": target_model,
                    "error": "Vision model returned empty response.",
                }

            return {
                "success": True,
                "provider": self.provider_name,
                "model": target_model,
                "analysis": content,
                "latency_ms": round(elapsed_ms, 2),
            }

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            err_msg = f"Ollama at {self.base_url} unreachable: {e}"
            logger.warning(f"[JARVIS] {err_msg}")
            return {"success": False, "provider": self.provider_name, "error": err_msg}

        except httpx.ReadTimeout:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            partial = "".join(full_text).strip()
            if partial:
                logger.warning(f"[JARVIS] Ollama timed out at {elapsed_ms:.0f}ms, partial response")
                return {
                    "success": True,
                    "provider": self.provider_name,
                    "model": target_model,
                    "analysis": partial + " [truncated due to timeout]",
                    "latency_ms": round(elapsed_ms, 2),
                    "partial": True,
                }
            return {
                "success": False,
                "provider": self.provider_name,
                "model": target_model,
                "error": f"Vision model timed out after {elapsed_ms:.0f}ms.",
            }

        except Exception as e:
            logger.error(f"[JARVIS] OllamaVisionProvider error: {e}")
            return {"success": False, "provider": self.provider_name, "error": str(e)}

    async def health_check(self) -> bool:
        model = await self.get_best_vision_model()
        return model is not None
