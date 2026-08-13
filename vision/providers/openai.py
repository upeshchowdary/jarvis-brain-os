"""OpenAI Vision Provider Adapter."""

import time
import httpx
from typing import Dict, Any, Optional
from vision.providers.base import BaseVisionProvider
from loguru import logger


class OpenAIVisionProvider(BaseVisionProvider):
    """Adapter for OpenAI Vision Models (gpt-4o, gpt-4o-mini)."""

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", timeout: float = 60.0, **kwargs: Any) -> None:
        super().__init__(provider_name="openai", model_name=model_name, api_key=api_key)
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        self.timeout = timeout

    async def analyze_image(self, image: Any, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        if not self.api_key:
            return {"success": False, "provider": self.provider_name, "error": "OPENAI_API_KEY is not configured."}

        base64_img = self.image_to_base64_jpeg(image)
        if not base64_img:
            return {"success": False, "provider": self.provider_name, "error": "Image is invalid or empty."}

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
                        {"type": "image_url", "image_url": {"url": base64_img}},
                    ],
                }
            ],
            "max_tokens": kwargs.get("max_tokens", 1024),
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code != 200:
                    return {"success": False, "provider": self.provider_name, "error": f"OpenAI error [{response.status_code}]: {response.text}"}

                data = response.json()
                content = data["choices"][0]["message"]["content"] or ""
                return {
                    "success": True,
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "analysis": content,
                    "latency_ms": round(elapsed_ms, 2),
                }
        except Exception as e:
            logger.error(f"OpenAIVisionProvider error: {e}")
            return {"success": False, "provider": self.provider_name, "error": str(e)}

    async def health_check(self) -> bool:
        return bool(self.api_key)
