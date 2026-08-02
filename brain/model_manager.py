"""Model Manager for Groq, Gemini, and Ollama local LLM interaction with intelligent local vs. cloud fallback."""

import time
import httpx
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from brain.brain_config import brain_config
from brain.logger import logger


class ModelManager:
    """Manages LLM connectivity across Ollama (Local), Groq, and Gemini Cloud APIs."""

    def __init__(self, api_key: Optional[str] = None, current_model: Optional[str] = None) -> None:
        self.groq_api_key = api_key or brain_config.GROQ_API_KEY
        self.gemini_api_key = brain_config.GEMINI_API_KEY
        self.ollama_base_url = brain_config.OLLAMA_BASE_URL.rstrip('/')
        self.ollama_model = brain_config.OLLAMA_MODEL
        self.current_model = current_model or brain_config.DEFAULT_MODEL
        self.groq_endpoint = f"{brain_config.GROQ_BASE_URL.rstrip('/')}/chat/completions"
        self.gemini_endpoint = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

        self.available_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            f"ollama/{self.ollama_model}",
        ]

    def switch_model(self, new_model: str) -> bool:
        """Switch active LLM model dynamically."""
        if not new_model:
            return False
        logger.info(f"Switching ModelManager active model from '{self.current_model}' to '{new_model}'")
        self.current_model = new_model
        return True

    def list_models(self) -> List[str]:
        """Return list of supported models."""
        return self.available_models.copy()

    async def health_check(self) -> bool:
        """Verify API keys and endpoint availability."""
        try:
            test_messages = [{"role": "user", "content": "ping"}]
            res = await self.generate(messages=test_messages, max_tokens=5)
            return bool(res.get("content"))
        except Exception as exc:
            logger.error(f"ModelManager health check failed: {exc}")
            return False

    async def _call_ollama(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Execute request against local Ollama LLM endpoint."""
        endpoint = f"{self.ollama_base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        # Clean model name if passed with prefix
        clean_model = model.replace("ollama/", "")
        payload = {
            "model": clean_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code != 200:
                raise RuntimeError(f"Ollama local API error [{response.status_code}]: {response.text}")

            data = response.json()
            content = data.get("message", {}).get("content", "")
            eval_count = data.get("eval_count", 0)
            prompt_eval_count = data.get("prompt_eval_count", 0)

            return {
                "content": content,
                "model": clean_model,
                "provider": "ollama",
                "latency_ms": round(elapsed_ms, 2),
                "finish_reason": "stop" if data.get("done") else None,
                "usage": {
                    "prompt_tokens": prompt_eval_count,
                    "completion_tokens": eval_count,
                    "total_tokens": prompt_eval_count + eval_count,
                },
            }

    async def _call_groq(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Execute request against Groq Cloud API endpoint."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=brain_config.TIMEOUT_SECONDS) as client:
            response = await client.post(self.groq_endpoint, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 429:
                raise RuntimeError(f"Groq API rate limit exceeded (429): {response.text}")
            if response.status_code != 200:
                raise RuntimeError(f"Groq API call error [{response.status_code}]: {response.text}")

            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            usage = data.get("usage", {})

            return {
                "content": content,
                "model": model,
                "provider": "groq",
                "latency_ms": round(elapsed_ms, 2),
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

    async def _call_gemini(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Execute request against Google Gemini OpenAI-compatible API endpoint."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.gemini_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=brain_config.TIMEOUT_SECONDS) as client:
            response = await client.post(self.gemini_endpoint, headers=headers, json=payload)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code == 429:
                raise RuntimeError(f"Gemini API rate limit exceeded (429): {response.text}")
            if response.status_code != 200:
                raise RuntimeError(f"Gemini API call error [{response.status_code}]: {response.text}")

            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            usage = data.get("usage", {})

            return {
                "content": content,
                "model": model,
                "provider": "gemini",
                "latency_ms": round(elapsed_ms, 2),
                "finish_reason": choice.get("finish_reason"),
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                },
            }

    async def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        is_realtime_query: bool = False,
    ) -> Dict[str, Any]:
        """Execute completion request with intelligent local (Ollama) vs cloud (Groq/Gemini) fallback."""
        target_model = model or self.current_model
        temp = temperature if temperature is not None else brain_config.TEMPERATURE
        tokens = max_tokens or brain_config.MAX_TOKENS

        # Candidate Queue: List of (provider_name, model_name)
        candidates: List[Tuple[str, str]] = []

        # If query does NOT need internet/real-time data, prefer local Ollama model first
        if not is_realtime_query:
            candidates.append(("ollama", self.ollama_model))

        # Add requested primary model
        if "gemini" in target_model.lower():
            candidates.append(("gemini", target_model))
        elif "ollama" in target_model.lower():
            candidates.append(("ollama", target_model.replace("ollama/", "")))
        else:
            candidates.append(("groq", target_model))

        # Fallback cloud model queue
        fallback_queue = [
            ("groq", "llama-3.1-8b-instant"),
            ("groq", "mixtral-8x7b-32768"),
            ("gemini", "gemini-1.5-flash"),
            ("gemini", "gemini-1.5-pro"),
            ("groq", "gemma2-9b-it"),
            ("ollama", self.ollama_model),
        ]

        for prov, mod in fallback_queue:
            if (prov, mod) not in candidates:
                candidates.append((prov, mod))

        last_error = None
        for prov, mod in candidates:
            try:
                if prov == "ollama":
                    res = await self._call_ollama(messages, mod, temp, tokens)
                    logger.info(f"ModelManager generated response using local [{prov} : {mod}]")
                    return res
                elif prov == "groq" and self.groq_api_key:
                    res = await self._call_groq(messages, mod, temp, tokens)
                    logger.info(f"ModelManager generated response using cloud [{prov} : {mod}]")
                    return res
                elif prov == "gemini" and self.gemini_api_key:
                    res = await self._call_gemini(messages, mod, temp, tokens)
                    logger.info(f"ModelManager generated response using cloud [{prov} : {mod}]")
                    return res
            except Exception as exc:
                logger.warning(
                    f"Model candidate [{prov} : {mod}] unavailable or hit limit ({exc}). "
                    f"Switching automatically to next candidate..."
                )
                last_error = exc
                continue

        logger.error(f"All model candidates failed. Last error: {last_error}")
        return {
            "content": "All configured LLMs and local engines were unavailable. Please check your network or start Ollama.",
            "model": "all_models_exceeded",
            "provider": "fallback",
            "latency_ms": 0.0,
            "finish_reason": "quota_exhausted",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream completion tokens asynchronously from active provider API."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured.")

        target_model = model or self.current_model
        temp = temperature if temperature is not None else brain_config.TEMPERATURE
        tokens = max_tokens or brain_config.MAX_TOKENS

        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": True,
        }

        async with httpx.AsyncClient(timeout=brain_config.TIMEOUT_SECONDS) as client:
            async with client.stream("POST", self.groq_endpoint, headers=headers, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise RuntimeError(f"Groq API stream failed [{response.status_code}]: {error_body.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = httpx.json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except Exception:
                            continue


# Global model manager singleton
model_manager = ModelManager()
