"""Model Manager for Groq, Gemini, and Ollama local LLM interaction with intelligent Hybrid Router."""

import time
import httpx
from enum import Enum
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from brain.brain_config import brain_config
from brain.logger import logger
from brain.utils import is_internet_available


class QueryComplexity(str, Enum):
    SIMPLE = "simple"       # Route to Local Ollama to save API tokens
    COMPLEX = "complex"     # Route to Cloud API (Groq/Gemini) for high reasoning
    REALTIME = "realtime"   # Route to Cloud API with web search context


class ModelManager:
    """Manages LLM connectivity across Ollama (Local), Groq, and Gemini Cloud APIs with Autonomous Hybrid Routing."""

    def __init__(self, api_key: Optional[str] = None, current_model: Optional[str] = None) -> None:
        self.groq_api_key = api_key or brain_config.GROQ_API_KEY
        self.gemini_api_key = brain_config.GEMINI_API_KEY
        self.ollama_base_url = brain_config.OLLAMA_BASE_URL.rstrip('/').replace("localhost", "127.0.0.1")
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

    @staticmethod
    def classify_query_complexity(
        user_query: str,
        is_realtime: bool = False,
        intent_code: Optional[str] = None,
    ) -> QueryComplexity:
        """Sub-millisecond heuristic query classification for hybrid routing (Local vs Cloud)."""
        if is_realtime or intent_code == "REALTIME_KNOWLEDGE_SEARCH":
            return QueryComplexity.REALTIME

        q_clean = user_query.strip().lower()
        query_len = len(q_clean)

        # Triggers indicating complex analytical reasoning or code architecture
        complex_keywords = [
            "architecture", "algorithm", "refactor", "debug", "explain in detail",
            "compare", "difference between", "mathematical proof", "design pattern",
            "database schema", "step-by-step", "tradeoffs", "optimize code"
        ]

        if any(kw in q_clean for kw in complex_keywords):
            return QueryComplexity.COMPLEX

        if intent_code in ("TASK_PLANNING", "CODE_GENERATION") or query_len > 250:
            return QueryComplexity.COMPLEX

        # Simple casual / direct queries -> route to Ollama to save tokens
        simple_intents = ("GENERAL_CONVERSATION", "OPEN_APPLICATION", "SYSTEM_TELEMETRY", "FILESYSTEM_OPERATION")
        if intent_code in simple_intents or query_len < 80:
            return QueryComplexity.SIMPLE

        return QueryComplexity.SIMPLE

    async def _get_installed_ollama_models(self) -> List[str]:
        """Fetch list of models currently pulled in local Ollama service."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.ollama_base_url}/api/tags")
                if res.status_code == 200:
                    models = res.json().get("models", [])
                    return [m.get("name") for m in models if m.get("name")]
        except Exception:
            pass
        return []

    async def _call_ollama(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """Execute request against local Ollama LLM endpoint with automatic model tag fallback."""
        endpoint = f"{self.ollama_base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        clean_model = model.replace("ollama/", "")

        # Format system messages into a consolidated system prompt for Ollama compatibility
        formatted_messages: List[Dict[str, str]] = []
        system_chunks: List[str] = []
        other_messages: List[Dict[str, str]] = []

        for m in messages:
            if m.get("role") == "system":
                system_chunks.append(m.get("content", ""))
            else:
                other_messages.append(m)

        if system_chunks:
            formatted_messages.append({"role": "system", "content": "\n\n".join(system_chunks)})
        formatted_messages.extend(other_messages)

        payload = {
            "model": clean_model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 8192,  # Expand Ollama context window so system memory and past chats are NOT truncated
            },
        }

        start_time = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                if response.status_code == 404 or "not found" in response.text.lower():
                    installed = await self._get_installed_ollama_models()
                    for alt_model in installed:
                        if alt_model != clean_model:
                            logger.info(f"Model '{clean_model}' not found in Ollama. Trying installed model '{alt_model}'...")
                            payload["model"] = alt_model
                            resp_alt = await client.post(endpoint, headers=headers, json=payload)
                            if resp_alt.status_code == 200:
                                data = resp_alt.json()
                                content = data.get("message", {}).get("content", "")
                                eval_count = data.get("eval_count", 0)
                                prompt_eval_count = data.get("prompt_eval_count", 0)
                                return {
                                    "content": content,
                                    "model": alt_model,
                                    "provider": "ollama",
                                    "latency_ms": round(elapsed_ms, 2),
                                    "finish_reason": "stop" if data.get("done") else None,
                                    "usage": {
                                        "prompt_tokens": prompt_eval_count,
                                        "completion_tokens": eval_count,
                                        "total_tokens": prompt_eval_count + eval_count,
                                    },
                                }
                    raise RuntimeError(f"Ollama model '{clean_model}' not found. Installed models: {installed}")

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
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise RuntimeError(f"Ollama local engine is not running at {self.ollama_base_url}. Please start Ollama ('ollama serve').") from exc

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
        intent_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute completion request with Autonomous Hybrid Router (Local Ollama vs Cloud APIs)."""
        target_model = model or self.current_model
        temp = temperature if temperature is not None else brain_config.TEMPERATURE
        tokens = max_tokens or brain_config.MAX_TOKENS

        # 1. Fast Internet Availability Check
        online = is_internet_available()

        # Extract last user query string for complexity classification
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break

        # Candidate Queue: List of (provider_name, model_name)
        candidates: List[Tuple[str, str]] = []

        if not online:
            logger.warning("No active internet connection detected. Switching directly to local Ollama LLM provider.")
            candidates.append(("ollama", self.ollama_model))
            installed = await self._get_installed_ollama_models()
            for inst_m in installed:
                if ("ollama", inst_m) not in candidates:
                    candidates.append(("ollama", inst_m))
        else:
            # 2. Hybrid Complexity Classification when online
            complexity = self.classify_query_complexity(user_query, is_realtime_query, intent_code)

            if brain_config.ENABLE_HYBRID_ROUTING and model is None:
                if complexity == QueryComplexity.SIMPLE:
                    logger.info(
                        f"Hybrid Router: Simple query detected ('{user_query[:40]}...'). "
                        f"Routing to local Ollama [{self.ollama_model}] to conserve Cloud API tokens."
                    )
                    candidates.append(("ollama", self.ollama_model))
                    installed = await self._get_installed_ollama_models()
                    for inst_m in installed:
                        if ("ollama", inst_m) not in candidates:
                            candidates.append(("ollama", inst_m))
                else:
                    logger.info(
                        f"Hybrid Router: {complexity.value.upper()} query detected ('{user_query[:40]}...'). "
                        f"Routing to high-capacity Cloud API [{target_model}]."
                    )

            # Add primary target model
            if "gemini" in target_model.lower():
                if ("gemini", target_model) not in candidates:
                    candidates.append(("gemini", target_model))
            elif "ollama" in target_model.lower():
                clean_m = target_model.replace("ollama/", "")
                if ("ollama", clean_m) not in candidates:
                    candidates.append(("ollama", clean_m))
            else:
                if ("groq", target_model) not in candidates:
                    candidates.append(("groq", target_model))

            # Fallback queue
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
            if not online and prov in ("groq", "gemini"):
                logger.info(f"Skipping cloud provider [{prov} : {mod}] because internet is disconnected.")
                continue

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
        fallback_msg = (
            "Internet is offline and local Ollama engine was unavailable. "
            "Please start Ollama ('ollama serve') with a pulled model (e.g. 'ollama pull qwen2.5')."
            if not online else
            "All configured LLMs and local engines were unavailable. Please check your network or start Ollama."
        )
        return {
            "content": fallback_msg,
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
