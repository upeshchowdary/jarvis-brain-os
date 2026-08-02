# JARVIS AI Operating System - Phase 1: The Brain

A modular, production-quality AI Operating System built from scratch using Clean Architecture principles in Python 3.12+.

---

## 🏗️ Architectural Overview

JARVIS Phase 1 implements **The Brain**—the cognitive core responsible for query intent classification, prompt management, LLM abstraction, reasoning, multi-step planning, state telemetry, and REST API exposure.

```
┌────────────────────────────────────────────────────────┐
│                   API Layer (FastAPI)                  │
│       Routes: POST /chat, GET /health, /config, /status│
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                   Application Layer                    │
│    (Brain Orchestrator, Context Builder, Reasoning)    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│                     Domain Layer                       │
│    (Entities, Base LLM Interface, Base Tool Interface, │
│        Base Memory Protocol, Pydantic Schemas)         │
└───────────────────────────▲────────────────────────────┘
                            │
┌───────────────────────────┴────────────────────────────┐
│                 Infrastructure Layer                   │
│   (Concrete LLM Providers: OpenAI, Groq, Gemini,       │
│    Anthropic, OpenRouter, Ollama, DeepSeek, Loguru,    │
│            Async SQLite Telemetry Database)            │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Supported LLM Providers

JARVIS features a completely provider-agnostic abstraction layer (`BaseLLMProvider` & `LLMFactory`). You can switch between providers seamlessly by updating `LLM_PROVIDER` in `.env`:

- **OpenAI** (`LLM_PROVIDER=openai`)
- **Groq** (`LLM_PROVIDER=groq`)
- **Anthropic** (`LLM_PROVIDER=anthropic`)
- **Google Gemini** (`LLM_PROVIDER=gemini`)
- **OpenRouter** (`LLM_PROVIDER=openrouter`)
- **Ollama Local** (`LLM_PROVIDER=ollama`)
- **DeepSeek** (`LLM_PROVIDER=deepseek`)
- **LM Studio** (`LLM_PROVIDER=lmstudio`)

---

## ⚙️ Configuration & Environment

Copy `.env.example` to `.env` and provide your API keys:

```bash
cp .env.example .env
```

Example `.env` snippet:
```ini
LLM_PROVIDER=openai
MODEL_NAME=gpt-4o
OPENAI_API_KEY=your_openai_key_here
GROQ_API_KEY=your_groq_key_here
GEMINI_API_KEY=your_gemini_key_here
```

---

## 💻 Running the Server

Launch the FastAPI production server using `uvicorn`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be accessible at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 🧪 Running Tests

Execute the automated pytest suite:

```bash
python -m pytest -v
```

---

## 🔌 Future Module Extensibility

JARVIS is built to accommodate future modules without modifying core Brain logic:

- **Tools**: Simply inherit from `BaseTool` in `app/tools/` and register with `tool_registry`.
- **Memory**: Implement `BaseMemory` in `app/memory/` for Short-term, Long-term, and ChromaDB Vector Memory.
- **Voice / Vision / Automation**: Future modules plug into pre-defined protocol interfaces.
