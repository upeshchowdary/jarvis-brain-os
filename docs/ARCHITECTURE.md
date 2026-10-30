# Architecture

```
app/            FastAPI app: routes (chat, stream, automation, productivity, voice, events, metrics), auth,
                web (site at /, console at /console; static files in app/static/web, strict CSP)
brain/          pipeline.py (turn handling), router.py (fast path + utterance shape), prompts.py, conversation & persona
core/           config (single source), paths (data/), logging (request IDs), llm/ (provider gateway),
                session (per-session LRU/TTL state), events (bus), metrics, connectivity
tools/          Tool base + registry + selection; builtin/ = one module per capability (80+ tools)
automation/     gateway.py (ActionGateway) · ops/ (private OS operations) · engine/ (computer-use planner,
                UIA-first action hierarchy, verification, recovery) · browser_agent.py · controllers · workflows
services/       scheduler, time parsing, routines, documents (BM25), health monitor, clipboard, triggers, notify, store
memory/         long-term facts (SQLite), short-term buffers, vector/keyword recall
vision/         screen capture, OCR, UI detection, multimodal providers (Gemini, Groq, Claude, OpenAI, Ollama)
voice/          capture, VAD, wake word (text or openWakeWord), STT (faster-whisper → Google), TTS (Kokoro → SAPI), barge-in
integrations/   MCP server (expose JARVIS) and MCP client (consume other servers)
routines/       built-in example routines (YAML)
```

## A turn, step by step (`brain/pipeline.py`)

1. **Session** — `core.session` (model override, vision cache; LRU + TTL) and conversation history (SQLite).
2. **Pending confirmation** — if this session has an action waiting: "yes" → `gateway.confirm`; "no" → cancel;
   anything else cancels it and continues as a new request. (MCP sessions can't confirm.)
3. **Fast path** — `router.fast_route` matches the whole normalized utterance against ~70 anchored rules
   ("volume 40", "remind me at 6 pm to …", "start study mode", "kill notepad.exe"). A match runs the tool directly:
   no LLM, a few milliseconds. Tools that need synthesis (e.g. today's commits) hand their result to the agent.
4. **Agent** — the LLM gets a system prompt (persona, rules, time, relevant facts/memories, voice style) and a
   *selected* subset of tool schemas (`tools/selection.py`): read-only tools for questions/statements, all tools
   for imperative requests. Up to `AGENT_MAX_STEPS` tool rounds:
   * tool results with untrusted content are wrapped in `<untrusted_data>` and taint the turn;
   * a tool returning `needs_confirmation` ends the turn with the preview;
   * a round consisting only of self-describing actions answers without a second LLM call.
5. **Respond** — streamed text (SSE / voice sentences), metrics, telemetry, and background memory extraction.

## LLM gateway (`core/llm/gateway.py`)

One OpenAI-compatible client for Gemini, Groq, OpenAI, Anthropic, OpenRouter, DeepSeek, Ollama and LM Studio.
Model strings are `provider/model`. Features: native tool calling, streaming with tool-call deltas, retries with
exponential backoff on 429/5xx/network errors, fallback chain (`DEFAULT_MODEL` → `LLM_FALLBACKS`), per-call timeouts,
a shared keep-alive `httpx.AsyncClient` per event loop, local-first ordering when offline, `local_only` for privacy.

## Computer use (`automation/engine`, `automation/ops/computer.py`)

Goals become a list of steps (`navigate`, `open_app`, `click`, `type`, `fill`, `press`, `hotkey`, `scroll`, `wait`,
`focus_window`) — fast deterministic plans for common web tasks (direct search URLs), the LLM planner otherwise
(shell steps are dropped). The plan runs as one `computer.run_plan` gateway operation; each step goes through the
action hierarchy (DOM → Windows UI Automation → keyboard → mouse/OCR → vision) with verification and recovery,
following the UIA-first, vision-last design of Microsoft UFO.

The **browser agent** (`automation/browser_agent.py`) is a Playwright observe→decide→act loop over numbered
interactive elements (set-of-marks over the DOM) with a persistent profile, password and risky-click guards.

## Voice (`voice/`)

mic → (optional openWakeWord) → energy+ZCR VAD (numpy) → faster-whisper (local) or Google → normalizer →
pipeline (`channel="voice"`: short spoken style) → sentence splitter → TTS queue (Kokoro or SAPI), interruptible
by barge-in ("stop" also triggers the emergency stop). Confirmations are read aloud; "yes" in the follow-up window
needs no wake word.

## Background services (`services/`)

Started with the API when `BACKGROUND_SERVICES=true`: reminder scheduler (1 s tick, SQLite), health monitor (30 s),
clipboard history (1 s), workflow triggers (2 s), MCP client connections. Notifications go to Windows toasts, the
voice (if running) and the console's live activity stream (`/events`).

## Data (`data/`, git-ignored)

`jarvis.db` (telemetry, conversation buffers), `jarvis_memory.db` (facts), `jarvis_productivity.db` (reminders,
notes, to-dos, clipboard), `audit.jsonl`, `workflows/*.yaml`, `routines/*.yaml`, `doc_index/`, `browser_profile/`,
`app_index.json`, `file_backups/`, `models/` (optional Kokoro files), `mcp_servers.json`, `risk_policy.json`.

## Libraries chosen (licences checked)

| Need | Choice | Licence | Notes |
|---|---|---|---|
| LLM access | own OpenAI-compatible client (httpx) | — | LiteLLM rejected: PyPI package compromised twice in 2026 |
| Local STT | faster-whisper | MIT | int8 CPU / fp16 GPU |
| Neural TTS | kokoro-onnx + Kokoro-82M | MIT / Apache-2.0 | Piper's maintained fork is GPL — not used |
| Wake word | openWakeWord (optional) | Apache-2.0 code | bundled models CC BY-NC-SA (personal use) |
| MCP | official `mcp` SDK 2.x | MIT | |
| Browser | Playwright | Apache-2.0 | |
| Recycle Bin | send2trash | BSD | |
| Documents | pypdf, python-docx, python-pptx, openpyxl, pdfplumber | BSD / MIT | PyMuPDF (AGPL) only if you install it yourself |
| Tray / hotkey | pystray, pynput (optional) | LGPL-3.0 | used as unmodified libraries |
