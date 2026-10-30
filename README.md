# JARVIS Brain OS

A fast, voice-first personal AI assistant that can actually operate your Windows PC — **safely**.

Talk to it ("Hey JARVIS, organize my downloads") or type in the dashboard. It answers questions with any LLM you
like (Gemini, Groq, OpenAI, Claude, OpenRouter, DeepSeek, or fully local with Ollama / LM Studio), and performs
real actions — apps, windows, volume, files, reminders, routines, browser tasks, messages — through a single
safety gateway that previews risky actions and asks before doing them.

```
"what's quantum tunnelling?"            → answers (no tools, no actions)
"set volume to 30"                      → done in ~5 ms (no LLM call)
"restart my pc"                         → "This needs your confirmation… Restart in 30 s" → "yes" → scheduled, "cancel shutdown" aborts
"How do I reboot my router?"            → explains — it will never reboot anything
```

---

## Quick start (≈10 minutes)

**You need:** Windows 10/11, Python 3.12+ ([python.org](https://www.python.org/downloads/)), and either one free API
key (e.g. [Gemini](https://aistudio.google.com/app/apikey) or [Groq](https://console.groq.com/keys)) **or**
[Ollama](https://ollama.com) for fully offline use (`ollama pull qwen3.5`).

```bash
git clone https://github.com/upeshchowdary/jarvis-brain-os.git
```

```bash
cd jarvis-brain-os
```

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -e ".[all]"
```

```bash
copy .env.example .env
```

Open `.env` and set at least one key (e.g. `GEMINI_API_KEY=…`), then:

```bash
python jarvis.py --open
```

The dashboard opens at <http://127.0.0.1:8000/dashboard>. Paste the API token printed in the console (it is also saved
in `.env` as `JARVIS_API_TOKEN`). Done.

Optional extras:

| Want | Do |
|---|---|
| Hands-free voice | `python jarvis.py --voice` (uses local **faster-whisper**; downloads the model on first run) |
| Natural neural voice | download `kokoro-v1.0.onnx` and `voices-v1.0.bin` from the [kokoro-onnx releases](https://github.com/thewh1teagle/kokoro-onnx/releases) into `data/models/` |
| "Hey JARVIS" acoustic wake word | `pip install -e ".[wakeword]"` and set `WAKE_WORD_ENGINE=openwakeword` |
| Tray icon + Win+J hotkey | `python jarvis.py --tray` |
| Browser agent | `playwright install chromium` |
| Terminal chat | `python chat.py` (streams tokens) |
| Use JARVIS from Claude Desktop / Cursor | see [MCP](#mcp) |

---

## What it can do (demo commands)

| Area | Try saying / typing |
|---|---|
| **Conversation** | "explain recursion with an example", "compare Rust and Go" |
| **Live info** | "what's the latest AI news", "research the best laptop for students" (multi-source, cited) |
| **System control** | "volume 40", "mute", "brightness up", "lock the screen", "snap window left", "take a screenshot", "battery level" |
| **Power** (confirmed) | "restart my pc" → "yes" · "cancel shutdown" · "put the laptop to sleep" |
| **Apps & web** | "open spotify", "close chrome", "open github.com", "play lofi beats on youtube", "google best pizza near me" |
| **Processes** (confirmed) | "kill notepad.exe" — exact name only, never JARVIS or system processes |
| **Reminders** | "remind me at 6 pm to submit the report", "every weekday at 9 open my class links", "set a timer for 10 minutes", "what are my reminders" |
| **Routines** | "start study mode", "run the shutdown routine", "list routines" (edit `data/routines/*.yaml`) |
| **Files** (undoable) | "organize my downloads folder", "find duplicates in downloads", "rename all photos in pictures to goa_{n}", "zip the report folder", "undo that" |
| **Documents** | "summarize ~/Documents/syllabus.pdf", "according to my notes folder, when is the physics exam?", "extract the tables from results.docx to excel" |
| **Productivity** | "add buy milk to my to-do list", "take a note wifi password is on the router", "paste what I copied 3 items ago", "give me my briefing", "what's the weather" |
| **Memory** | "remember that my exam is on Friday", "what do you know about me", "forget my exam" |
| **Messaging** (preview + confirm) | "tell Rahul on WhatsApp I'll be 10 minutes late", "email hr@company.com asking for the offer letter, formally" |
| **Screen** | "what's this error on my screen?", "read the text on screen" |
| **Computer use** | "open chrome, search for python jobs in Hyderabad and open the first result" (plan shown first) |
| **Browser agent** | "use the browser to find the cheapest flight from Hyderabad to Goa next Friday" (stops before paying) |
| **Developer** | "git status of jarvis", "summarize today's commits in jarvis", "run the tests in jarvis", "open my jarvis project" |
| **Workflows** | "start recording workflow morning setup" … "stop recording" · "replay morning setup" · triggers in YAML |
| **Safety** | "emergency stop", "resume automation", "enable dry run mode", "automation status" |
| **Models** | "switch to groq", "use ollama", "switch to claude" (per conversation) |

Proactive: low-battery / high-CPU / full-disk alerts, reminder toasts, scheduled routines.

---

## How it works

```
 voice (wake word → VAD → faster-whisper)   chat.py   dashboard   REST API   MCP clients
                       │                       │          │          │          │
                       └───────────────┬───────┴──────────┴──────────┴──────────┘
                                       ▼
                         brain/pipeline.py  (session → route → act → respond, streamed)
                   ┌───────────────────┼─────────────────────────────┐
                   ▼                   ▼                             ▼
     pending confirmation?   brain/router.py fast path      LLM agent (native tool calling)
       "yes" → run it        anchored grammar, ~1 ms         core/llm: 8 providers, streaming,
                             no LLM needed                    retries, fallbacks, local-only mode
                                       │                             │
                                       └──────────────┬──────────────┘
                                                      ▼
                                    tools/  (80+ tools: schema + risk level)
                                                      │ side effects only via
                                                      ▼
                       automation/gateway.py  ActionGateway
                       risk policy · preview · confirmation · dry-run · audit log · e-stop · undo
                                                      ▼
                       automation/ops/*  (private)  system · apps · files · messaging ·
                       computer-use engine (UIA → keyboard → mouse/OCR → vision) · browser agent · MCP
```

* **Questions never trigger actions.** Questions and statements are only offered read-only tools; 120 benign
  questions are tested against a model that *tries* to act.
* **Low latency.** Common commands bypass the LLM; replies stream token-by-token and voice starts speaking at the
  first sentence; only relevant tools are sent to the model; HTTP connections are reused.

More: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/SAFETY.md](docs/SAFETY.md) · [CHANGELOG.md](CHANGELOG.md)

---

## Safety model (short)

| Risk | Examples | Behaviour |
|---|---|---|
| LOW | open app/website, volume, brightness, windows, lock, screenshot, read-only lookups | runs immediately |
| MEDIUM | clicking/typing in apps, writing/renaming files, email drafts, browser agent | asks first (`CONFIRM_MEDIUM_RISK=false` to relax) |
| HIGH | shutdown/restart/sleep, kill process, delete (Recycle Bin), move files, send messages, shell, wifi off | **always** asks |

Everything is logged to `data/audit.jsonl` and visible in the dashboard. Dry-run (`DRY_RUN=true` or "enable dry run")
simulates everything. Web pages, files and screen text are treated as untrusted data and can't trigger actions.
`PRIVACY_MODE=local` keeps file and screen content on your machine (Ollama / LM Studio only). The API listens on
127.0.0.1 only and requires the token.

---

## Configuration

All settings live in `.env` (see [.env.example](.env.example)). The important ones:

| Setting | Default | Meaning |
|---|---|---|
| `DEFAULT_MODEL` | `gemini/gemini-3.5-flash-lite` | `provider/model` used for chat |
| `LLM_FALLBACKS` | groq → openai → ollama | tried in order if the default fails |
| `PRIVACY_MODE` | `cloud` | `local` = files/screens only go to local models |
| `CONFIRM_MEDIUM_RISK` | `true` | ask before medium-risk actions |
| `DRY_RUN` | `false` | simulate every action |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | API bind address |
| `STT_ENGINE` / `WHISPER_MODEL` | `auto` / `base.en` | speech-to-text engine and model size |
| `TTS_ENGINE` | `auto` | `kokoro` (if model files present) or `pyttsx3` |

---

## MCP

JARVIS speaks the [Model Context Protocol](https://modelcontextprotocol.io):

* **As a server** — add to Claude Desktop's `claude_desktop_config.json`:
  ```json
  {"mcpServers": {"jarvis": {"command": "C:/path/to/jarvis-brain-os/.venv/Scripts/python.exe",
                             "args": ["-m", "integrations.mcp_server"],
                             "cwd": "C:/path/to/jarvis-brain-os"}}}
  ```
  MCP clients can never approve their own risky actions — you approve them in JARVIS.
* **As a client** — list servers (GitHub, filesystem, Google Calendar/Gmail …) in `data/mcp_servers.json`; see
  [integrations/mcp_servers.example.json](integrations/mcp_servers.example.json). Their write tools go through the gateway.

---

## API

Every route except `/health` needs `Authorization: Bearer <JARVIS_API_TOKEN>`.

| Route | Purpose |
|---|---|
| `POST /chat`, `POST /chat/stream` (SSE) | talk to JARVIS |
| `POST /automation/execute` `{"command", "dry_run"}` | run a command (dry-run returns the plan only) |
| `GET /automation/pending`, `POST /automation/confirm/{id}`, `POST /automation/cancel/{id}` | approvals |
| `GET /automation/audit`, `POST /automation/undo`, `POST /automation/stop`, `/resume` | audit, undo, e-stop |
| `/reminders`, `/todos`, `/notes`, `/routines`, `/clipboard`, `/briefing` | productivity |
| `GET /events` (SSE), `GET /metrics`, `GET /status`, `GET /config` | live feed, latency p50/p95, status |
| `/voice/*` | microphone and speech control |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "I can't reach a language model" | add an API key to `.env`, or start Ollama (`ollama serve`) and `ollama pull qwen3.5` |
| Dashboard says token rejected | copy `JARVIS_API_TOKEN` from `.env` (no quotes) |
| Volume/brightness/windows do nothing | install the Windows extras: `pip install -e ".[windows]"` |
| Voice doesn't hear me | check `/voice/devices`, select the mic with `POST /voice/device/select`; lower `ENERGY_THRESHOLD` |
| Whisper is slow | use `WHISPER_MODEL=tiny.en` or a CUDA GPU; or `STT_ENGINE=google` |
| A command was misunderstood | rephrase as a command ("open …", "set …"); questions are always answered, never executed |
| Something went wrong | say "emergency stop", check the audit log in the dashboard, "undo that" for file operations |

---

## Development

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
ruff check .
```

```bash
pyright
```

Tests never touch the real desktop: GUI/OS modules are mocked and data goes to a temp folder. CI runs on Windows and
Linux (`.github/workflows/ci.yml`).
