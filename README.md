# JARVIS Brain OS

A fast, voice-first personal AI assistant that can actually operate your Windows PC — **safely**.

Talk to it ("Hey JARVIS, organize my downloads"), or type or speak in the web console. It answers questions with any LLM you
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
key — [Gemini](https://aistudio.google.com/app/apikey) (Gemini Flash answers chat and actions) and
[Groq](https://console.groq.com/keys) (gpt-oss-120b understands each message first) are the defaults — **or**
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

The console opens at <http://127.0.0.1:8000/console>. Paste the API token printed in the terminal (it is also saved
in `.env` as `JARVIS_API_TOKEN`). Done. The project's own page is at <http://127.0.0.1:8000/>.

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
| Semantic memory search | part of `[all]` (the `[memory]` extra: chromadb + fastembed). The first start downloads the embedding model (`EMBEDDING_MODEL`, a few hundred MB) into `data/models/` and then indexes up to 5000 past turns in the background; without it, memory search is keyword-only |
| Traces of every turn in Langfuse | `pip install -e ".[tracing]"`, set `LANGFUSE_ENABLED=true` and the keys. Opt-in: your query, the reply and the rewritten request are sent to the Langfuse host |

---

## The console

`/console` is where you work with JARVIS from a browser. Phone widths are supported.

| View | What you do there |
|---|---|
| **Talk** | Type or press the mic and speak. Replies stream in. Each reply shows how it was handled (fast path or model, risk, time), and risky actions get Approve and Cancel buttons. Turns spoken to the PC microphone appear here too. |
| **Waiting for you** | Every action waiting for approval, from voice, schedules, routines or other apps. |
| **Today** | Reminders ("in 20 minutes", "every weekday at 8"), to-dos, notes, and your briefing. |
| **Routines** | Rehearse (dry run) or run routines and recorded workflows. |
| **Activity log** | Everything JARVIS did or refused, filtered by risk, plus "Undo last file change". |
| **System** | Latency per step, models and fallbacks, safety settings, and the PC microphone on or off. |

**Stop everything** in the top bar halts all automation until you resume. The **Dry run** switch rehearses
whatever you send from that console. Browser speech recognition works in Chrome and Edge. In other browsers the
mic button switches on the PC microphone instead.

---

## What it can do (demo commands)

| Area | Try saying / typing |
|---|---|
| **Conversation** | "explain recursion with an example", "compare Rust and Go" |
| **Live info** | "what's the latest AI news", "research the best laptop for students" (multi-source, cited) |
| **System control** | "volume 40", "mute", "brightness up", "lock the screen", "snap window left", "take a screenshot", "battery level" |
| **Keyboard & tabs** (instant) | "scroll down", "next tab", "go to tab 3", "close tab", "reopen closed tab", "go back", "refresh", "press enter", "type hello world", "switch to chrome". They act on the window in front; typed in the console, they act on the window you were in before |
| **Power** (confirmed) | "restart my pc" → "yes" · "cancel shutdown" · "put the laptop to sleep" |
| **Apps & web** | "open spotify", "close chrome", "open github.com", "play lofi beats on youtube", "google best pizza near me" |
| **Processes** (confirmed) | "kill notepad.exe" — exact name only, never JARVIS or system processes |
| **Reminders** | "remind me at 6 pm to submit the report", "every weekday at 9 open my class links", "set a timer for 10 minutes", "what are my reminders" |
| **Routines** | "start study mode", "run the shutdown routine", "list routines" (edit `data/routines/*.yaml`) |
| **Files** (undoable) | "organize my downloads folder", "find duplicates in downloads", "rename all photos in pictures to goa_{n}", "zip the report folder", "undo that" |
| **Documents** | "summarize ~/Documents/syllabus.pdf", "according to my notes folder, when is the physics exam?", "extract the tables from results.docx to excel" |
| **Productivity** | "add buy milk to my to-do list", "take a note wifi password is on the router", "paste what I copied 3 items ago", "give me my briefing", "what's the weather" |
| **Memory** | "remember that my exam is on Friday", "what do you know about me", "forget my exam" |
| **Chat history** | "what did I ask you today?", "what did we do about WhatsApp yesterday?" |
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
 voice (wake word → VAD → faster-whisper)   chat.py   console    REST API   MCP clients
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
| LOW | open app/website, volume, brightness, windows, lock, screenshot, read-only lookups, your own keyboard commands ("close tab", "type hello") | runs immediately |
| MEDIUM | clicking/typing in apps, key presses and typing the model decides on, writing/renaming files, email drafts, browser agent | asks first (`CONFIRM_MEDIUM_RISK=false` to relax) |
| HIGH | shutdown/restart/sleep, kill process, delete (Recycle Bin), move files, send messages, shell, wifi off | asks first (`CONFIRM_HIGH_RISK=false` turns this off, see below) |

**Full-control mode** (`CONFIRM_MEDIUM_RISK=false` and `CONFIRM_HIGH_RISK=false`): JARVIS shuts down, kills
processes, deletes, sends messages and runs shell commands **without asking**. A misheard voice command or a
model mistake then acts at once; only dry-run, the emergency stop, undo for file operations and the 30 s
shutdown delay remain. Use it only on your own PC. MCP clients (other programs) still need your approval for
every MEDIUM or HIGH action in this mode, and for any key press or typing; their LOW actions (open an app or a
website, set the volume) run at once.

Everything is logged to `data/audit.jsonl` and visible in the console's activity log. Dry-run (`DRY_RUN=true` or "enable dry run")
simulates everything. Web pages, files and screen text are treated as untrusted data and can't trigger actions.
`PRIVACY_MODE=local` keeps file and screen content on your machine (Ollama / LM Studio only). The API listens on
127.0.0.1 only and requires the token.

**What leaves your PC:** every message that isn't an instant command goes through the *intent step*, which sends
it with the last few turns (`INTENT_HISTORY_TURNS`, from voice, console and API) and the current task to
`INTENT_MODEL`'s provider (Groq by default; Google Gemini when Groq fails). `PRIVACY_MODE=local` does not
change that: set `INTENT_MODEL` to an Ollama model, or empty to switch the step off. Replies use `DEFAULT_MODEL` /
`ACTION_MODEL` (Gemini Flash by default).

---

## Configuration

All settings live in `.env` (see [.env.example](.env.example)). The important ones:

| Setting | Default | Meaning |
|---|---|---|
| `DEFAULT_MODEL` | `gemini/gemini-3.5-flash` | `provider/model` used for chat (flash-lite is not the default: unusably slow when measured on 2026-11-03) |
| `ACTION_MODEL` | `gemini/gemini-3.5-flash` | model for turns that act on the PC |
| `LLM_FALLBACKS` | `groq/openai/gpt-oss-120b` | tried in order if the default fails |
| `INTENT_MODEL` / `INTENT_FALLBACK_MODEL` | `groq/openai/gpt-oss-120b` / `gemini/gemini-3.5-flash` | rewrites each message into a complete request ("next do" → "send Nithesh 'hi' on WhatsApp"); empty = off |
| `INTENT_HISTORY_TURNS` | `4` | recent turns (any channel) the intent step sees |
| `GEMINI_API_KEYS` | empty | extra Gemini keys, comma separated: on a 429 / rejected key JARVIS moves to the next one |
| `GROQ_API_KEYS` | empty | extra Groq keys, comma separated, rotated the same way (limits are per Groq organisation) |
| `LLM_KEEPWARM` | `true` | keep provider connections warm (lists models, no tokens) for 15 min after a request |
| `PRIVACY_MODE` | `cloud` | `local` = files/screens only go to local models |
| `CONFIRM_MEDIUM_RISK` | `true` | ask before medium-risk actions |
| `CONFIRM_HIGH_RISK` | `true` | ask before high-risk actions; `false` = full control (see Safety) |
| `DRY_RUN` | `false` | simulate every action |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | local model for semantic memory search |
| `LANGFUSE_ENABLED` | `false` | send turn traces to Langfuse (opt-in) |
| `SILENCE_TIMEOUT_MS` | `600` | pause that ends a spoken command |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | API bind address |
| `STT_ENGINE` / `WHISPER_MODEL` | `auto` / `base.en` | speech-to-text engine and model size |
| `TTS_ENGINE` | `auto` | `kokoro` (if model files present) or `pyttsx3` |

**Free-tier quota:** the intent step runs on Groq (gpt-oss-120b, ~0.5–0.7 s) and chat and actions on Gemini Flash.
Groq's free tier allows 8,000 tokens per minute and 1,000 requests per day **per Groq organisation**, which is only
a few intent turns per minute. Put keys from *separate* organisations in `GROQ_API_KEYS`: JARVIS moves to the next
key when one answers 429, so each extra organisation adds capacity (keys from the same account share one limit).
Only when every Groq key is exhausted does JARVIS skip Groq until its reset time and use Gemini Flash for the
intent step. Gemini's free limits are **per Google Cloud project**, so only `GEMINI_API_KEYS` from different
projects add quota. `gemini-3.5-flash-lite` is no longer a default: when measured on 2026-11-03 it took 18 s for
a one-token reply.

---

## MCP

JARVIS speaks the [Model Context Protocol](https://modelcontextprotocol.io):

* **As a server** — add to Claude Desktop's `claude_desktop_config.json`:
  ```json
  {"mcpServers": {"jarvis": {"command": "C:/path/to/jarvis-brain-os/.venv/Scripts/python.exe",
                             "args": ["-m", "integrations.mcp_server"],
                             "cwd": "C:/path/to/jarvis-brain-os"}}}
  ```
  MCP clients can never approve their own MEDIUM/HIGH actions or any key press / typing — you approve them in
  JARVIS, even in full-control mode. Their LOW actions (open an app or a website, volume, system status) run
  directly. MCP turns don't see your recent conversation, current task, saved facts or recalled memories, can't
  save or erase facts, and aren't recalled into your own conversations later. They are still kept in the chat
  log (audit), and `chat_history` shows them only when asked for `channel=mcp`.
* **As a client** — list servers (GitHub, filesystem, Google Calendar/Gmail …) in `data/mcp_servers.json`; see
  [integrations/mcp_servers.example.json](integrations/mcp_servers.example.json). Their write tools go through the gateway.

---

## API

Every route except `/health` and the static web pages (`/`, `/console`, `/web/*`) needs
`Authorization: Bearer <JARVIS_API_TOKEN>`.

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
| Console says the token was rejected | copy `JARVIS_API_TOKEN` from `.env` (no quotes) |
| Volume/brightness/windows do nothing | install the Windows extras: `pip install -e ".[windows]"` |
| Voice doesn't hear me | check `/voice/devices`, select the mic with `POST /voice/device/select`; lower `ENERGY_THRESHOLD` |
| Whisper is slow | use `WHISPER_MODEL=tiny.en` or a CUDA GPU; or `STT_ENGINE=google` |
| A command was misunderstood | rephrase as a command ("open …", "set …"); questions are always answered, never executed |
| Something went wrong | say "emergency stop" (or press Stop everything), check the console's activity log, "undo that" for file operations |

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
