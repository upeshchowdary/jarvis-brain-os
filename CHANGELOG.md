# Changelog

All notable changes to JARVIS Brain OS. Baseline before the v3 upgrade: commit `f3b703e`
(local tag `v2.5-pre-upgrade`).

## Phase 0 — Cross-platform imports & test harness

**What changed**
- `core/config.py` is now the single configuration (merges `app/config/settings.py` and
  `brain/brain_config.py`, which remain as thin re-export shims). Server binds `127.0.0.1` by default.
- `core/paths.py`: every database / cache / workflow lives under `data/` (or `JARVIS_DATA_DIR`),
  never relative to the current working directory.
- `core/logging.py`: one loguru setup (with request-ID field) instead of two competing ones.
- Windows-only imports are guarded (`winreg`), optional ones are lazy (`ddgs`, `speech_recognition`,
  `pyttsx3`). The TTS thread is no longer started at import time.
- App discovery (PowerShell `Get-StartApps`, registry, Program Files walk) no longer runs on import;
  it runs on first use and is cached in `data/app_index.json` for 24 h.
- Removed `shell=True` from application launching (argument lists only).
- Fixed the `import memory` → `import brain` circular import (lazy `brain/__init__`, memory depends on `core`).
- Deleted the unused duplicate stacks `app/llm`, `app/prompts`, `app/domain`, `app/tools`, `app/memory`
  (only tests and `/health` imported them; the chat path never did) and the user-specific `pyrightconfig.json`.
- Test harness: GUI/OS modules are mocked on every platform, data dir is a temp folder, network tests are
  opt-in (`pytest -m network`). Project ruff config bans `shell=True` / `os.system`.

**Verify**: `python -m pytest -q` (Linux or Windows) and `python -c "import memory, brain"`.

## Phases 1–3 — Safety gateway, intent routing, one architecture

(Delivered together: the gateway, the router and the new LLM layer depend on each other.)

**Phase 1 — Safety & security**
- `automation/gateway.py` — **ActionGateway**, the only door to side effects. Every operation is registered with a
  risk level (LOW / MEDIUM / HIGH), previewed, dry-run aware, confirmed (HIGH always; MEDIUM unless
  `CONFIRM_MEDIUM_RISK=false`), audited to `data/audit.jsonl`, blocked by emergency stop, and can record an undo.
  Pending confirmations are per session with a 60 s TTL; "yes/confirm/do it" runs them, anything else cancels.
  Risk overrides in `data/risk_policy.json` can relax MEDIUM but never HIGH.
- Operations live in the private package `automation/ops/` (`tests/test_gateway_isolation.py` fails if anything
  else imports it, or if `shell=True` / `os.system` / `create_subprocess_shell` appears anywhere).
- Shutdown/restart are scheduled with a 30 s delay and `cancel shutdown` support.
- `process.kill`: exact executable-name or PID match only; never JARVIS's own process tree or protected system
  processes; the confirmation lists the matched PIDs.
- No free-form shell: the planner/LLM can't emit `run_command`; `shell.run_named` runs allow-listed argument lists.
- Dry-run has one source of truth (`settings.DRY_RUN`) plus a per-request override passed to the gateway;
  `POST /automation/execute {"dry_run": true}` returns the plan only.
- API: binds 127.0.0.1, bearer token (auto-generated into `.env`) on every route except `/health`, CORS closed by
  default, per-client rate limit, request IDs, docs hidden unless `DEBUG=true`.
- Messaging: draft → preview → confirmation → open chat → unique exact contact match → chat-header verification →
  send; clipboard restored; ambiguous contacts are listed; Discord/Slack say "not supported yet" (no WhatsApp
  fallback); email opens a Gmail draft for the user to send; no `os.system`.
- Prompt injection: web pages, file contents and screen text are passed as tool results inside
  `<untrusted_data>` blocks, and once such content is in context side-effect tools are withdrawn for the rest of
  the turn. `PRIVACY_MODE=local` keeps file/screen content on Ollama/LM Studio.

**Phase 2 — Intent understanding that doesn't misfire**
- `brain/router.py`: anchored whole-utterance command grammar (~60 rules) for unambiguous commands — no LLM, ~1 ms.
- Everything else goes to the LLM agent with **native tool calling**. Questions/statements are only offered
  read-only tools; side-effect tools are offered for imperative requests only. A `clarify` tool asks instead of guessing.
- `tests/test_routing_negatives.py`: 120 benign questions (every query from the bug report included) — no
  side-effect tool is selected, and an adversarial scripted LLM that *tries* to restart/kill/send cannot act.

**Phase 3 — Consolidated architecture**
- One LLM gateway `core/llm` (OpenAI-compatible: Gemini, Groq, OpenAI, Anthropic, OpenRouter, DeepSeek, Ollama,
  LM Studio) with tool calling, streaming (text + tool-call deltas), retries/backoff on 429/5xx, fallback chain,
  shared keep-alive client, async cached connectivity hint. Fixes the dead `stream()` and the Ollama "not found"
  misdetection. LiteLLM not adopted (two PyPI compromises in 2026).
- One config (`core/config.py`), one automation stack (engine plans → `computer.run_plan` gateway op), one tool
  registry (`tools/`, 40+ tools with JSON schemas and risk levels), and a small pipeline (`brain/pipeline.py`)
  replacing the 670-line `execute_cognitive_pipeline`.
- Deleted duplicates/placeholders: legacy `system_controller` (read-only file helpers kept in
  `automation/files_read.py`), `task_manager`, `action_executor`, `safety_manager`, `orchestrator`, `intent_engine`,
  `model_manager`, `tool_router`, `planner`/`reasoning`/`reflection`/`knowledge` stubs, prompt stores, and the old
  `tools/*_tool.py` (the old filesystem tool wrote files without any safety check).
- Per-session state with LRU/TTL (`core/session.py`): model choice, vision cache — switching models in one
  session no longer changes it for voice/CLI/API.
- Memory: names only from explicit statements ("I am tired" ≠ name, names containing "a" work), stable sha1 keys,
  DB initialised once, relevance-ranked & capped facts per turn, `remember / what do you know about me / forget`,
  optional LLM durable-fact extraction in the background.
- `.env.example` added (and allowed by `.gitignore`); `chat.py` streams and authenticates.

**Verify**: `python -m pytest -q` (758 tests), `ruff check .`, then try `restart my pc` → confirmation → `yes`.

## Phase 4 — Voice, streaming, MCP, browser agent

- **Streaming chat**: `POST /chat/stream` (Server-Sent Events: `text`, `tool`, `done`); `chat.py` prints tokens live.
- **Voice**
  - STT: local **faster-whisper** (MIT; int8 on CPU / float16 on GPU, model pre-loaded at start) with the free Google
    web API as fallback; Whisper's "Thank you." hallucinations on noise are discarded.
  - TTS: optional **Kokoro-82M** neural voice (kokoro-onnx MIT + Apache-2.0 weights, local) with interruptible playback;
    pyttsx3/SAPI fallback. **Streaming TTS** speaks each sentence as soon as the LLM produces it.
  - Optional acoustic wake word via **openWakeWord** ("hey jarvis"): nothing is transcribed until you address JARVIS.
    (Its bundled pre-trained models are non-commercial — fine for personal use.)
  - Spoken confirmations: "I need your confirmation… Say yes to confirm"; the follow-up "yes" needs no wake word.
  - numpy-vectorised RMS/VAD; snappier 0.9 s end-of-speech; "like" is no longer stripped as a filler word.
- **MCP**: `python -m integrations.mcp_server` exposes JARVIS (chat, status, search, files, apps, volume, pending
  actions) to MCP clients — they can never confirm their own actions. `data/mcp_servers.json` plugs external MCP
  servers in as tools; write tools go through the gateway op `mcp.call`.
- **Browser agent** (`browser_task` tool / `browser.agent` op): Playwright observe→decide→act loop over numbered
  interactive elements with a persistent profile; never types passwords; stops before buy/pay/send/delete/post.
  Verified live in Chromium (`tests/test_browser_agent.py`).

**Verify**: `pytest tests/test_voice_pipeline.py tests/test_mcp.py tests/test_browser_agent.py`.

## Phase 5 — New capabilities

Every action-taking feature runs through the ActionGateway with the right risk level.

- **Reminders & schedules** (`services/scheduler.py`, `services/timeparse.py`): "remind me at 6 pm to submit the
  report", "every weekday at 9 open my class links", "set a timer for 10 minutes", list/cancel. Persisted in
  SQLite, survives restarts (late reminders are delivered and marked late), recurring daily/weekdays/weekly/every-N.
  Windows toast + spoken alert + dashboard event. Scheduled commands/routines still obey the gateway.
- **Routines / modes** (`services/routines.py`, `routines/*.yaml`, `data/routines/*.yaml`): YAML with variables
  and conditions (`battery_below`, `app_running`, `weekday`, `after/before`), timers, spoken lines, briefing.
  All OS steps run as ONE gateway plan (one confirmation). Built-ins: study mode, shutdown routine, morning.
- **Workflow recorder v2**: editable YAML (old JSON still loads), `{param}` placeholders ("replay send report with
  name=Rahul"), triggers (time, file created in a folder, app launched, USB connected), replay through
  `computer.run_plan` with verification/retry.
- **File management with undo** (`automation/ops/files.py`): create/append/rename/move/copy, delete → Recycle Bin
  (`send2trash`), organize a folder by type or month, bulk rename with preview, zip/unzip (zip-slip safe),
  duplicate finder (hash), content search, "undo that". System folders and JARVIS data are refused.
- **Documents** (`services/documents.py`): read/summarise PDF/DOCX/PPTX/XLSX, local Q&A over a folder (BM25, cached
  index, cited passages), extract tables to Excel, convert md/txt ↔ docx, csv ↔ xlsx, any → txt/md.
- **Productivity**: notes, to-dos, clipboard history ("paste what I copied 3 items ago"), daily briefing (weather via
  free wttr.in, reminders, to-dos, PC health), weather.
- **System health monitor** (`services/health.py`): low battery, sustained high CPU, RAM, disk, temperature →
  throttled proactive alerts with suggested fixes (ending a process still needs confirmation).
- **Developer helper**: find projects by name (remembered), git status / today's commits (summarised), run tests
  (confirmed), open project in VS Code.
- **Web research**: multi-query search + concurrent page reading → numbered sources for cited answers.
- **Desktop UX**: `/dashboard` (chat with live streaming, approve/deny buttons, live activity feed, reminders,
  to-dos, notes, routines, audit log, emergency stop; strict CSP, no external scripts) and `jarvis.py --tray`
  (system-tray menu + Win+J hotkey). `jarvis.py` launches everything.
- **Tool selection**: only the relevant subset of the 80+ tools is sent to the LLM per request (core + keyword
  groups), keeping prompts small and tool choice accurate.

**Verify**: `pytest tests/test_productivity_features.py`, then `python jarvis.py --open` and try
"remind me in 1 minute to stretch", "start study mode", "organize my downloads" → approve → "undo that".
