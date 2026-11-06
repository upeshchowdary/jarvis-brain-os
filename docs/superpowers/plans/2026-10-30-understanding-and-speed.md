# Understanding & Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** JARVIS understands follow-ups and multi-part requests, carries tasks to completion without asking, remembers what it did, can search its own chat history, and answers faster.

**Architecture:** A new intent step (Groq gpt-oss-120b → Gemini flash-lite fallback) rewrites each non-instant message into a standalone request with kind/tool-groups, running concurrently with memory lookups. A PC-wide current task (with plan + steps, persisted in SQLite) and a cross-channel recent-turns buffer give it context. The agent loop runs to completion for multi-step actions on gemini-3.5-flash. Memory search fuses SQLite FTS5 with embedded ChromaDB (local bge-base via fastembed).

**Tech Stack:** Python 3.13, FastAPI, httpx, aiosqlite + SQLite FTS5, chromadb 1.5.x (embedded), fastembed 0.9.x (`BAAI/bge-base-en-v1.5`), pywinauto keyboard, pytest + pytest-asyncio, optional langfuse.

**Spec:** `docs/superpowers/specs/2026-10-30-understanding-and-speed-design.md`

## Global Constraints

- Run everything with the venv: `.venv\Scripts\python.exe -m pytest …` from `E:\jarvis`.
- Tests never call real LLMs or touch the real desktop (existing `tests/conftest.py` guarantees; keep it that way). Network evals only under `-m network`.
- Settings (exact names/defaults, `core/config.py`): `INTENT_MODEL="groq/openai/gpt-oss-120b"`, `INTENT_FALLBACK_MODEL="gemini/gemini-3.5-flash-lite"`, `ACTION_MODEL="gemini/gemini-3.5-flash"`, `REASONING_EFFORT={"gemini-3.5-flash-lite":"minimal","gemini-3.5-flash":"minimal","openai/gpt-oss-120b":"low"}`, `LLM_KEEPWARM=True`, `TASK_IDLE_SECONDS=900`, `AGENT_MAX_STEPS_ACTION=15` (AGENT_MAX_STEPS stays 5), `INTENT_HISTORY_TURNS=8`, `INTENT_HISTORY_WINDOW_SECONDS=1800`, `EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"`, `LANGFUSE_ENABLED=False`. Document each in `.env.example`.
- Voice: `SILENCE_TIMEOUT_MS=600`.
- Questions never act: `intent.kind != "action"` ⇒ only `registry.read_only()` tools.
- Safety router rules (stop / resume / dry run / automation status) always run first, regardless of task state.
- Unchanged: emergency stop, dry run, audit log, undo, taint rule (no side-effect tools after untrusted content in a turn).
- Messaging tests use mocks only; live messaging checks only with the user's own WhatsApp "Message yourself" chat.
- **Commits:** the user has not authorized commits yet. Run the commit steps only once they say so; if they do, first create branch `feat/understanding-and-speed`.
- Existing suite must stay green except the two known order-dependent flakes (`test_mcp_client_cannot_confirm_its_own_action`, `test_productivity_api` / `test_undo_last_runs_recorded_inverse` when run after productivity tests); each must pass in isolation.

## Review Focus

1. **`chat_history` "today" across the UTC boundary** — `chat_logs.created_at` is UTC ISO; the user is in IST (UTC+5:30). A turn at 00:30 IST belongs to *today*, though its UTC date is yesterday. → test in Task 4.
2. **"stop" during an active task** must still trigger the emergency stop (fast path), never be swallowed by the intent step. → test in Task 6.
3. **A question in the middle of a task** ("what time is it?" while a WhatsApp task is active) is answered without acting and without replacing/shelving the task. → test in Task 6.
4. **Intent model rate-limited (429) then fallback also failing** → legacy routing still answers; no exception reaches the user. → test in Task 5.
5. **Embedding model unavailable (offline first run / fastembed import error)** → FTS-only search, no crash, startup not blocked. → test in Task 3.

---

### Task 1: LLM gateway — reasoning effort, structured output, keep-warm

**Files:**
- Modify: `core/config.py` (new settings from Global Constraints: INTENT_MODEL, INTENT_FALLBACK_MODEL, ACTION_MODEL, REASONING_EFFORT, LLM_KEEPWARM)
- Modify: `core/llm/gateway.py` (`_payload`, `chat`, `stream_events`, new `warm_up`, `start_keepwarm`, `stop_keepwarm`)
- Modify: `app/main.py` lifespan (call `await llm.warm_up()` and `llm.start_keepwarm()` when `BACKGROUND_SERVICES`; `llm.stop_keepwarm()` on shutdown)
- Modify: `.env.example`
- Test: `tests/test_llm_gateway.py`

**Interfaces:**
- Produces: `settings.REASONING_EFFORT: Dict[str, str]` (JSON in env; validator accepts JSON string). `LLMGateway.chat(..., response_format: Optional[Dict[str, Any]] = None)`; `LLMGateway.warm_up() -> None` (async, one `max_tokens=1` request per configured cloud provider in the DEFAULT/ACTION/INTENT models, errors swallowed); `LLMGateway.start_keepwarm(interval: float = 50.0) -> None` / `stop_keepwarm() -> None` (background task re-running `warm_up` while idle; no-op when `LLM_KEEPWARM` false).

- [ ] **Step 1: Write failing tests**

```python
async def test_reasoning_effort_sent_per_model(gw, monkeypatch):
    monkeypatch.setattr(settings, "REASONING_EFFORT", {"gemini-test": "minimal", "llama-test": "low"})
    seen = gw(lambda r: ok("x"))
    await llm.chat([{"role": "user", "content": "x"}])
    assert json.loads(seen[0].content)["reasoning_effort"] == "minimal"
    seen.clear()
    await llm.chat([{"role": "user", "content": "x"}], model="ollama/qwen-test")
    assert "reasoning_effort" not in json.loads(seen[0].content)

async def test_response_format_passthrough(gw):
    seen = gw(lambda r: ok("{}"))
    fmt = {"type": "json_schema", "json_schema": {"name": "intent", "schema": {"type": "object"}}}
    await llm.chat([{"role": "user", "content": "x"}], response_format=fmt)
    assert json.loads(seen[0].content)["response_format"] == fmt

async def test_warm_up_hits_each_configured_cloud_provider_once(gw, monkeypatch):
    monkeypatch.setattr(settings, "INTENT_MODEL", "groq/llama-test")
    monkeypatch.setattr(settings, "ACTION_MODEL", "gemini/gemini-test")
    seen = gw(lambda r: ok("x"))
    await llm.warm_up()
    hosts = sorted({r.url.host for r in seen})
    assert hosts == ["api.groq.com", "generativelanguage.googleapis.com"]

async def test_warm_up_swallows_errors(gw):
    gw(lambda r: httpx.Response(500, text="down"))
    await llm.warm_up()  # must not raise
```

- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_llm_gateway.py -q` — Expected: 4 new FAIL.
- [ ] **Step 3: Implement.** In `_payload` add `reasoning_effort` when `settings.REASONING_EFFORT.get(model)` is set and the provider is not local; add `response_format` when given. Thread `response_format` through `chat` only (intent step is non-streaming). Keep-warm uses `asyncio.create_task` with a loop sleeping `interval`; skip a cycle if a real request happened in the last `interval` seconds (track `self._last_request` in `_post`/stream).
- [ ] **Step 4: Run** the gateway tests — Expected: all PASS.
- [ ] **Step 5: Commit** (when authorized) `feat(llm): per-model reasoning effort, structured output, keep-warm`.

---

### Task 2: Current task + recent turns + `task_plan` tool

**Files:**
- Create: `brain/task_state.py`
- Create: `tools/builtin/task_tools.py`; register in `tools/builtin/__init__.py`
- Modify: `app/database/connection.py` (table `tasks`, `save_task`, `load_recent_tasks`)
- Modify: `app/main.py` lifespan (`await tasks.load()` after `init_db`)
- Modify: `core/config.py` (`TASK_IDLE_SECONDS`, `INTENT_HISTORY_TURNS`, `INTENT_HISTORY_WINDOW_SECONDS`)
- Test: `tests/test_task_state.py`

**Interfaces:**
- Produces (in `brain/task_state.py`):
  - `@dataclass PlanItem(subgoal: str, status: str = "todo")` — status ∈ todo|doing|done|failed
  - `@dataclass TaskStep(tool: str, args_summary: str, result_summary: str, ok: bool, at: float)`
  - `@dataclass Task(id: str, goal: str, status: str, channel: str, plan: List[PlanItem], steps: List[TaskStep], created_at: float, updated_at: float)` with `to_dict()`, `from_dict()`, `summary(max_steps: int = 3) -> str` (goal, status, plan with ✓/→ markers, last steps — used in prompts)
  - `class TaskTracker`: `current() -> Optional[Task]` (returns None and shelves the task when idle > `TASK_IDLE_SECONDS`); `start(goal: str, channel: str) -> Task` (shelves the previous active one); `record_step(tool: str, args: Dict, result_message: str, ok: bool) -> None`; `set_plan(subgoals: List[str]) -> None`; `mark_plan(index: int, status: str) -> None`; `set_status(status: str) -> None`; `revive_last() -> Optional[Task]`; `async load() -> None`. Every mutation updates `updated_at` and schedules `db_manager.save_task(task.to_dict())` (fire-and-forget). Thread-safe via `threading.Lock` (voice and API run concurrently).
  - `@dataclass Turn(channel: str, session_id: str, user: str, assistant: str, at: float)`; `class RecentTurns`: `add(channel, session_id, user, assistant) -> None` (ring buffer, 50), `recent(max_turns: int, window_s: float) -> List[Turn]` (oldest→newest, all channels)
  - Singletons: `tasks = TaskTracker()`, `recent_turns = RecentTurns()`
- Produces (tool `task_plan`, side_effect False, final False): params `action: "set"|"done"|"failed"|"revise"`, `subgoals: List[str]` (for set/revise), `index: int` (for done/failed). Returns the updated plan text.

- [ ] **Step 1: Write failing tests** — names and assertions:
  - `test_start_shelves_previous_task`: start A, start B → `current().goal == "B"`, A's status `"shelved"`.
  - `test_idle_task_is_shelved` (monkeypatch `time.time`): after `TASK_IDLE_SECONDS + 1` → `current() is None`; `revive_last().goal == "A"` and status `"active"`.
  - `test_record_step_and_summary`: two steps → `summary()` contains goal, both tool names, `"✓"`.
  - `test_plan_next_unfinished`: `set_plan(["open WhatsApp","find Nithesh","send"])`, `mark_plan(0,"done")` → summary marks item 2 with `"→"`.
  - `test_task_persists_and_reloads`: start + step, `await asyncio.sleep(0)`, new `TaskTracker()` → `await load()` → `current().goal` equal and 1 step.
  - `test_recent_turns_cross_channel_window`: add voice turn at t, console turn at t+10, old turn at t-4000 → `recent(8, 1800)` returns the two newest, oldest first, channels `["voice","text"]`.
  - `test_task_plan_tool_sets_and_marks`: run tool `set` then `done index=0` → `tasks.current().plan[0].status == "done"`.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_task_state.py -q` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement** the interfaces above. `tasks` table: `id TEXT PRIMARY KEY, status TEXT, data TEXT (json), updated_at REAL`; `load_recent_tasks(limit=5)` orders by `updated_at DESC`. `load()` restores the newest non-done/failed task as current if within idle window, else keeps it as last shelved.
- [ ] **Step 4: Run** task-state tests + full suite — Expected: PASS.
- [ ] **Step 5: Commit** (when authorized) `feat(brain): PC-wide current task with plan, recent turns, task_plan tool`.

---

### Task 3: Hybrid memory search (FTS5 + ChromaDB/fastembed)

**Files:**
- Modify: `pyproject.toml` — `memory = ["chromadb>=1.5,<2", "fastembed>=0.9,<1"]`; add `memory` to `all`. Install: `uv pip install --python .venv\Scripts\python.exe -e ".[all,dev]"`.
- Modify: `app/database/connection.py` — FTS5 table `chat_fts` (content=`chat_logs`, columns `user_query, response_text`, `content_rowid=id`) + insert trigger; one-time backfill `INSERT INTO chat_fts(chat_fts) VALUES('rebuild')` when the FTS table is new.
- Modify: `memory/vector_memory.py` — Chroma `PersistentClient(path=data_path("chroma_db"))`, collection `jarvis_memory` (cosine), embedding function = fastembed `TextEmbedding(settings.EMBEDDING_MODEL)` wrapper; `set_embedding_function(fn)`; `warm_up()`; LRU(256) cache of query embeddings; all Chroma/fastembed calls via `asyncio.to_thread`; keep the keyword fallback path.
- Create: `memory/search.py`
- Modify: `app/main.py` lifespan — start `vector_memory.warm_up()` as a background task (must not block startup).
- Modify: `brain/pipeline.py` `_persist` — index every turn (not only `conversation`) as `"User: {query} | JARVIS: {response[:300]}"` with metadata `{chat_id, created_at, channel}`.
- Test: `tests/test_memory_search.py`

**Interfaces:**
- Consumes: `db_manager` (FTS queries), `vector_memory`.
- Produces (in `memory/search.py`): `@dataclass Hit(chat_id: Optional[int], text: str, created_at: str, score: float, source: str)`; `async def search(query: str, limit: int = 10, since_utc: Optional[datetime] = None, until_utc: Optional[datetime] = None) -> List[Hit]` — FTS5 (bm25) and Chroma results fused with reciprocal rank fusion, `k=60`, then a recency tie-break; `async def recent(limit: int, since_utc: Optional[datetime] = None, until_utc: Optional[datetime] = None, channel: Optional[str] = None) -> List[Hit]` (no query; newest first).
- Produces (in `memory/vector_memory.py`): `vector_memory.set_embedding_function(fn: Callable[[List[str]], List[List[float]]]) -> None`; `async warm_up() -> None`; existing `search_similar` / `add_document_async` keep their signatures.

- [ ] **Step 1: Write failing tests** (fixture installs a deterministic fake embedding — e.g. 16-dim bag-of-hashed-words — via `set_embedding_function`; never downloads a model):
  - `test_fts_finds_exact_name`: log turns "message Nithesh on WhatsApp" and "play lofi" → `search("Nithesh")[0].text` contains "Nithesh".
  - `test_rrf_merges_both_sources`: a doc found only by vectors and one only by FTS both appear in top 5; a doc found by both ranks first.
  - `test_time_window_filters`: `since_utc` excludes older turns.
  - `test_fts_only_when_embeddings_unavailable` (Review Focus 5): make the embedding function raise → `search` still returns FTS hits; no exception.
  - `test_warm_up_does_not_block`: embedding fn sleeps 0.5 s; `asyncio.wait_for(asyncio.sleep(0.05), 0.2)` completes while `warm_up()` runs as a task.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_memory_search.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** per the interfaces. RRF score = Σ 1/(60 + rank). Chat ids in Chroma metadata make fusion keyed by `chat_id`; non-chat docs (facts/notes) are keyed by doc id.
- [ ] **Step 4: Run** memory tests + `tests/test_persistence.py tests/test_phase2_tools_memory.py` — Expected: PASS.
- [ ] **Step 5: Live latency check:** script in the scratchpad indexes the existing `data/jarvis.db` turns with the real bge-base model, then times 20 searches. Expected: median ≤ 30 ms after warm-up. Record the number.
- [ ] **Step 6: Commit** (when authorized) `feat(memory): hybrid FTS5 + ChromaDB search with local bge-base`.

---

### Task 4: `chat_history` tool

**Files:**
- Create: `tools/builtin/history.py`; register in `tools/builtin/__init__.py`
- Modify: `tools/selection.py` — group `"history"`: tools `["chat_history"]`, keywords `["chat", "chats", "asked you", "earlier", "yesterday", "conversation", "we did", "history", "last time", "before"]`
- Test: `tests/test_history_tool.py`

**Interfaces:**
- Consumes: `memory.search.search`, `memory.search.recent`.
- Produces: tool `chat_history` (read-only, `final=False`), params `query: str` (optional), `since: str` (optional; `today|yesterday|this week|last week|YYYY-MM-DD`), `channel: str` (optional), `limit: int` (default 20, max 50). Local-day boundaries computed in the machine's local timezone, converted to UTC for filtering. Output: one line per turn `"[HH:MM] You: … → JARVIS: …"` (local time), newest last; `content_for_llm` = same text.

- [ ] **Step 1: Write failing tests:**
  - `test_today_filter_uses_local_midnight` (Review Focus 1): monkeypatch local tz to UTC+5:30; turn stored at `2026-10-29T19:00:00+00:00` (00:30 IST on the 30th) and one at `2026-10-29T18:00:00+00:00` (23:30 IST on the 29th); "now" = 2026-10-30 10:00 IST → `since="today"` returns only the first.
  - `test_query_uses_search`: `query="whatsapp"` returns the WhatsApp turn first.
  - `test_empty_history_message`: no turns → message `"No chats found for that period."`
- [ ] **Step 2: Run** — Expected: FAIL. **Step 3: Implement.** **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** (when authorized) `feat(tools): chat_history tool`.

---

### Task 5: Intent step

**Files:**
- Create: `brain/intent.py`
- Modify: `core/config.py` (intent settings already added in Task 1/2)
- Test: `tests/test_intent.py`

**Interfaces:**
- Consumes: `core.llm.llm.chat(messages, model=…, response_format=…, max_tokens=300, temperature=0)`, `Turn`, `Task` (Task 2), `router.is_imperative`, `tools.selection.GROUPS`.
- Produces:
  - `@dataclass Intent(request: str, kind: str, continues_task: bool, single_step: bool, tool_groups: List[str], needs_screen: bool, source: str)` — `kind` ∈ action|question|chat; `source` ∈ llm|legacy
  - `INTENT_SCHEMA: Dict[str, Any]` — JSON schema of the fields above (minus `source`), `tool_groups` items enum = `list(GROUPS) + ["keyboard", "history"]`
  - `async def analyze(message: str, turns: List[Turn], task: Optional[Task]) -> Intent` — tries `settings.INTENT_MODEL` then `settings.INTENT_FALLBACK_MODEL`, each with one retry on unparsable/invalid output; on total failure returns `legacy_intent(message)`. Logs and never raises.
  - `def legacy_intent(message: str) -> Intent` — `kind="action" if router.is_imperative(message) else "question"`, `request=message`, `continues_task=False`, `single_step=False`, `tool_groups=[]`, `source="legacy"`.
- Prompt (static part first for caching): the rules — rewrite into one complete standalone instruction resolving pronouns/"next"/"that"/"yes" from the conversation and the current task's first unfinished plan item; `kind="action"` whenever the user wants something done on the computer, however phrased ("can you…", "yes open", "i said in the previous message"); `kind="question"` for how-to/explanations even if they contain verbs ("how do I restart my router?"); `single_step=true` only for one atomic action; never invent details not in the conversation. Dynamic part last: current task summary, recent turns, new message.

- [ ] **Step 1: Write failing tests** (patch `brain.intent.llm` with a `FakeLLM` returning JSON text):
  - `test_parses_valid_json`: fake returns the "next do" JSON → `Intent.request == "Send a WhatsApp message to Nithesh saying hi"`, `kind == "action"`, `source == "llm"`.
  - `test_prompt_contains_task_and_turns`: the fake's recorded messages include the task goal and both prior turns, and the system message comes first and contains no turn text (cacheable).
  - `test_retry_then_fallback_model`: intent model returns garbage twice → fallback model called (assert `fake.calls[2]["model"] == settings.INTENT_FALLBACK_MODEL`).
  - `test_total_failure_returns_legacy` (Review Focus 4): fake raises `LLMUnavailable` for every call → `source == "legacy"`, no exception; `"open chrome"` → `kind == "action"`; `"how do I restart my router?"` → `kind == "question"`.
  - `test_unknown_tool_groups_dropped`: response with `["messaging","bogus"]` → `["messaging"]`.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_intent.py -q` — Expected: FAIL. **Step 3: Implement.** **Step 4: Run** — Expected: PASS.
- [ ] **Step 5: Commit** (when authorized) `feat(brain): intent step with standalone rewrite and fallbacks`.

---

### Task 6: Pipeline integration + act-don't-ask rules

**Files:**
- Modify: `brain/pipeline.py` (`stream`, `_agent`, `finish`, `_memory_context`)
- Modify: `brain/router.py` (`fast_route(text, task_active=False)`, `CONTEXT_WORDS`, safety-rule precedence)
- Modify: `brain/prompts.py` (`AGENT_RULES` rewrite; `build_system_prompt(..., task_summary: Optional[str] = None)`)
- Modify: `tools/selection.py` (`select_tools(query, available, context="", groups: Optional[List[str]] = None)`)
- Modify: `brain/conversation_manager.py` (no API change; assistant content now carries action notes)
- Test: `tests/test_understanding.py`; keep `tests/test_pipeline.py`, `tests/test_routing_*.py` green (update tests that assert the old "stop after a final action" behaviour only where `single_step` semantics now apply)

**Interfaces:**
- Consumes: `analyze`, `Intent` (Task 5); `tasks`, `recent_turns` (Task 2); `search` (Task 3); `chat_history`, `task_plan` tools; `settings.ACTION_MODEL`, `settings.AGENT_MAX_STEPS_ACTION`.
- Produces:
  - `router.CONTEXT_WORDS = {"next", "previous", "back", "again", "that", "it", "do it", "continue", "this", "same", "more", "go on"}`; `fast_route(text, task_active)` returns `None` when `task_active` and the normalized text is in `CONTEXT_WORDS`; safety rules (names `stop`, `resume`, `dry_run_on`, `dry_run_off`, `automation_status`) always match first.
  - `select_tools(..., groups=[…])`: when `groups` is given, CORE + those groups' tools + `task_plan` + `chat_history`, keyword matching skipped; when `None`, existing behaviour.
  - New stream event `{"type": "progress", "text": str}` emitted when a multi-step action task starts (`"On it."`) and after each successful tool step (the tool's result message, first sentence, ≤ 60 chars).
- Behaviour (in order):
  1. Pending confirmation (unchanged).
  2. `fast_route(query, task_active=tasks.current() is not None)`; on a hit, run as today and `tasks.record_step(...)` if a task is active.
  3. `asyncio.gather(analyze(query, recent_turns.recent(settings.INTENT_HISTORY_TURNS, settings.INTENT_HISTORY_WINDOW_SECONDS), tasks.current()), self._memory_context(query, ...))`.
  4. If `intent.request != query`, retry `fast_route(intent.request, task_active=False)`; a hit runs instantly.
  5. Task bookkeeping: `kind=="action"` and not `continues_task` → `tasks.start(intent.request, ctx.channel)`; `continues_task` with no current task → `tasks.revive_last()`; `kind!="action"` → task untouched (Review Focus 3).
  6. Agent: base tools = `registry.all()` if `kind=="action"` else `registry.read_only()`; `select_tools(intent.request, base, groups=intent.tool_groups or None)`; model = `model_override or (settings.ACTION_MODEL if kind=="action" else settings.DEFAULT_MODEL)`; max steps = `AGENT_MAX_STEPS_ACTION` for actions; user message = `intent.request` (the original text is appended as `"(user's exact words: …)"` when they differ); system prompt includes `tasks.current().summary()`.
  7. The "all tools final → end turn" shortcut runs only when `intent.single_step`.
  8. Every executed tool → `tasks.record_step`. Turn end: action task with no remaining todo plan items and a successful last step → `set_status("done")`; agent ended after a failed step → `"failed"`.
  9. `finish`: assistant history content = response + `" [did: tool(arg) ✓/✗, …]"` (≤ 200 chars); `recent_turns.add(...)`.
- `AGENT_RULES` (replace the "If you are not sure … call `clarify`" line; keep the untrusted-data, never-claim-success, sources and concision lines): act on action requests immediately with sensible defaults (installed desktop app before web version, default browser, top result); ask only when a required value cannot be inferred (e.g. missing message text), never for permission; for multi-step tasks call `task_plan` first, then work through it; if a step fails, try up to 2 different approaches (another tool, keyboard instead of mouse, web instead of app) before reporting the blocker in one sentence; `clarify` is a last resort.

- [ ] **Step 1: Write failing tests** (`FakeLLM` for the agent; patch `brain.pipeline.analyze` to return scripted `Intent`s; OS ops mocked as in `tests/test_pipeline.py::win`):
  - `test_multi_step_continues_after_final_tool`: intent action, single_step False; agent script = `call("open_app", name="whatsapp")`, `call("send_message", …)`, `text("Sent.")` → both tools ran, response "Sent.".
  - `test_single_step_keeps_shortcut`: single_step True, `call("open_app", …)` → exactly one LLM call.
  - `test_followup_uses_rewritten_request`: intent.request "Send a WhatsApp message to Nithesh saying hi" → last user message sent to the fake contains it and `"(user's exact words: next do)"`.
  - `test_yes_open_gets_action_tools`: message "yes open", intent kind action → offered tools include `open_app`.
  - `test_question_gets_read_only_tools`: kind question → no side-effect tool offered.
  - `test_next_mid_task_not_media_key`: active task; "next" → `media_control` not run; intent called.
  - `test_next_without_task_is_media_key`: no task; "next" → `media_control` ran, intent not called.
  - `test_stop_during_task_is_emergency_stop` (Review Focus 2): active task; "stop" → route fast_path, `gateway.is_stopped`; intent not called.
  - `test_question_mid_task_keeps_task` (Review Focus 3): active task "send WhatsApp…"; intent kind question for "what time is it?" → `tasks.current().goal` unchanged, status active, no side-effect tool offered.
  - `test_action_notes_in_history`: after an `open_app` turn, `conversation_manager.get_history(sid)[-1]["content"]` contains `"[did: open_app("`.
  - `test_action_uses_action_model`: kind action → `fake.calls[0]["model"] == settings.ACTION_MODEL`; kind chat → `settings.DEFAULT_MODEL`.
  - `test_intent_failure_falls_back_to_legacy`: `analyze` returns a legacy Intent → "open chrome now please" still works through the agent with action tools.
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/test_understanding.py -q` — Expected: FAIL.
- [ ] **Step 3: Implement** behaviour 1–9 and the router/selection/prompt changes.
- [ ] **Step 4: Run** `.venv\Scripts\python.exe -m pytest -q` — Expected: all PASS except the known flakes (each passes alone).
- [ ] **Step 5: Commit** (when authorized) `feat(brain): intent-driven routing, current task, run-to-completion agent`.

---

### Task 7: Keyboard & window fast path

**Files:**
- Create: `automation/ops/keyboard.py`; add to imports in `automation/ops/__init__.py`
- Create: `tools/builtin/keyboard.py`; register in `tools/builtin/__init__.py`
- Modify: `brain/router.py` (rules below), `tools/selection.py` (group `"keyboard"`: tools `["press_keys", "type_text", "focus_window"]`, keywords `["tab", "scroll", "type", "press", "paste", "copy", "select all", "refresh", "go back", "switch to", "enter", "escape"]`)
- Test: `tests/test_keyboard_fast_path.py`

**Interfaces:**
- Produces gateway ops (all `Risk.LOW`, Windows-only, implemented with `pywinauto.keyboard.send_keys` via `asyncio.to_thread`): `keyboard.press` args `{"keys": str}` in a normalized form like `"ctrl+tab"`, `"ctrl+shift+t"`, `"pagedown"`, `"alt+left"`, `"f5"`; `keyboard.type` args `{"text": str}`; `window.focus` args `{"name": str}` (reuses the existing window-matching helper in `automation/ops/system_impl.py`; also used for minimize/maximize via the existing `window_control` tool).
- Produces tools: `press_keys(keys: str)`, `type_text(text: str)`, `focus_window(name: str)` — side_effect True, `final=True`.
- Router rules → `RouteDecision`s:

| Utterance | Decision |
|---|---|
| scroll down / scroll up | `press_keys` `"pagedown"` / `"pageup"` |
| page down / page up | same |
| go to (the) top / bottom | `"ctrl+home"` / `"ctrl+end"` |
| next tab / previous tab | `"ctrl+tab"` / `"ctrl+shift+tab"` |
| go to tab N / switch to tab N (1–8; "last" = 9) | `"ctrl+N"` |
| new tab / close (this) tab / reopen (closed) tab | `"ctrl+t"` / `"ctrl+w"` / `"ctrl+shift+t"` |
| go back / go forward / refresh / reload | `"alt+left"` / `"alt+right"` / `"f5"` |
| select all / copy / paste / undo / redo | `"ctrl+a"` / `"ctrl+c"` / `"ctrl+v"` / `"ctrl+z"` / `"ctrl+y"` |
| press <key> (enter, escape, tab, space, backspace, delete, up, down, left, right, f1–f12) | `press_keys` that key |
| type <text> | `type_text` text (original casing preserved) |
| switch to <app> | `focus_window` name |
| minimize this / maximize this | existing `window_control` with the foreground window |

  "next"/"back" alone are **not** keyboard rules (context words, Task 6).

- [ ] **Step 1: Write failing tests:** parametrized over the table (`fast_route(utterance)` → expected tool + args); `test_type_preserves_case` ("type Hello World" → `"Hello World"`); `test_ops_call_send_keys` with `send_keys` mocked: `keyboard.press "ctrl+tab"` → `send_keys("^{TAB}")`, `"ctrl+shift+t"` → `send_keys("^+t")`, `"pagedown"` → `send_keys("{PGDN}")`; `keyboard.type "a+b"` escapes special chars (`send_keys("a{+}b", with_spaces=True)`).
- [ ] **Step 2: Run** — Expected: FAIL. **Step 3: Implement.** **Step 4: Run** this file + `tests/test_routing_*.py` + `tests/test_gateway_isolation.py` — Expected: PASS.
- [ ] **Step 5: Commit** (when authorized) `feat(control): keyboard and window instant commands`.

---

### Task 8: Voice — shorter pause, spoken progress

**Files:**
- Modify: `voice/voice_config.py` (`SILENCE_TIMEOUT_MS: int = 600`)
- Modify: `voice/voice_manager.py` `respond()` — on `{"type": "progress"}` events call `tts_engine.speak(speakable(text))`; skip progress already spoken as a sentence.
- Test: `tests/test_voice_pipeline.py`

**Interfaces:** Consumes the `progress` event (Task 6).

- [ ] **Step 1: Write failing tests:** `test_silence_timeout_is_600`; `test_progress_events_are_spoken` (patch `brain.stream` to yield progress "Opening WhatsApp.", progress "Message sent.", then done → spoken list starts with those two).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** voice tests — PASS.
- [ ] **Step 5: Commit** (when authorized) `feat(voice): 600 ms end-of-speech, spoken progress`.

---

### Task 9: Timings, optional Langfuse tracing

**Files:**
- Modify: `brain/pipeline.py` — `timing_breakdown_ms` gains `intent_ms`, `memory_ms`, `llm_ms`, `tool_ms` (sum), and `metrics.observe("turn.intent", …)`.
- Modify: `app/static/web/js/console.js` — System view lists p50/p95 for `turn.intent` next to existing stages (data from `/metrics`).
- Create: `core/tracing.py` — `trace_turn(...)` context manager; no-op unless `settings.LANGFUSE_ENABLED` and the `langfuse` package imports; records intent, each LLM call (model, tokens, latency) and tool call.
- Modify: `pyproject.toml` — extra `tracing = ["langfuse>=3,<4"]` (not in `all`). `.env.example`: `LANGFUSE_ENABLED=false`, `LANGFUSE_PUBLIC_KEY=`, `LANGFUSE_SECRET_KEY=`, `LANGFUSE_HOST=`.
- Test: `tests/test_tracing.py`, timing assertions in `tests/test_understanding.py`

- [ ] **Step 1: Write failing tests:** `test_timing_breakdown_has_intent_and_memory` (agent turn → keys `intent_ms`, `memory_ms`, `llm_ms` present, numeric); `test_tracing_noop_when_disabled` (no langfuse import attempted: patch `builtins.__import__` to fail on "langfuse" → turn succeeds); `test_tracing_records_when_enabled` (fake langfuse module in `sys.modules` → one trace with an `intent` span).
- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** (when authorized) `feat(observability): per-stage timings, optional Langfuse traces`.

---

### Task 10: Replay eval set + live verification

**Files:**
- Create: `tests/evals/__init__.py`, `tests/evals/intent_cases.yaml`, `tests/evals/test_intent_eval.py`
- Modify: `pyproject.toml` pytest `testpaths` unchanged (evals live under `tests/`), marker `network` already exists.

**Interfaces:** Consumes `analyze` (Task 5). Case format:

```yaml
- id: next-do-whatsapp
  turns: [{user: "open whatsapp and msg nithesh say ee em chesadu", assistant: "Opened WhatsApp. [did: open_app(whatsapp) ✓]"}]
  task: "Send 'ee em chesadu' to Nithesh on WhatsApp desktop"
  message: "next do"
  expect: {kind: action, continues_task: true, request_contains: ["Nithesh", "ee em chesadu"], tool_groups_include: [messaging]}
```

Required cases (from the user's 2026-10-30 chats): `next do`, `i said in the previous message right`, `not windows but in the chrome browser tabs`, `yes open`, `yes take screen shots`, `not in the chrome , i have app in my system , open that and message nithesh`, `scroll down` (after YouTube), `ok good` (→ kind chat, no action), `no need for now` (→ chat), `how do I restart my router?` (→ question), `look at the recent chats I had with you today` (→ tool_groups includes history), `open youtube for me and go to mr beast channel` (→ action, single_step false).

- [ ] **Step 1: Write** the YAML and two tests: `test_eval_cases_offline` (FakeLLM returns each case's expected JSON → validates the harness/schema plumbing for every case) and `@pytest.mark.network test_eval_cases_live` (real `analyze`; computes accuracy; asserts ≥ 0.90; prints failing ids).
- [ ] **Step 2: Run** `.venv\Scripts\python.exe -m pytest tests/evals -q` — offline PASS. Run `.venv\Scripts\python.exe -m pytest tests/evals -m network -q -s` — Expected: accuracy ≥ 90 %; if below, tune the intent prompt (Task 5) and re-run; record the score.
- [ ] **Step 3: Live checks** with JARVIS running (`.venv\Scripts\python.exe jarvis.py --voice`):
  - "open notepad and type hello from jarvis" → Notepad opens with the text; one turn.
  - In the console: "open youtube and go to mr beast channel", then "scroll down", then "next tab" → each acts.
  - Voice: "open notepad"; console: "type it again" → resolves to the same text.
  - "what did I ask you today?" → lists today's turns with local times.
  - Messaging live check only with the user's "Message yourself" WhatsApp chat.
  - Read `/metrics` p50: fast path < 50 ms; first model-handled action ≈ 2–2.5 s; chat first token ≈ 1.5–2 s. Record actual numbers.
- [ ] **Step 4: Full suite** `.venv\Scripts\python.exe -m pytest -q` — Expected: PASS except known flakes (each passes alone).
- [ ] **Step 5: Commit** (when authorized) `test(evals): replay set from real chats; live verification`.
