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
