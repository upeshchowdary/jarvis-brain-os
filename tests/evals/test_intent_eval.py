"""Replay eval for the intent step: the user's real messages (plus unseen paraphrases) → expected intent.

* ``test_eval_cases_offline`` (always runs): validates every case and the scoring harness with a FakeLLM
  that returns each case's expected output.
* ``test_eval_cases_live`` (``-m network``): real ``analyze`` against ``INTENT_MODEL``; prints a per-case
  table, failing cases with the actual output, and latency; asserts accuracy ≥ ``TARGET``.

The suite's conftest blanks every API key and disables the intent step, so the live test reads the real
keys from ``E:\\jarvis\\.env`` itself (never printing them) and skips when they are missing.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import yaml

from brain import intent as intent_mod
from brain.intent import KINDS, TOOL_GROUPS, Intent, analyze
from brain.task_state import PlanItem, Task, TaskStep, Turn
from core.config import BASE_DIR, settings
from tests.fakes import FakeLLM, text

CASES_FILE = Path(__file__).with_name("intent_cases.yaml")
TARGET = 0.90
LIVE_INTENT_MODEL = "groq/openai/gpt-oss-120b"
LIVE_FALLBACK_MODEL = "gemini/gemini-3.5-flash-lite"
_EXPECT_KEYS = {"kind", "continues_task", "single_step", "request_contains", "tool_groups_include"}

# Groq free tier for gpt-oss-120b is 8k tokens/minute: pace calls so the eval measures the model, not 429s.
_TOKENS_PER_MINUTE = 7500
_RATE_LIMIT_RETRIES = 2
_RATE_LIMIT_WAIT_S = 30.0


# ── Loading ─────────────────────────────────────────────────────────────

def load_cases() -> List[Dict[str, Any]]:
    return yaml.safe_load(CASES_FILE.read_text(encoding="utf-8"))


def _alternatives(item: Any) -> List[str]:
    return [str(x) for x in item] if isinstance(item, list) else [str(item)]


def build_turns(case: Dict[str, Any]) -> List[Turn]:
    raw = case.get("turns") or []
    now = time.time()
    return [Turn(t.get("channel", "text"), "eval", t["user"], t["assistant"], now - 60 * (len(raw) - i))
            for i, t in enumerate(raw)]


def build_task(case: Dict[str, Any]) -> Optional[Task]:
    raw = case.get("task")
    if not raw:
        return None
    if isinstance(raw, str):
        raw = {"goal": raw}
    now = time.time()
    return Task(
        id=f"eval-{case['id']}", goal=raw["goal"], status=raw.get("status", "active"),
        channel=raw.get("channel", "text"),
        plan=[PlanItem(p["subgoal"], p.get("status", "todo")) for p in raw.get("plan", [])],
        steps=[TaskStep(s["tool"], s.get("args", ""), s.get("result", ""), bool(s.get("ok", True)), now - 30)
               for s in raw.get("steps", [])],
        created_at=now - 120, updated_at=now - 30,
    )


# ── Scoring ─────────────────────────────────────────────────────────────

def score(intent: Intent, expect: Dict[str, Any]) -> List[str]:
    """Every failed check of one case ([] = pass). Only the checks the case specifies are scored."""
    problems: List[str] = []
    if intent.kind != expect["kind"]:
        problems.append(f"kind {intent.kind!r} != {expect['kind']!r}")
    for flag in ("continues_task", "single_step"):
        if flag in expect and getattr(intent, flag) != expect[flag]:
            problems.append(f"{flag} {getattr(intent, flag)} != {expect[flag]}")
    request = intent.request.lower()
    for item in expect.get("request_contains", []):
        alts = _alternatives(item)
        if not any(a.lower() in request for a in alts):
            problems.append(f"request lacks {' / '.join(alts)!r}")
    for item in expect.get("tool_groups_include", []):
        alts = _alternatives(item)
        if not any(a in intent.tool_groups for a in alts):
            problems.append(f"tool_groups lacks {' / '.join(alts)!r}")
    return problems


def expected_output(expect: Dict[str, Any]) -> Dict[str, Any]:
    """A model reply that satisfies ``expect`` (first alternative of every item)."""
    parts = [_alternatives(item)[0] for item in expect.get("request_contains", [])]
    return {
        "request": " ".join(parts) or "ok",
        "kind": expect["kind"],
        "continues_task": expect.get("continues_task", False),
        "single_step": expect.get("single_step", False),
        "tool_groups": [_alternatives(item)[0] for item in expect.get("tool_groups_include", [])],
        "needs_screen": False,
    }


def _intent_json(i: Intent) -> str:
    return json.dumps({"request": i.request, "kind": i.kind, "continues_task": i.continues_task,
                       "single_step": i.single_step, "tool_groups": i.tool_groups,
                       "needs_screen": i.needs_screen, "source": i.source}, ensure_ascii=False)


# ── Offline: schema + harness plumbing ──────────────────────────────────

def test_cases_are_well_formed():
    cases = load_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    assert len(cases) >= 20
    for c in cases:
        assert c["message"].strip(), c["id"]
        assert c.get("source") in ("real", "paraphrase"), c["id"]
        expect = c["expect"]
        assert set(expect) <= _EXPECT_KEYS, (c["id"], set(expect) - _EXPECT_KEYS)
        assert expect["kind"] in KINDS, c["id"]
        for item in expect.get("tool_groups_include", []):
            for g in _alternatives(item):
                assert g in TOOL_GROUPS, (c["id"], g)
        for t in c.get("turns") or []:
            assert t["user"] and t["assistant"], c["id"]
        build_task(c)  # task layout parses
    assert sum(c["source"] == "paraphrase" for c in cases) >= 8


def test_required_real_messages_present():
    messages = {c["message"] for c in load_cases() if c["source"] == "real"}
    for m in ["next do", "i said in the previous message right", "not windows but in the chrome browser tabs",
              "yes open", "yes take screen shots",
              "not in the chrome , i have app in my system , open that and message nithesh",
              "scroll down", "ok good", "no need for now", "how do I restart my router?",
              "look at the recent chats I had with you today", "open youtube for me and go to mr beast channel"]:
        assert m in messages, m


async def test_eval_cases_offline(monkeypatch):
    """FakeLLM answers each case with its expected JSON: every case must parse and score as a pass."""
    monkeypatch.setattr(settings, "INTENT_MODEL", LIVE_INTENT_MODEL)
    monkeypatch.setattr(settings, "INTENT_FALLBACK_MODEL", LIVE_FALLBACK_MODEL)
    for case in load_cases():
        fake = FakeLLM([text(json.dumps(expected_output(case["expect"])))])
        monkeypatch.setattr(intent_mod, "llm", fake)
        got = await analyze(case["message"], build_turns(case), build_task(case))
        assert got.source == "llm", case["id"]
        assert score(got, case["expect"]) == [], (case["id"], score(got, case["expect"]))
        prompt = fake.calls[0]["messages"][-1]["content"]
        assert f"New message: {case['message'].strip()}" in prompt, case["id"]
        for t in case.get("turns") or []:
            assert t["user"].split()[0] in prompt, case["id"]
        if case.get("task"):
            assert "Current task:\nTask:" in prompt, case["id"]


def test_score_catches_each_kind_of_miss():
    expect = {"kind": "action", "continues_task": True, "single_step": False,
              "request_contains": ["Nithesh", ["3rd", "third"]], "tool_groups_include": [["web", "keyboard"]]}
    good = Intent("Message Nithesh about the third tab", "action", True, False, ["keyboard"])
    assert score(good, expect) == []
    assert score(Intent(good.request, "chat", True, False, ["keyboard"]), expect)
    assert score(Intent(good.request, "action", False, False, ["keyboard"]), expect)
    assert score(Intent(good.request, "action", True, True, ["keyboard"]), expect)
    assert score(Intent("Message Nithesh about a tab", "action", True, False, ["keyboard"]), expect)
    assert score(Intent(good.request, "action", True, False, ["messaging"]), expect)
    assert score(Intent("anything", "chat", False, True, []), {"kind": "chat"}) == []


# ── Live ────────────────────────────────────────────────────────────────

class _Recorder:
    """Wraps the real gateway: records model, latency, token usage and errors of each intent call."""

    def __init__(self, real: Any, secrets: Sequence[str]):
        self.real, self.secrets, self.calls = real, [s for s in secrets if s], []

    def _scrub(self, msg: str) -> str:
        for s in self.secrets:
            msg = msg.replace(s, "***")
        return msg[:160]

    async def chat(self, messages, **kw):
        rec: Dict[str, Any] = {"model": kw.get("model"), "at": time.monotonic(), "ok": False, "error": "cancelled"}
        self.calls.append(rec)
        started = time.perf_counter()
        try:
            resp = await self.real.chat(messages, **kw)
            rec.update(ok=True, error="", usage=resp.usage)
            return resp
        except Exception as exc:
            rec["error"] = self._scrub(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            rec["ms"] = round((time.perf_counter() - started) * 1000)


def _live_keys() -> Tuple[str, List[str]]:
    from dotenv import dotenv_values

    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        pytest.skip(f"{env_file} not found")
    env = dotenv_values(env_file)
    groq = (env.get("GROQ_API_KEY") or "").strip()
    gemini = [k.strip() for k in [env.get("GEMINI_API_KEY") or "", *(env.get("GEMINI_API_KEYS") or "").split(",")]
              if k.strip()]
    if not groq:
        pytest.skip("GROQ_API_KEY missing in .env")
    return groq, gemini


async def _pace(recorder: _Recorder, budget: int = _TOKENS_PER_MINUTE) -> None:
    """Wait until the last minute's primary-model token use leaves room for one more call."""
    primary = [c for c in recorder.calls if c["model"] == LIVE_INTENT_MODEL and c.get("usage")]
    est = (primary[-1]["usage"].get("total_tokens", 0) if primary else 0) or 2500
    while True:
        now = time.monotonic()
        used = sum(c["usage"].get("total_tokens", 0) for c in primary if now - c["at"] < 60)
        if used + est <= budget:
            return
        oldest = min(c["at"] for c in primary if now - c["at"] < 60)
        await asyncio.sleep(max(0.5, 60 - (now - oldest) + 0.5))


@pytest.mark.network
async def test_eval_cases_live(monkeypatch):
    from core import connectivity

    groq, gemini = _live_keys()
    monkeypatch.setattr(settings, "GROQ_API_KEY", groq)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", gemini[0] if gemini else "")
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", gemini[1:])
    monkeypatch.setattr(settings, "INTENT_MODEL", LIVE_INTENT_MODEL)
    monkeypatch.setattr(settings, "INTENT_FALLBACK_MODEL", LIVE_FALLBACK_MODEL)
    connectivity.set_state_for_tests(True)
    recorder = _Recorder(intent_mod.llm, [groq, *gemini])
    monkeypatch.setattr(intent_mod, "llm", recorder)

    cases = load_cases()
    rows: List[Dict[str, Any]] = []
    rate_limited = 0
    try:
        for case in cases:
            for attempt in range(_RATE_LIMIT_RETRIES + 1):
                await _pace(recorder)
                first = len(recorder.calls)
                started = time.perf_counter()
                got = await analyze(case["message"], build_turns(case), build_task(case))
                ms = round((time.perf_counter() - started) * 1000)
                calls = recorder.calls[first:]
                primary_err = next((c["error"] for c in calls if c["model"] == LIVE_INTENT_MODEL and c["error"]), "")
                if "429" in primary_err or "rate" in primary_err.lower():
                    rate_limited += 1
                    if attempt < _RATE_LIMIT_RETRIES:
                        print(f"  [rate-limited] {case['id']}: waiting {_RATE_LIMIT_WAIT_S:.0f}s and retrying")
                        await asyncio.sleep(_RATE_LIMIT_WAIT_S)
                        continue
                break
            answered = next((c["model"] for c in reversed(calls) if c["ok"]), "legacy")
            rows.append({"id": case["id"], "source": case["source"], "intent": got, "ms": ms,
                         "answered_by": answered if got.source == "llm" else "legacy",
                         "primary_error": primary_err, "problems": score(got, case["expect"])})
    finally:
        connectivity.set_state_for_tests(None)

    passed = [r for r in rows if not r["problems"]]
    accuracy = len(passed) / len(rows)
    print(f"\n=== intent eval: {len(passed)}/{len(rows)} = {accuracy:.1%} (target {TARGET:.0%}) ===")
    for r in rows:
        mark = "PASS" if not r["problems"] else "FAIL"
        print(f"{mark} {r['id']:<38} {r['ms']:>5} ms  {r['answered_by']}")
    for kind in ("real", "paraphrase"):
        sub = [r for r in rows if r["source"] == kind]
        print(f"  {kind}: {sum(not r['problems'] for r in sub)}/{len(sub)}")
    failing = [r for r in rows if r["problems"]]
    if failing:
        print("--- failing cases ---")
        for r in failing:
            print(f"{r['id']}: {'; '.join(r['problems'])}\n    actual: {_intent_json(r['intent'])}")
            if r["primary_error"]:
                print(f"    primary error: {r['primary_error']}")
    lat = [r["ms"] for r in rows if r["answered_by"] == LIVE_INTENT_MODEL]
    if lat:
        print(f"latency ({LIVE_INTENT_MODEL}, whole analyze): min {min(lat)} / median "
              f"{statistics.median(lat):.0f} / max {max(lat)} ms over {len(lat)} cases")
    others = [r["id"] for r in rows if r["answered_by"] != LIVE_INTENT_MODEL]
    print(f"answered by fallback/legacy: {others or 'none'}; rate-limited attempts: {rate_limited}")
    print("--- all outputs ---")
    for r in rows:
        print(f"{r['id']}: {_intent_json(r['intent'])}")
    assert accuracy >= TARGET, f"accuracy {accuracy:.1%} < {TARGET:.0%}; failing: {[r['id'] for r in failing]}"
