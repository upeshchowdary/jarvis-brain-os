"""Optional Langfuse tracing: strict no-op when off, a trace with intent/LLM/tool spans when on."""

import builtins
import sys
import types

import pytest

import brain.pipeline as pipeline
from brain.intent import Intent
from brain.pipeline import brain
from core import tracing
from core.config import settings
from tests.fakes import FakeLLM, call, install, text
from tools.base import ToolResult
from tools.registry import load_builtin_tools, registry
from unittest.mock import AsyncMock



def _intent(kind="action"):
    return Intent(request="Launch WhatsApp from the Start menu", kind=kind, continues_task=False, single_step=True,
                  tool_groups=["system"], needs_screen=False, source="llm")


def _fake_langfuse(monkeypatch):
    """A stand-in mirroring the Langfuse 3.15 signatures (keyword-only, no start_span/start_generation on the
    client path): calling anything the real SDK lacks, or with a wrong keyword, raises TypeError/AttributeError."""
    log = []

    class Obs:
        __slots__ = ("kind", "name", "kw", "updates", "ended")

        def __init__(self, kind, name, kw):
            self.kind, self.name, self.kw, self.updates, self.ended = kind, name, kw, [], False
            log.append(self)

        def start_observation(self, *, name, as_type, input=None, output=None, metadata=None, model=None,
                              usage_details=None, cost_details=None, model_parameters=None, level=None,
                              status_message=None, version=None, completion_start_time=None, prompt=None):
            return Obs(as_type, name, dict(input=input, metadata=metadata, model=model, usage_details=usage_details))

        def update(self, *, name=None, input=None, output=None, metadata=None, model=None, usage_details=None,
                   level=None, status_message=None, **kwargs):
            self.updates.append(dict(output=output, metadata=metadata, usage_details=usage_details))
            return self

        def end(self, *, end_time=None):
            self.ended = True

    class Langfuse:
        instances = []

        def __init__(self, *, public_key=None, secret_key=None, base_url=None, host=None, timeout=None, **kw):
            self.kw = dict(public_key=public_key, secret_key=secret_key, base_url=base_url)
            Langfuse.instances.append(self)

        def start_observation(self, *, name, as_type="span", input=None, output=None, metadata=None, **kw):
            return Obs(as_type, name, dict(input=input, metadata=metadata))

    mod = types.ModuleType("langfuse")
    mod.Langfuse = Langfuse
    monkeypatch.setitem(sys.modules, "langfuse", mod)
    return log, Langfuse


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    tracing._reset()
    yield
    tracing._reset()


@pytest.fixture
def open_app(monkeypatch):
    load_builtin_tools()
    mock = AsyncMock(return_value=ToolResult.ok("Opened WhatsApp."))
    monkeypatch.setattr(registry.get("open_app"), "run", mock)
    monkeypatch.setattr(pipeline, "analyze", AsyncMock(return_value=_intent()))
    return mock


async def test_tracing_noop_when_disabled(monkeypatch, open_app):
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    real_import = builtins.__import__
    attempts = []

    def guarded(name, *a, **kw):
        if name.split(".")[0] == "langfuse":
            attempts.append(name)
            raise ImportError("blocked")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    res = await brain.handle("whatsapp kholo", "u-trace-off")
    assert res.status == "ok"
    assert attempts == [] and "langfuse" not in sys.modules
    assert tracing._client is None and not tracing._client_failed


async def test_tracing_records_when_enabled(monkeypatch, open_app):
    log, Langfuse = _fake_langfuse(monkeypatch)
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(settings, "LANGFUSE_HOST", "http://langfuse.invalid")
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    res = await brain.handle("whatsapp kholo", "u-trace-on")
    assert res.status == "ok"
    assert len(Langfuse.instances) == 1
    assert Langfuse.instances[0].kw["base_url"] == "http://langfuse.invalid"
    turns = [o for o in log if o.name == "turn"]
    assert len(turns) == 1 and turns[0].ended
    assert [o.name for o in log if o.name == "intent"] == ["intent"]
    assert any(o.kind == "generation" for o in log)
    assert any(o.name == "tool:open_app" for o in log)
    assert all("sk-test" not in repr(o.kw) + repr(o.updates) for o in log)


async def test_tracing_errors_never_break_a_turn(monkeypatch, open_app):
    mod = types.ModuleType("langfuse")

    class Boom:
        def __init__(self, **kw):
            raise RuntimeError("no network")

    mod.Langfuse = Boom
    monkeypatch.setitem(sys.modules, "langfuse", mod)
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    res = await brain.handle("whatsapp kholo", "u-trace-boom")
    assert res.status == "ok"


async def test_tracing_noop_when_package_missing(monkeypatch, open_app):
    monkeypatch.setitem(sys.modules, "langfuse", None)  # import raises ImportError
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    res = await brain.handle("whatsapp kholo", "u-trace-missing")
    assert res.status == "ok"
