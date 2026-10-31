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
    """A stand-in for the Langfuse v3 client: records every span/generation it is asked to make."""
    log = []

    class Obs:
        def __init__(self, kind, name, kw, parent=None):
            self.kind, self.name, self.kw, self.updates, self.ended = kind, name, kw, [], False
            log.append(self)

        def start_span(self, name, **kw):
            return Obs("span", name, kw)

        def start_generation(self, name, **kw):
            return Obs("generation", name, kw)

        def update(self, **kw):
            self.updates.append(kw)
            return self

        def end(self, **kw):
            self.ended = True

    class Langfuse:
        instances = []

        def __init__(self, **kw):
            self.kw = kw
            Langfuse.instances.append(self)

        def start_span(self, name, **kw):
            return Obs("span", name, kw)

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

    def guarded(name, *a, **kw):
        if name.split(".")[0] == "langfuse":
            raise AssertionError("langfuse must not be imported when tracing is off")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    res = await brain.handle("whatsapp kholo", "u-trace-off")
    assert res.status == "ok"


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
    assert Langfuse.instances[0].kw["host"] == "http://langfuse.invalid"
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
