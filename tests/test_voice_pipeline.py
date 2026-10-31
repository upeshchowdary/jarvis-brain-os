"""Voice pipeline: streamed sentence-by-sentence speech, spoken confirmations, STT engine chain."""

from unittest.mock import MagicMock

import numpy as np

from automation.gateway import gateway
from tests.fakes import FakeLLM, install, text
import importlib

from voice.voice_manager import confirmation_speech, speakable, split_sentences, voice_manager

vm_module = importlib.import_module("voice.voice_manager")


def test_split_sentences_streaming():
    done, rest = split_sentences("Hello there friend. How are")
    assert done == ["Hello there friend."] and rest.strip() == "How are"
    done, rest = split_sentences("Hi. Okay so the answer is 42! ")
    assert done == ["Hi. Okay so the answer is 42!"] and rest == ""


def test_speakable_strips_markdown():
    assert speakable("**Bold** `code` [link](http://x) # Title") == "Bold code link Title"


def test_confirmation_speech_reads_preview():
    s = confirmation_speech({"pending": {"preview": "• Restart this PC in 30 seconds"}})
    assert s.startswith("I need your confirmation.") and "Restart this PC" in s and "Say yes" in s


async def test_respond_speaks_each_sentence(monkeypatch):
    spoken = []
    monkeypatch.setattr(vm_module, "tts_engine", MagicMock(speak=lambda t, **k: spoken.append(t), stop=lambda: None))
    install(monkeypatch, FakeLLM([text("Paris is the capital of France. It is famous for the Eiffel Tower.")]))
    res = await voice_manager.respond("what is the capital of france")
    assert res["success"] and len(spoken) >= 2
    assert spoken[0].startswith("Paris is the capital")


async def test_voice_confirmation_flow(monkeypatch):
    import automation.ops.system as sysops

    monkeypatch.setattr(sysops, "_IS_WINDOWS", True)
    restart = MagicMock(return_value=True)
    monkeypatch.setattr(sysops.impl, "restart_system", restart)
    gateway.resume()
    spoken = []
    monkeypatch.setattr(vm_module, "tts_engine", MagicMock(speak=lambda t, **k: spoken.append(t), stop=lambda: None))
    voice_manager.session_id = "voice-confirm-test"
    first = await voice_manager.respond("restart the computer")
    assert first["status"] == "needs_confirmation" and "Say yes to confirm" in spoken[-1]
    restart.assert_not_called()
    second = await voice_manager.respond("yes")
    assert second["status"] == "ok"
    restart.assert_called_once()


async def test_voice_loop_does_not_starve_event_loop(monkeypatch):
    """A live mic must not freeze the API server sharing the same event loop."""
    import asyncio
    import threading

    from voice.audio_capture import AudioCaptureManager

    mic = AudioCaptureManager()
    silence = b"\x00\x00" * 512
    mic._push_frame(silence)
    monkeypatch.setattr(vm_module, "audio_capture", mic)
    monkeypatch.setattr(vm_module, "tts_engine", MagicMock(is_speaking=False))
    monkeypatch.setattr(voice_manager, "_is_running", True)
    # Safety net: if the loop starves the event loop, only another thread can end it.
    starved = threading.Event()

    def stop_from_thread():
        starved.set()
        voice_manager._is_running = False

    guard = threading.Timer(1.5, stop_from_thread)
    guard.start()

    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(10):
            mic._push_frame(silence)  # a live mic keeps delivering frames
            await asyncio.sleep(0.01)
            ticks += 1
        voice_manager._is_running = False

    try:
        await asyncio.gather(voice_manager._voice_loop(), ticker())
    finally:
        guard.cancel()
    assert not starved.is_set() and ticks == 10


async def test_stt_falls_back_to_google_and_filters_hallucinations(monkeypatch):
    from voice import stt as stt_mod

    engine = stt_mod.SpeechToTextEngine()
    whisper_model = MagicMock()
    whisper_model.transcribe.return_value = ([MagicMock(text=" Thank you.")], None)
    monkeypatch.setattr(engine.whisper, "_load", lambda: whisper_model)
    monkeypatch.setattr(engine.whisper, "available", lambda: True)
    monkeypatch.setattr(engine.google, "available", lambda: True)
    monkeypatch.setattr(engine.google, "transcribe", lambda raw, sr: "open chrome")
    monkeypatch.setattr(stt_mod.voice_config, "STT_ENGINE", "auto")
    audio = (np.sin(np.linspace(0, 200, 16000)) * 8000).astype("<i2").tobytes()
    res = await engine.transcribe(audio)
    # Whisper's "Thank you." on noise is discarded, Google's result is used.
    assert res["success"] and res["engine"] == "google" and res["text"].lower() == "open chrome"


async def test_stt_uses_whisper_when_it_hears_words(monkeypatch):
    from voice import stt as stt_mod

    engine = stt_mod.SpeechToTextEngine()
    model = MagicMock()
    model.transcribe.return_value = ([MagicMock(text=" Lock the screen")], None)
    monkeypatch.setattr(engine.whisper, "_load", lambda: model)
    monkeypatch.setattr(engine.whisper, "available", lambda: True)
    monkeypatch.setattr(stt_mod.voice_config, "STT_ENGINE", "faster_whisper")
    res = await engine.transcribe(b"\x01\x00" * 8000)
    assert res["engine"] == "faster_whisper" and res["text"] == "Lock the screen"


def test_silence_timeout_is_600():
    from voice.voice_config import voice_config

    assert voice_config.SILENCE_TIMEOUT_MS == 600


def _patch_stream(monkeypatch, events):
    import brain.pipeline as bp

    async def fake_stream(*a, **k):
        for e in events:
            yield e

    monkeypatch.setattr(bp.brain, "stream", fake_stream)


class _Result:
    def __init__(self, response, status="ok"):
        self._d = {"response": response, "status": status, "intent": {"intent": "x"}, "metadata": {}}

    def to_dict(self):
        return self._d


async def test_progress_events_are_spoken(monkeypatch):
    spoken = []
    monkeypatch.setattr(vm_module, "tts_engine", MagicMock(speak=lambda t, **k: spoken.append(t), stop=lambda: None))
    _patch_stream(monkeypatch, [{"type": "progress", "text": "Opening WhatsApp."},
                                {"type": "progress", "text": "Message sent."},
                                {"type": "done", "result": _Result("All done, sir.")}])
    await voice_manager.respond("message mom")
    assert spoken[:2] == ["Opening WhatsApp.", "Message sent."]
    assert spoken[2] == "All done, sir." and len(spoken) == 3


async def test_progress_duplicate_of_final_response_is_heard_once(monkeypatch):
    spoken = []
    monkeypatch.setattr(vm_module, "tts_engine", MagicMock(speak=lambda t, **k: spoken.append(t), stop=lambda: None))
    _patch_stream(monkeypatch, [{"type": "progress", "text": "Message sent."},
                                {"type": "progress", "text": "Message sent."},
                                {"type": "done", "result": _Result("Message sent.")}])
    await voice_manager.respond("message mom")
    assert spoken == ["Message sent."]


async def _run(monkeypatch, events):
    spoken = []
    monkeypatch.setattr(vm_module, "tts_engine", MagicMock(speak=lambda t, **k: spoken.append(t), stop=lambda: None))
    _patch_stream(monkeypatch, events)
    await voice_manager.respond("do it")
    return spoken


async def test_streamed_sentence_then_matching_progress_heard_once(monkeypatch):
    spoken = await _run(monkeypatch, [{"type": "text", "delta": "Opening WhatsApp. "},
                                      {"type": "progress", "text": "Opening WhatsApp."},
                                      {"type": "done", "result": _Result("Opening WhatsApp.")}])
    assert spoken == ["Opening WhatsApp."]


async def test_final_response_repeats_only_new_sentences(monkeypatch):
    spoken = await _run(monkeypatch, [{"type": "progress", "text": "Message sent."},
                                      {"type": "done", "result": _Result("Message sent. Anything else?")}])
    assert spoken == ["Message sent.", "Anything else?"]


async def test_progress_longer_than_six_words_is_skipped(monkeypatch):
    spoken = await _run(monkeypatch, [{"type": "progress", "text": "I have now opened the chat with your mother."},
                                      {"type": "progress", "text": "Chat is open now."},
                                      {"type": "done", "result": _Result("Finished.")}])
    assert spoken == ["Chat is open now.", "Finished."]
