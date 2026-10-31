"""Hands-free voice loop: mic → (wake word) → VAD → STT → brain (streamed) → TTS.

Latency features:
* the reply is streamed from the LLM and each complete sentence is spoken as soon as
  it arrives (``STREAM_TTS``), so speech starts after the first sentence;
* local faster-whisper STT is pre-loaded at start;
* optional acoustic wake word (openWakeWord) skips transcription until "hey jarvis".

Safety: commands go through the same pipeline + ActionGateway as text. When an action
needs confirmation, JARVIS reads the preview aloud and the next "yes" (within the
follow-up window, no wake word needed) confirms it; "stop" interrupts speech and
triggers the emergency stop. A command runs as its own task while the loop keeps listening, so "stop"
also ends a multi-step task that is still working (other speech during a command is ignored).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Dict, Optional

from loguru import logger

from voice.audio_capture import audio_capture
from voice.barge_in import barge_in_monitor
from voice.state import VoiceState, voice_state_manager
from voice.stt import stt_engine
from voice.tts import tts_engine
from voice.vad import vad_detector
from voice.voice_config import voice_config
from voice.wake_word import acoustic_wake_word, wake_word_detector

_PROGRESS_MAX_WORDS = 6  # spoken progress lines stay short (spec 4.5)
_SENTENCE_END = re.compile(r"([.!?])(\s+|$)")


def split_sentences(buffer: str) -> tuple[list[str], str]:
    """Return (complete sentences, remainder) from a streaming text buffer."""
    sentences, start = [], 0
    for m in _SENTENCE_END.finditer(buffer):
        candidate = buffer[start:m.end()].strip()
        if len(candidate) >= 12 or m.end() == len(buffer):
            sentences.append(candidate)
            start = m.end()
    return sentences, buffer[start:]


def speakable(text: str) -> str:
    """Strip Markdown/code so it reads naturally."""
    text = re.sub(r"```[\s\S]*?```", " (code omitted) ", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[*_#>|~]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def confirmation_speech(result: Dict[str, Any]) -> str:
    pending = result.get("pending") or {}
    preview = speakable(pending.get("preview") or "")
    preview = re.sub(r"^[•\-\d. ]+", "", preview)
    return f"I need your confirmation. {preview} Say yes to confirm, or anything else to cancel."


class VoiceManager:
    def __init__(self) -> None:
        self.session_id: str = "voice_session_primary"
        self._is_running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._on_command_callbacks: list[Callable[[str], None]] = []
        self._awaiting_command = False
        self._command_task: Optional[asyncio.Task] = None  # the spoken command being handled, if any

    @property
    def is_running(self) -> bool:
        return self._is_running

    def register_command_listener(self, callback: Callable[[str], None]) -> None:
        self._on_command_callbacks.append(callback)

    async def start(self) -> bool:
        if self._is_running:
            return True
        logger.info("[VoiceManager] Starting voice system…")
        stt_engine.warmup()
        if not audio_capture.start_stream():
            logger.error("[VoiceManager] Failed to start audio capture stream.")
            return False
        audio_capture.calibrate_ambient_noise(duration_sec=voice_config.NOISE_CALIBRATION_DURATION)
        barge_in_monitor.start_monitoring()
        self._is_running = True
        self._loop_task = asyncio.create_task(self._voice_loop())
        voice_state_manager.transition_to(VoiceState.MIC_READY, "Voice system online & listening")
        return True

    async def stop(self) -> None:
        if not self._is_running:
            return
        self._is_running = False
        for task in (self._loop_task, self._command_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        barge_in_monitor.stop_monitoring()
        tts_engine.stop()
        audio_capture.stop_stream()
        voice_state_manager.transition_to(VoiceState.MIC_OFF, "Voice system offline")

    def _use_acoustic_wake(self) -> bool:
        return voice_config.WAKE_WORD_ENABLED and voice_config.WAKE_WORD_ENGINE == "openwakeword" and acoustic_wake_word.available()

    async def process_spoken_command(self, raw_audio: bytes, wake_already_detected: bool = False) -> Dict[str, Any]:
        stt_result = await stt_engine.transcribe(raw_audio)
        if not stt_result.get("success") or not stt_result.get("text"):
            voice_state_manager.transition_to(VoiceState.MIC_READY, "No speech recognized")
            return {"success": False, "error": "Unrecognized speech"}
        spoken = stt_result["text"]
        logger.info(f"[VoiceManager] Heard: '{spoken}'")

        if wake_already_detected:
            command = wake_word_detector.strip_wake_word(spoken)
            wake_word_detector.record_interaction()
        else:
            has_wake, command = wake_word_detector.detect_in_text(spoken)
            if voice_config.WAKE_WORD_ENABLED and not has_wake:
                voice_state_manager.transition_to(VoiceState.MIC_READY, "Awaiting wake word")
                return {"success": False, "ignored": True, "reason": "No wake word"}

        if not command.strip():
            tts_engine.speak("Yes?", priority=True)
            wake_word_detector.record_interaction()
            return {"success": True, "action": "wake_acknowledgment"}

        if barge_in_monitor.is_interrupt_command(command):
            barge_in_monitor.trigger_interruption("User voice stop command")
            tts_engine.speak("Stopped.", priority=True)
            return {"success": True, "action": "interrupted"}

        for cb in self._on_command_callbacks:
            try:
                cb(command)
            except Exception:
                pass
        return await self.respond(command)

    async def respond(self, command: str) -> Dict[str, Any]:
        """Run the command through the brain, speaking streamed sentences as they arrive."""
        from brain.pipeline import brain

        voice_state_manager.transition_to(VoiceState.UNDERSTANDING, f"Understanding: '{command}'")
        tts_engine.stop()
        buffer, spoken_any, result = "", False, None
        heard: set = set()  # normalised progress lines already spoken this turn (never repeat them)

        def norm(t: str) -> str:
            return re.sub(r"\W+", " ", t).strip().lower()

        def speak_unheard(text: str, **kw) -> None:
            """Speak a final text sentence by sentence, skipping sentences already heard this turn."""
            sentences, rest = split_sentences(text)
            for sentence in sentences + ([rest] if rest.strip() else []):
                line = speakable(sentence)
                if line and norm(line) not in heard:
                    heard.add(norm(line))
                    tts_engine.speak(line, **kw)

        async for event in brain.stream(command, self.session_id, channel="voice", stream_tokens=voice_config.STREAM_TTS):
            if event["type"] == "progress":
                line = speakable(event.get("text", ""))
                if line and len(line.split()) <= _PROGRESS_MAX_WORDS and norm(line) not in heard:
                    heard.add(norm(line))
                    tts_engine.speak(line, concise=False)  # queued, does not block the stream
            elif event["type"] == "text":
                buffer += event["delta"]
                sentences, buffer = split_sentences(buffer)
                for sentence in sentences:
                    line = speakable(sentence)
                    if norm(line) not in heard:
                        heard.add(norm(line))
                        tts_engine.speak(line, concise=False)
                    spoken_any = True
            elif event["type"] == "tool":
                voice_state_manager.transition_to(VoiceState.EXECUTING, event["name"])
            elif event["type"] == "done":
                result = event["result"].to_dict()
        wake_word_detector.record_interaction()
        if result is None:
            return {"success": False}
        if result["status"] == "needs_confirmation":
            tts_engine.speak(confirmation_speech(result), priority=True, concise=False)
        elif buffer.strip() and spoken_any:
            speak_unheard(buffer, concise=False)
        elif not spoken_any:
            final = speakable(result["response"])
            if not final:
                tts_engine.speak("Done.")
            else:
                speak_unheard(final)
        return {"success": True, "command": command, "intent": result["intent"]["intent"],
                "response": result["response"], "status": result["status"], "metadata": result["metadata"]}

    @staticmethod
    def is_stop_command(text: str) -> bool:
        """An interrupt word ("stop", "cancel", "wait"…) or one of the router's emergency-stop phrases."""
        from brain import router

        if barge_in_monitor.is_interrupt_command(text):
            return True
        decision = router.fast_route(text)
        return decision is not None and decision.rule == "stop"

    def _busy(self) -> bool:
        return self._command_task is not None and not self._command_task.done()

    async def _run_command(self, raw_audio: bytes, wake_already_detected: bool) -> None:
        try:
            await self.process_spoken_command(raw_audio, wake_already_detected=wake_already_detected)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"[VoiceManager] Command failed: {e}")
        finally:
            audio_capture.skip_to_live()  # audio captured during the turn is not replayed into the VAD
            vad_detector.reset()

    async def _stop_if_asked(self, raw_audio: bytes) -> None:
        """While a command runs, only a spoken stop is acted on (it ends the running task); the rest is ignored."""
        stt_result = await stt_engine.transcribe(raw_audio)
        text = (stt_result.get("text") or "").strip() if stt_result.get("success") else ""
        command = wake_word_detector.strip_wake_word(text) if text else ""
        if command and self.is_stop_command(command):
            barge_in_monitor.trigger_interruption("User voice stop command")  # emergency stop
            tts_engine.speak("Stopped.", priority=True)
        elif text:
            logger.info(f"[VoiceManager] Busy with a command; ignored: '{text}'")

    async def _voice_loop(self) -> None:
        acoustic = self._use_acoustic_wake()
        while self._is_running:
            try:
                if tts_engine.is_speaking:
                    audio_capture.skip_to_live()  # never feed JARVIS's own voice into the VAD
                    await asyncio.sleep(0.05)
                    continue
                # Never block here: this loop shares the event loop with the API server.
                chunk = audio_capture.read_chunk(timeout=0)
                if chunk is None:
                    await asyncio.sleep(0.02)
                    continue

                in_followup = wake_word_detector.is_in_followup_window()
                if acoustic and not self._awaiting_command and not in_followup:
                    if acoustic_wake_word.process(chunk):
                        self._awaiting_command = True
                        tts_engine.speak("Yes?", priority=True)
                    continue

                pre_buf = [audio_capture.get_buffered_audio(duration_sec=voice_config.PRE_SPEECH_BUFFER_MS / 1000)]
                _, is_complete, audio_data = vad_detector.process_frame(
                    raw_frame=chunk, energy_threshold=audio_capture.energy_threshold, pre_buffer=pre_buf
                )
                if is_complete and audio_data:
                    woke = self._awaiting_command or in_followup
                    self._awaiting_command = False
                    if self._busy():
                        await self._stop_if_asked(audio_data)
                    else:  # keep listening while it runs, so "stop" can end a long task
                        self._command_task = asyncio.create_task(
                            self._run_command(audio_data, acoustic and woke))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"[VoiceManager] Voice loop exception: {e}")
                await asyncio.sleep(0.1)


voice_manager = VoiceManager()
