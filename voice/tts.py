"""Interruptible Text-To-Speech (TTS) Engine for JARVIS Voice Module.

Features:
- Interruptible speech (immediate cancellation on user interruption / barge-in)
- Non-blocking queue-based speech worker thread
- Concise natural response phrasing
- Windows SAPI5 / pyttsx3 native engine with voice, rate, and volume controls
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Optional, List, Dict
from loguru import logger
try:
    import pyttsx3
except ImportError:  # optional [voice] dependency
    pyttsx3 = None

from voice.voice_config import voice_config
from voice.state import VoiceState, voice_state_manager


class TextToSpeechEngine:
    """Manages audio speech synthesis with instant cancellation capabilities."""

    def __init__(self) -> None:
        self.enabled = voice_config.TTS_ENABLED
        self.rate = voice_config.TTS_RATE_WPM
        self.volume = voice_config.TTS_VOLUME
        self.preferred_voice = voice_config.TTS_VOICE_NAME

        self._queue: queue.Queue = queue.Queue()
        self._is_speaking = False
        self._interrupted = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_worker = threading.Event()
        self._lock = threading.Lock()
        self.backend_name = "none"

    def _start_worker(self) -> None:
        """Starts dedicated background TTS synthesis thread (lazily, on first speech)."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_worker.clear()
        self._worker_thread = threading.Thread(target=self._speech_worker, daemon=True, name="JARVIS-TTS")
        self._worker_thread.start()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    def list_voices(self) -> List[Dict[str, str]]:
        """List all installed TTS voices."""
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            return [{"id": v.id, "name": v.name, "languages": getattr(v, "languages", [])} for v in voices]
        except Exception:
            return []

    def set_voice(self, voice_name_or_id: str) -> bool:
        """Change TTS voice."""
        self.preferred_voice = voice_name_or_id
        return True

    def set_rate(self, rate_wpm: int) -> None:
        """Set speech speed in words per minute."""
        self.rate = max(100, min(300, rate_wpm))

    def set_volume(self, volume: float) -> None:
        """Set speech volume (0.0 to 1.0)."""
        self.volume = max(0.0, min(1.0, volume))

    def speak(self, text: str, priority: bool = False, concise: bool = True) -> None:
        """
        Enqueues text to be spoken.
        If priority is True, cancels any currently playing speech immediately.
        """
        if not self.enabled or not text.strip():
            return

        cleaned_text = self._prepare_concise_text(text, concise=concise)
        if not cleaned_text:
            return
        if priority:
            self.stop()

        self._start_worker()
        self._queue.put(cleaned_text)

    async def speak_async(self, text: str, wait_for_completion: bool = True) -> None:
        """Async convenience method to speak and optionally await completion."""
        if not self.enabled or not text.strip():
            return

        cleaned_text = self._prepare_concise_text(text)
        self.stop()
        self._start_worker()
        self._queue.put(cleaned_text)

        if wait_for_completion:
            # Wait until speech completes or is interrupted
            while self._is_speaking or not self._queue.empty():
                await asyncio.sleep(0.05)

    def stop(self) -> None:
        """Immediately interrupts and halts all active and queued speech."""
        with self._lock:
            self._interrupted.set()
            # Clear remaining queue
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            if self._is_speaking:
                logger.info("[TTS] Speech interrupted (barge-in triggered).")
                voice_state_manager.transition_to(VoiceState.INTERRUPTED, "TTS interrupted")

    def _prepare_concise_text(self, text: str, concise: bool = True) -> str:
        """Keeps spoken responses natural, brief, and free of markdown symbols."""
        import re
        clean = text.strip()
        # Remove code blocks
        clean = re.sub(r"```[\s\S]*?```", "Code block omitted.", clean)
        # Remove markdown headers and asterisks
        clean = re.sub(r"[#*_`~>\[\]\(\)]", "", clean)
        # Remove URLs
        clean = re.sub(r"https?://\S+", "the website", clean)
        # Condense extra spaces
        clean = re.sub(r"\s+", " ", clean).strip()

        # If concise mode is active, trim long paragraphs
        if concise and voice_config.CONCISE_SPEECH_MODE and len(clean) > 280:
            sentences = re.split(r"(?<=[.!?]) +", clean)
            if sentences:
                clean = " ".join(sentences[:2])

        return clean

    # ── Backends ────────────────────────────────────────────────────────
    def _make_kokoro(self):
        """Kokoro-82M neural TTS (kokoro-onnx: MIT code, Apache-2.0 weights). Needs the model files."""
        from pathlib import Path

        from core.paths import BASE_DIR

        model = Path(voice_config.KOKORO_MODEL_PATH)
        voices = Path(voice_config.KOKORO_VOICES_PATH)
        model = model if model.is_absolute() else BASE_DIR / model
        voices = voices if voices.is_absolute() else BASE_DIR / voices
        if not (model.exists() and voices.exists()):
            raise RuntimeError(f"Kokoro model files not found ({model.name}, {voices.name}); see README → Voice")
        from kokoro_onnx import Kokoro  # type: ignore
        import sounddevice as sd  # type: ignore

        kokoro = Kokoro(str(model), str(voices))

        def speak(sentence: str) -> None:
            samples, rate = kokoro.create(sentence, voice=voice_config.KOKORO_VOICE, speed=voice_config.KOKORO_SPEED, lang="en-us")
            sd.play(samples, rate)
            # Poll so barge-in can cut the audio instantly.
            duration = len(samples) / float(rate)
            end = time.monotonic() + duration
            while time.monotonic() < end:
                if self._interrupted.is_set():
                    sd.stop()
                    return
                time.sleep(0.03)
            sd.wait()

        return speak

    def _make_pyttsx3(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 is not installed (pip install .[voice])")
        engine = pyttsx3.init()
        try:
            for v in engine.getProperty("voices"):
                if self.preferred_voice.lower() in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass

        def speak(sentence: str) -> None:
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            engine.say(sentence)
            engine.runAndWait()

        return speak

    def _make_backend(self):
        choice = voice_config.TTS_ENGINE.lower()
        order = {"kokoro": ["kokoro", "pyttsx3"], "pyttsx3": ["pyttsx3"]}.get(choice, ["kokoro", "pyttsx3"])
        for name in order:
            try:
                backend = self._make_kokoro() if name == "kokoro" else self._make_pyttsx3()
                self.backend_name = name
                logger.info(f"[TTS] Using {name} voice")
                return backend
            except Exception as exc:
                logger.info(f"[TTS] {name} unavailable: {exc}")
        return None

    def _speech_worker(self) -> None:
        """Background thread: speaks queued text sentence by sentence (interruptible)."""
        import re

        speak = self._make_backend()
        if speak is None:
            logger.error("[TTS] No speech backend available.")
            return

        while not self._stop_worker.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._interrupted.clear()
            self._is_speaking = True
            voice_state_manager.transition_to(VoiceState.SPEAKING, f"Speaking: '{text[:40]}...'")
            try:
                for sentence in re.split(r"(?<=[.!?;:]) +", text):
                    if self._interrupted.is_set():
                        break
                    if sentence.strip():
                        speak(sentence.strip())
            except Exception as e:
                logger.warning(f"[TTS] Speech synthesis warning: {e}")
            finally:
                self._is_speaking = not self._queue.empty() and not self._interrupted.is_set()
                if not self._is_speaking and not self._interrupted.is_set() and voice_state_manager.current_state == VoiceState.SPEAKING:
                    voice_state_manager.transition_to(VoiceState.MIC_READY, "Speech completed")


tts_engine = TextToSpeechEngine()
