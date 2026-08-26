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
from typing import Optional, List, Dict, Any
from loguru import logger
import pyttsx3

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

        self._start_worker()

    def _start_worker(self) -> None:
        """Starts dedicated background TTS synthesis thread."""
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

    def speak(self, text: str, priority: bool = False) -> None:
        """
        Enqueues text to be spoken.
        If priority is True, cancels any currently playing speech immediately.
        """
        if not self.enabled or not text.strip():
            return

        cleaned_text = self._prepare_concise_text(text)
        if priority:
            self.stop()

        self._queue.put(cleaned_text)

    async def speak_async(self, text: str, wait_for_completion: bool = True) -> None:
        """Async convenience method to speak and optionally await completion."""
        if not self.enabled or not text.strip():
            return

        cleaned_text = self._prepare_concise_text(text)
        self.stop()
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

    def _prepare_concise_text(self, text: str) -> str:
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
        if voice_config.CONCISE_SPEECH_MODE and len(clean) > 280:
            sentences = re.split(r"(?<=[.!?]) +", clean)
            if sentences:
                clean = " ".join(sentences[:2])

        return clean

    def _speech_worker(self) -> None:
        """Background thread executing speech via pyttsx3."""
        try:
            engine = pyttsx3.init()
        except Exception as e:
            logger.error(f"[TTS] Engine initialization failed: {e}")
            return

        # Setup voice
        try:
            voices = engine.getProperty("voices")
            for v in voices:
                if self.preferred_voice.lower() in v.name.lower():
                    engine.setProperty("voice", v.id)
                    break
        except Exception:
            pass

        while not self._stop_worker.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            self._interrupted.clear()
            self._is_speaking = True
            voice_state_manager.transition_to(VoiceState.SPEAKING, f"Speaking: '{text[:40]}...'")

            try:
                engine.setProperty("rate", self.rate)
                engine.setProperty("volume", self.volume)

                # Speak sentence by sentence so barge-in interruption is instant
                import re
                sentences = re.split(r"(?<=[.!?,;:]) +", text)

                for sentence in sentences:
                    if self._interrupted.is_set():
                        break
                    if sentence.strip():
                        engine.say(sentence.strip())
                        engine.runAndWait()

            except Exception as e:
                logger.warning(f"[TTS] Speech synthesis warning: {e}")
            finally:
                self._is_speaking = False
                if not self._interrupted.is_set() and voice_state_manager.current_state == VoiceState.SPEAKING:
                    voice_state_manager.transition_to(VoiceState.MIC_READY, "Speech completed")


tts_engine = TextToSpeechEngine()
