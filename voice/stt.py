"""Speech-to-text with a local-first engine chain.

* **faster-whisper** (MIT, CTranslate2) — runs locally on CPU (int8) or GPU (float16);
  no audio leaves the PC and latency is ~0.2–0.6 s for a short command with ``base.en``.
  The model is loaded once in the background so the first command isn't slow.
* **Google Web Speech** (via SpeechRecognition) — free cloud fallback, no key needed.

``STT_ENGINE=auto`` uses faster-whisper when it is installed and falls back to Google.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
import wave
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from voice.normalizer import command_normalizer
from voice.state import VoiceState, voice_state_manager
from voice.voice_config import voice_config

try:
    import speech_recognition as sr
except ImportError:  # optional [voice] dependency
    sr = None

# Phrases Whisper hallucinates on silence/noise — never treat them as commands.
_HALLUCINATIONS = {"thank you.", "thanks for watching!", "you", "thank you", "bye.", ".", "okay.", "subtitles by the amara.org community"}


class FasterWhisperEngine:
    name = "faster_whisper"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None

    @staticmethod
    def available() -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def _load(self):
        with self._lock:
            if self._model is None and self._error is None:
                try:
                    from faster_whisper import WhisperModel

                    device = voice_config.WHISPER_DEVICE
                    if device == "auto":
                        try:
                            import ctranslate2

                            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                        except Exception:
                            device = "cpu"
                    compute = voice_config.WHISPER_COMPUTE_TYPE
                    if compute == "auto":
                        compute = "float16" if device == "cuda" else "int8"
                    started = time.perf_counter()
                    self._model = WhisperModel(voice_config.WHISPER_MODEL, device=device, compute_type=compute)
                    logger.info(f"[STT] faster-whisper '{voice_config.WHISPER_MODEL}' on {device}/{compute} "
                                f"loaded in {time.perf_counter() - started:.1f}s")
                except Exception as exc:
                    self._error = str(exc)
                    logger.warning(f"[STT] faster-whisper unavailable: {exc}")
            return self._model

    def warmup(self) -> None:
        threading.Thread(target=self._load, daemon=True, name="JARVIS-WhisperLoad").start()

    def transcribe(self, raw_audio: bytes, sample_rate: int) -> str:
        model = self._load()
        if model is None:
            raise RuntimeError(self._error or "faster-whisper failed to load")
        audio = np.frombuffer(raw_audio, dtype="<i2").astype(np.float32) / 32768.0
        if sample_rate != 16000:  # whisper expects 16 kHz
            idx = np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / sample_rate)).astype(np.int64)
            audio = audio[idx]
        segments, _info = model.transcribe(
            audio,
            language=voice_config.WHISPER_LANGUAGE or None,
            beam_size=1,  # greedy = fastest; commands are short
            vad_filter=False,  # we already segmented the utterance
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return "" if text.lower() in _HALLUCINATIONS else text


class GoogleEngine:
    name = "google"

    def __init__(self) -> None:
        self.recognizer = sr.Recognizer() if sr else None

    @staticmethod
    def available() -> bool:
        return sr is not None

    def transcribe(self, raw_audio: bytes, sample_rate: int) -> str:
        if self.recognizer is None:
            raise RuntimeError("SpeechRecognition is not installed")
        audio = sr.AudioData(raw_audio, sample_rate, 2)
        for lang in (voice_config.STT_LANGUAGE, voice_config.STT_FALLBACK_LANGUAGE):
            try:
                return self.recognizer.recognize_google(audio, language=lang)
            except sr.UnknownValueError:
                continue
        return ""


class SpeechToTextEngine:
    """Transcribes raw 16-bit PCM utterances, local engine first."""

    def __init__(self) -> None:
        self.sample_rate = voice_config.SAMPLE_RATE
        self.whisper = FasterWhisperEngine()
        self.google = GoogleEngine()

    def engines(self):
        choice = voice_config.STT_ENGINE.lower()
        if choice == "google":
            return [self.google]
        if choice in ("faster_whisper", "whisper"):
            return [self.whisper, self.google]
        return [e for e in (self.whisper, self.google) if e.available()]

    def warmup(self) -> None:
        if self.whisper in self.engines() and self.whisper.available():
            self.whisper.warmup()

    def wav_bytes_from_raw(self, raw_audio: bytes) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(voice_config.CHANNELS)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(raw_audio)
        return buffer.getvalue()

    async def transcribe(self, raw_audio: bytes) -> Dict[str, Any]:
        if not raw_audio or len(raw_audio) < 1000:
            return {"success": False, "text": "", "error": "Audio buffer too short or empty", "latency_ms": 0.0}
        started = time.perf_counter()
        voice_state_manager.transition_to(VoiceState.TRANSCRIBING, "Transcribing voice")
        errors = []
        for engine in self.engines():
            try:
                transcript = await asyncio.to_thread(engine.transcribe, raw_audio, self.sample_rate)
            except Exception as exc:
                errors.append(f"{engine.name}: {exc}")
                continue
            if transcript:
                elapsed = (time.perf_counter() - started) * 1000
                normalized = command_normalizer.normalize(transcript)
                logger.info(f"[STT] {engine.name} {elapsed:.0f} ms: '{normalized}'")
                return {"success": True, "text": normalized, "raw_text": transcript, "engine": engine.name,
                        "latency_ms": round(elapsed, 1), "confidence": 0.9}
        elapsed = (time.perf_counter() - started) * 1000
        return {"success": False, "text": "", "error": "; ".join(errors) or "Speech not recognized", "latency_ms": round(elapsed, 1)}


stt_engine = SpeechToTextEngine()
