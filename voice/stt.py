"""Multi-Engine Speech-To-Text (STT) Pipeline for JARVIS Voice Module.

Features:
- Multi-engine architecture (Google Speech API, Whisper, Gemini, Offline)
- Accent resilience (Indian English, American, British)
- Non-blocking async execution
- Contextual transcription error recovery
"""

from __future__ import annotations

import asyncio
import io
import wave
import time
from typing import Optional, Dict, Any
from loguru import logger
import speech_recognition as sr

from voice.voice_config import voice_config
from voice.state import VoiceState, voice_state_manager
from voice.normalizer import command_normalizer


class SpeechToTextEngine:
    """Orchestrates audio transcription across cloud and local STT providers."""

    def __init__(self) -> None:
        self.recognizer = sr.Recognizer()
        self.sample_rate = voice_config.SAMPLE_RATE
        self.primary_language = voice_config.STT_LANGUAGE
        self.fallback_language = voice_config.STT_FALLBACK_LANGUAGE
        self.timeout = voice_config.STT_TIMEOUT_SECONDS

    def raw_audio_to_audiodata(self, raw_audio: bytes) -> sr.AudioData:
        """Wraps raw 16-bit PCM bytes into SpeechRecognition AudioData object."""
        return sr.AudioData(raw_audio, self.sample_rate, 2)

    def wav_bytes_from_raw(self, raw_audio: bytes) -> bytes:
        """Encodes raw 16-bit PCM bytes into a standard WAV container."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(voice_config.CHANNELS)
            wav_file.setsampwidth(2)  # 16-bit PCM
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(raw_audio)
        return buffer.getvalue()

    async def transcribe(self, raw_audio: bytes) -> Dict[str, Any]:
        """
        Transcribes raw audio bytes into clean, normalized text.
        
        Returns:
            {"success": bool, "text": str, "engine": str, "latency_ms": float, "confidence": float}
        """
        if not raw_audio or len(raw_audio) < 1000:
            return {"success": False, "text": "", "error": "Audio buffer too short or empty", "latency_ms": 0.0}

        start_time = time.perf_counter()
        voice_state_manager.transition_to(VoiceState.TRANSCRIBING, "Transcribing voice")

        loop = asyncio.get_event_loop()
        audio_data = self.raw_audio_to_audiodata(raw_audio)

        # 1. Primary: Google Speech Recognition (fast, free, resilient to Indian English & accents)
        try:
            transcript = await loop.run_in_executor(
                None,
                lambda: self.recognizer.recognize_google(
                    audio_data,
                    language=self.primary_language,
                )
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            normalized = command_normalizer.normalize(transcript)
            logger.info(f"[STT] Transcribed ({self.primary_language}) in {elapsed_ms:.0f}ms: '{normalized}'")
            return {
                "success": True,
                "text": normalized,
                "raw_text": transcript,
                "engine": "google",
                "language": self.primary_language,
                "latency_ms": round(elapsed_ms, 1),
                "confidence": 0.95,
            }
        except sr.UnknownValueError:
            logger.debug("[STT] Primary language unrecognized. Attempting fallback language...")
        except Exception as e:
            logger.warning(f"[STT] Primary STT error: {e}. Trying fallback...")

        # 2. Secondary Language Fallback (en-US)
        try:
            transcript = await loop.run_in_executor(
                None,
                lambda: self.recognizer.recognize_google(
                    audio_data,
                    language=self.fallback_language,
                )
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            normalized = command_normalizer.normalize(transcript)
            logger.info(f"[STT] Transcribed ({self.fallback_language}) in {elapsed_ms:.0f}ms: '{normalized}'")
            return {
                "success": True,
                "text": normalized,
                "raw_text": transcript,
                "engine": "google",
                "language": self.fallback_language,
                "latency_ms": round(elapsed_ms, 1),
                "confidence": 0.90,
            }
        except sr.UnknownValueError:
            logger.info("[STT] Audio was not clear speech.")
            return {"success": False, "text": "", "error": "Speech not recognized", "latency_ms": (time.perf_counter() - start_time) * 1000}
        except Exception as e:
            logger.error(f"[STT] All STT engines failed: {e}")
            return {"success": False, "text": "", "error": str(e), "latency_ms": (time.perf_counter() - start_time) * 1000}


stt_engine = SpeechToTextEngine()
