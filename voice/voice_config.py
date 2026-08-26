"""Configuration Settings for JARVIS Voice & Microphone Module."""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceConfig(BaseSettings):
    """Configuration schema for Audio capture, VAD, Wake-Word, STT, and TTS."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Audio Device & Hardware ──
    INPUT_DEVICE_INDEX: Optional[int] = None
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_SIZE: int = 1024
    ENERGY_THRESHOLD: int = 300
    DYNAMIC_ENERGY_THRESHOLD: bool = True
    NOISE_CALIBRATION_DURATION: float = 1.0

    # ── Voice Activity Detection (VAD) ──
    VAD_MODE: int = Field(default=2, ge=0, le=3)  # Aggressiveness 0-3
    MIN_SPEECH_DURATION_MS: int = 300
    SILENCE_TIMEOUT_MS: int = 1400  # Pause duration to consider speech complete (prevents cutting off)
    MAX_COMMAND_DURATION_SEC: float = 25.0
    PRE_SPEECH_BUFFER_MS: int = 400
    POST_SPEECH_BUFFER_MS: int = 400

    # ── Wake Word Engine ──
    WAKE_WORD_ENABLED: bool = True
    WAKE_WORDS: List[str] = ["hey jarvis", "jarvis", "okay jarvis", "hi jarvis", "ok jarvis"]
    WAKE_WORD_SENSITIVITY: float = 0.65
    REQUIRE_WAKE_WORD_FOR_FOLLOWUPS: bool = False
    FOLLOWUP_WINDOW_SECONDS: float = 15.0  # Window to accept follow-up commands without wake word

    # ── Speech-To-Text (STT) ──
    STT_ENGINE: str = "google"  # google | whisper | gemini | offline
    STT_LANGUAGE: str = "en-IN"  # Primary language code with accent resilience
    STT_FALLBACK_LANGUAGE: str = "en-US"
    STT_TIMEOUT_SECONDS: float = 10.0

    # ── Text-To-Speech (TTS) ──
    TTS_ENABLED: bool = True
    TTS_ENGINE: str = "pyttsx3"  # pyttsx3 (SAPI5) | edge_tts | gtts
    TTS_VOICE_NAME: str = "David"  # David / Zira / Desktop voices
    TTS_RATE_WPM: int = 185
    TTS_VOLUME: float = Field(default=1.0, ge=0.0, le=1.0)
    CONCISE_SPEECH_MODE: bool = True  # Keeps verbal responses concise and natural

    # ── Interruption & Barge-In ──
    ENABLE_BARGE_IN: bool = True
    INTERRUPT_KEYWORDS: List[str] = ["stop", "cancel", "abort", "pause", "wait", "hold on", "never mind", "shut up"]

    # ── Operation Modes ──
    PUSH_TO_TALK_ENABLED: bool = False
    PUSH_TO_TALK_KEY: str = "space"
    HANDS_FREE_ENABLED: bool = True
    DEBUG_AUDIO_LOGGING: bool = True


voice_config = VoiceConfig()
