"""Configuration Settings for JARVIS Voice & Microphone Module."""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.config import ENV_FILE


class VoiceConfig(BaseSettings):
    """Configuration schema for Audio capture, VAD, Wake-Word, STT, and TTS."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
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
    SILENCE_TIMEOUT_MS: int = 900  # pause that ends a command (lower = snappier, higher = more patient)
    MAX_COMMAND_DURATION_SEC: float = 25.0
    PRE_SPEECH_BUFFER_MS: int = 400
    POST_SPEECH_BUFFER_MS: int = 400

    # ── Wake Word Engine ──
    WAKE_WORD_ENABLED: bool = True
    WAKE_WORD_ENGINE: str = "text"  # text (match in transcript) | openwakeword (acoustic, low latency, private)
    OPENWAKEWORD_MODEL: str = "hey_jarvis"
    OPENWAKEWORD_THRESHOLD: float = 0.5
    WAKE_WORDS: List[str] = ["hey jarvis", "jarvis", "okay jarvis", "hi jarvis", "ok jarvis"]
    WAKE_WORD_SENSITIVITY: float = 0.65
    REQUIRE_WAKE_WORD_FOR_FOLLOWUPS: bool = False
    FOLLOWUP_WINDOW_SECONDS: float = 15.0  # Window to accept follow-up commands without wake word

    # ── Speech-To-Text (STT) ──
    STT_ENGINE: str = "auto"  # auto (faster_whisper if installed, else google) | faster_whisper | google
    WHISPER_MODEL: str = "base.en"  # tiny.en | base.en | small.en | distil-large-v3 | large-v3-turbo
    WHISPER_DEVICE: str = "auto"  # auto | cpu | cuda
    WHISPER_COMPUTE_TYPE: str = "auto"  # auto → int8 on CPU, float16 on GPU
    WHISPER_LANGUAGE: str = "en"
    STT_LANGUAGE: str = "en-IN"  # Primary language code with accent resilience
    STT_FALLBACK_LANGUAGE: str = "en-US"
    STT_TIMEOUT_SECONDS: float = 10.0

    # ── Text-To-Speech (TTS) ──
    TTS_ENABLED: bool = True
    TTS_ENGINE: str = "auto"  # auto (kokoro if model files exist, else pyttsx3) | kokoro | pyttsx3
    KOKORO_MODEL_PATH: str = "data/models/kokoro-v1.0.onnx"
    KOKORO_VOICES_PATH: str = "data/models/voices-v1.0.bin"
    KOKORO_VOICE: str = "am_michael"  # e.g. am_michael, bm_george, af_heart
    KOKORO_SPEED: float = 1.1
    STREAM_TTS: bool = True  # start speaking at the first complete sentence of the reply
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
    DEBUG_AUDIO_LOGGING: bool = False


voice_config = VoiceConfig()
