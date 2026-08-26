"""JARVIS Voice & Microphone Interaction Subsystem.

Provides:
- Real-time continuous microphone streaming
- Voice Activity Detection (VAD)
- Multi-trigger Wake Word detection ("Hey JARVIS", "JARVIS", "Okay JARVIS", "Hi JARVIS")
- Multi-Engine Speech-To-Text (STT)
- Spoken Command Normalization & Phonetic Correction
- Interruptible Text-To-Speech (TTS) with Barge-In
- Seamless integration with JARVIS Cognitive Brain & Task Orchestrator
"""

from voice.state import VoiceState, voice_state_manager
from voice.voice_config import voice_config, VoiceConfig
from voice.audio_capture import audio_capture, AudioCaptureManager
from voice.vad import vad_detector, VoiceActivityDetector
from voice.wake_word import wake_word_detector, WakeWordDetector
from voice.normalizer import command_normalizer, CommandNormalizer
from voice.stt import stt_engine, SpeechToTextEngine
from voice.tts import tts_engine, TextToSpeechEngine
from voice.barge_in import barge_in_monitor, BargeInMonitor
from voice.voice_manager import voice_manager, VoiceManager

__all__ = [
    "VoiceState",
    "voice_state_manager",
    "voice_config",
    "VoiceConfig",
    "audio_capture",
    "AudioCaptureManager",
    "vad_detector",
    "VoiceActivityDetector",
    "wake_word_detector",
    "WakeWordDetector",
    "command_normalizer",
    "CommandNormalizer",
    "stt_engine",
    "SpeechToTextEngine",
    "tts_engine",
    "TextToSpeechEngine",
    "barge_in_monitor",
    "BargeInMonitor",
    "voice_manager",
    "VoiceManager",
]
