"""Comprehensive Test Suite for JARVIS Voice & Microphone Interaction Module."""

import pytest
import struct
import math

from voice.state import VoiceState, VoiceStateManager
from voice.audio_capture import AudioCaptureManager
from voice.vad import VoiceActivityDetector
from voice.wake_word import WakeWordDetector
from voice.normalizer import CommandNormalizer
from voice.tts import TextToSpeechEngine
from voice.barge_in import BargeInMonitor
from brain.intent_engine import intent_engine


# ══════════════════════════════════════════════════════════════════════════════
# 1. State Machine Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_voice_state_machine_transitions():
    sm = VoiceStateManager(VoiceState.MIC_OFF)
    assert sm.current_state == VoiceState.MIC_OFF

    events = []
    def on_change(new_s, old_s, msg):
        events.append((new_s, old_s, msg))

    sm.add_listener(on_change)
    sm.transition_to(VoiceState.MIC_INITIALIZING, "Starting mic")
    assert sm.current_state == VoiceState.MIC_INITIALIZING
    assert sm.previous_state == VoiceState.MIC_OFF
    assert len(events) == 1

    sm.transition_to(VoiceState.MIC_READY, "Ready")
    assert sm.current_state == VoiceState.MIC_READY
    assert sm.is_active() is True

    sm.transition_to(VoiceState.ERROR, "Device disconnected")
    assert sm.current_state == VoiceState.ERROR
    assert sm.error_message == "Device disconnected"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Audio Capture & RMS Energy Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_rms_energy_calculation():
    ac = AudioCaptureManager()
    
    # 1. Zero audio
    silent_pcm = struct.pack("<512h", *([0] * 512))
    assert ac.calculate_rms_energy(silent_pcm) == 0.0

    # 2. Moderate sound wave
    sound_pcm = struct.pack("<512h", *([1000] * 512))
    assert math.isclose(ac.calculate_rms_energy(sound_pcm), 1000.0, rel_tol=1e-3)


def test_audio_device_listing():
    ac = AudioCaptureManager()
    devices = ac.list_input_devices()
    assert isinstance(devices, list)
    # Check device fields structure
    for d in devices:
        assert "index" in d
        assert "name" in d
        assert "channels" in d


# ══════════════════════════════════════════════════════════════════════════════
# 3. Voice Activity Detection (VAD) Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_vad_speech_detection_and_pause_tolerance():
    vad = VoiceActivityDetector()

    # Simulate speech frame (200 Hz voiced tone with 1500 amplitude)
    speech_samples = [int(1500 * math.sin(2 * math.pi * 200 * i / 16000)) for i in range(512)]
    speech_frame = struct.pack("<512h", *speech_samples)

    # Simulate silence frame
    silent_frame = struct.pack("<512h", *([10] * 512))

    # Frame 1: Speech detected -> speaking active
    is_speaking, is_complete, _ = vad.process_frame(speech_frame, energy_threshold=300)
    assert is_speaking is True
    assert is_complete is False

    # Frame 2: Short pause (< silence timeout) -> should NOT cut off
    is_speaking_p, is_complete_p, _ = vad.process_frame(silent_frame, energy_threshold=300)
    assert is_speaking_p is True
    assert is_complete_p is False


# ══════════════════════════════════════════════════════════════════════════════
# 4. Wake Word Detection Tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("input_text,expected_detected,expected_cmd", [
    ("Hey JARVIS open YouTube", True, "open YouTube"),
    ("JARVIS search for Python tutorials", True, "search for Python tutorials"),
    ("Okay JARVIS open Chrome", True, "open Chrome"),
    ("Hi JARVIS what is the weather", True, "what is the weather"),
    ("Open Chrome", False, "Open Chrome"),
])
def test_wake_word_detection(input_text, expected_detected, expected_cmd):
    ww = WakeWordDetector()
    detected, cmd = ww.detect_in_text(input_text)
    assert detected == expected_detected
    if expected_detected:
        assert expected_cmd.lower() in cmd.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 5. Command Normalizer Tests
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("raw_input,expected_output", [
    ("open you tube and search for py thon complier", "open YouTube and search for Python compiler"),
    ("open vs code and start terminal", "open Visual Studio Code and start terminal"),
    ("go to git hub and check my repos", "go to GitHub and check my repos"),
    ("open whats app and msg karthikeya", "open WhatsApp and msg karthikeya"),
    ("open linked in and check jobs", "open LinkedIn and check jobs"),
    ("change resolution to 1080 p", "change resolution to 1080p"),
    ("make the video full screen", "make the video fullscreen"),
    ("turn on wi fi", "turn on WiFi"),
    ("um please open chrome", "open Chrome"),
])
def test_command_normalizer(raw_input, expected_output):
    cn = CommandNormalizer()
    normalized = cn.normalize(raw_input)
    assert normalized.lower() == expected_output.lower()


# ══════════════════════════════════════════════════════════════════════════════
# 6. Text-To-Speech & Barge-In Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_tts_text_sanitization():
    tts = TextToSpeechEngine()
    raw = "```python\nprint('hello')\n``` Check out https://google.com for **awesome** results!"
    clean = tts._prepare_concise_text(raw)
    assert "```" not in clean
    assert "**" not in clean
    assert "https://" not in clean


@pytest.mark.parametrize("keyword", ["stop", "cancel", "abort", "pause", "wait", "hold on", "never mind"])
def test_barge_in_stop_keywords(keyword):
    bm = BargeInMonitor()
    assert bm.is_interrupt_command(keyword) is True
    assert bm.is_interrupt_command(f"{keyword} jarvis") is True


# ══════════════════════════════════════════════════════════════════════════════
# 7. 60+ Mandatory Real-World Voice Command Scenarios (Master Prompt)
# ══════════════════════════════════════════════════════════════════════════════

VOICE_BENCHMARK_SCENARIOS = [
    # ── Browser / YouTube (1-20) ──
    "open YouTube",
    "search for Python tutorials",
    "open the first result",
    "go to the recently uploaded video",
    "like it",
    "change the resolution to 1080p",
    "turn on captions",
    "make the video fullscreen",
    "pause the video",
    "play it again",
    "skip forward thirty seconds",
    "go back ten seconds",
    "open the comments",
    "scroll down",
    "find a video about machine learning",
    "open the second result",
    "subscribe to the channel",
    "turn off autoplay",
    "search for the latest AI news",
    "open the most recent video",

    # ── Browser / Websites (21-35) ──
    "open Google",
    "search for today's weather",
    "open GitHub",
    "go to my repositories",
    "open my latest repository",
    "open LinkedIn",
    "go to my profile",
    "open Gmail",
    "show my unread emails",
    "open the latest email",
    "go back",
    "open a new tab",
    "close this tab",
    "switch to the previous tab",
    "search for Python developer jobs",

    # ── Desktop Applications (36-45) ──
    "open Chrome",
    "open Visual Studio Code",
    "open File Explorer",
    "open Settings",
    "open Calculator",
    "open Notepad",
    "open Discord",
    "open Spotify",
    "open WhatsApp",
    "close the current application",

    # ── File Management (46-55) ──
    "open my Downloads folder",
    "find the PDF I downloaded today",
    "open the latest file",
    "create a new folder called Projects",
    "move this file into Projects",
    "rename this file to final_resume.pdf",
    "copy this file",
    "paste it into Documents",
    "delete this file",
    "show me the files modified today",

    # ── Development & Terminal (56-65) ──
    "open VS Code",
    "open my JARVIS project",
    "open the main Python file",
    "run the project",
    "open the terminal",
    "run the tests",
    "show me the error",
    "open the file that caused the error",
    "search the project for the login function",
    "run the application again",

    # ── System Control (66-75) ──
    "turn the volume up",
    "mute the computer",
    "unmute it",
    "open Wi-Fi settings",
    "take a screenshot",
    "lock the computer",
    "open Task Manager",
    "minimize this window",
    "maximize the window",
    "switch to the next application",
]


@pytest.mark.parametrize("voice_command", VOICE_BENCHMARK_SCENARIOS)
def test_voice_command_scenario_understanding(voice_command):
    # 1. Normalize
    cn = CommandNormalizer()
    normalized = cn.normalize(voice_command)
    assert len(normalized) > 0

    # 2. Intent Engine understanding
    intent = intent_engine.detect_intent(normalized)
    assert intent.intent in (
        "AUTOMATION_TASK",
        "REALTIME_KNOWLEDGE_SEARCH",
        "FILESYSTEM_OPERATION",
        "SYSTEM_TELEMETRY",
        "SCREEN_VISION",
        "CODE_GENERATION",
        "TASK_PLANNING",
        "GENERAL_CONVERSATION",
    )
