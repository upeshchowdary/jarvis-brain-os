"""JARVIS Interactive Voice Assistant — Hands-Free Desktop Console.

Usage:
    python voice_assistant.py

Features:
- Live hands-free continuous voice interaction
- Real-time VU audio meter & state indicator
- Wake word activation ("Hey JARVIS", "JARVIS", "Okay JARVIS")
- Barge-in voice interruption (say "Stop" to interrupt speech/automation)
- Connects directly to Cognitive Brain, Task Planner, and Automation Engine
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from loguru import logger

# Configure clean console logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

from voice.state import VoiceState, voice_state_manager
from voice.voice_manager import voice_manager
from voice.audio_capture import audio_capture
from voice.voice_config import voice_config
from app.database.connection import db_manager
from memory.long_term import long_term_memory
from memory.vector_memory import vector_memory


def draw_header() -> None:
    print("=" * 65)
    print("      🎙️  JARVIS AI OPERATING SYSTEM — VOICE ASSISTANT  🎙️")
    print("=" * 65)
    print("• Wake Words : 'Hey JARVIS', 'JARVIS', 'Okay JARVIS', 'Hi JARVIS'")
    print("• Interrupt  : Say 'Stop', 'Cancel', or 'Wait' to interrupt")
    print("• Status     : Continuous Listening (Hands-Free)")
    print("• Press Ctrl+C to stop.\n")


def on_state_change(new_state: VoiceState, old_state: VoiceState, message: str | None) -> None:
    icons = {
        VoiceState.MIC_OFF: "🔴 [MIC OFF]",
        VoiceState.MIC_INITIALIZING: "🟡 [INITIALIZING]",
        VoiceState.MIC_READY: "🟢 [LISTENING / STANDBY]",
        VoiceState.LISTENING: "👂 [HEARING SPEECH...]",
        VoiceState.WAKE_DETECTED: "⚡ [WAKE WORD DETECTED]",
        VoiceState.TRANSCRIBING: "📝 [TRANSCRIBING...]",
        VoiceState.UNDERSTANDING: "🧠 [THINKING / PLANNING...]",
        VoiceState.EXECUTING: "⚙️  [EXECUTING AUTOMATION...]",
        VoiceState.VERIFYING: "🔍 [VERIFYING ACTION...]",
        VoiceState.SPEAKING: "🗣️  [JARVIS SPEAKING...]",
        VoiceState.INTERRUPTED: "🛑 [INTERRUPTED]",
        VoiceState.ERROR: "⚠️ [VOICE ERROR]",
    }
    icon = icons.get(new_state, f"[{new_state.value}]")
    msg_str = f" : {message}" if message else ""
    print(f"\r{icon}{msg_str}{' ' * 20}")


async def main() -> None:
    draw_header()

    # 1. Initialize persistent memory and brain dependencies
    print("⚙️  Initializing Brain & Memory subsystems...")
    await db_manager.init_db()
    await long_term_memory.init_db()
    await vector_memory.load_from_db()

    # 2. Register State Change Callback
    voice_state_manager.add_listener(on_state_change)

    # 3. Start Voice Manager
    started = await voice_manager.start()
    if not started:
        print("❌ Failed to start Voice subsystem. Check microphone connections.")
        return

    print("✅ JARVIS Voice Engine Online! Speak now...\n")

    try:
        while True:
            # Idle heart-beat loop
            await asyncio.sleep(0.5)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n\n🛑 Stopping JARVIS Voice Assistant...")
        await voice_manager.stop()
        print("👋 Voice Assistant offline. Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
