"""Voice Activity Detection (VAD) Engine for JARVIS Voice Module.

Features:
- Dynamic energy & zero-crossing rate analysis
- Speech segment boundary detection (start of speech vs pause vs end of speech)
- Natural pause tolerance (does not cut off user mid-thought)
- Circular pre-speech and post-speech buffering
"""

from __future__ import annotations

import math
import struct
import time
from typing import List, Optional, Tuple
from loguru import logger

from voice.voice_config import voice_config
from voice.state import VoiceState, voice_state_manager


class VoiceActivityDetector:
    """Detects start and end of speech segments from continuous raw audio frames."""

    def __init__(self) -> None:
        self.sample_rate = voice_config.SAMPLE_RATE
        self.chunk_size = voice_config.CHUNK_SIZE
        self.min_speech_duration_ms = voice_config.MIN_SPEECH_DURATION_MS
        self.silence_timeout_ms = voice_config.SILENCE_TIMEOUT_MS
        self.max_command_duration_sec = voice_config.MAX_COMMAND_DURATION_SEC

        # Adaptive state
        self._speech_started = False
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
        self._speech_frames: List[bytes] = []

    def is_speech_frame(self, raw_audio_chunk: bytes, energy_threshold: float) -> bool:
        """Determines if a single chunk of audio contains human speech."""
        if not raw_audio_chunk:
            return False

        try:
            count = len(raw_audio_chunk) // 2
            format_str = f"<{count}h"
            shorts = struct.unpack(format_str, raw_audio_chunk)

            # 1. RMS Energy
            sum_squares = sum(s * s for s in shorts)
            rms = math.sqrt(sum_squares / count) if count > 0 else 0.0

            if rms < energy_threshold:
                return False

            # 2. Zero-Crossing Rate (ZCR) to differentiate voiced speech from static noise
            zcr = sum(1 for i in range(1, count) if (shorts[i] >= 0 > shorts[i - 1]) or (shorts[i] < 0 <= shorts[i - 1]))
            zcr_rate = zcr / count if count > 0 else 0.0

            # Human speech energy check
            return (rms >= energy_threshold) and (zcr_rate < 0.95)

        except Exception:
            return False

    def process_frame(
        self,
        raw_frame: bytes,
        energy_threshold: float,
        pre_buffer: Optional[List[bytes]] = None,
    ) -> Tuple[bool, bool, Optional[bytes]]:
        """
        Process a single audio frame.
        
        Returns:
            (is_speaking: bool, is_complete: bool, audio_data: Optional[bytes])
            - is_speaking: True if user is currently speaking or in temporary pause
            - is_complete: True when the user has finished speaking the complete command
            - audio_data: Raw byte stream of complete speech when is_complete is True
        """
        current_time = time.perf_counter()
        has_speech = self.is_speech_frame(raw_frame, energy_threshold)

        if has_speech:
            if not self._speech_started:
                # Speech just started
                self._speech_started = True
                self._speech_start_time = current_time
                self._last_speech_time = current_time
                self._speech_frames = []

                # Prepend pre-speech buffer so leading consonants aren't clipped
                if pre_buffer:
                    self._speech_frames.extend(pre_buffer)

                logger.debug("[VAD] User started speaking.")
                voice_state_manager.transition_to(VoiceState.LISTENING, "Speech detected")

            self._last_speech_time = current_time
            self._speech_frames.append(raw_frame)
            return True, False, None

        else:
            # Silence or background noise frame
            if self._speech_started:
                self._speech_frames.append(raw_frame)
                silence_duration_ms = (current_time - self._last_speech_time) * 1000
                total_duration_sec = current_time - self._speech_start_time

                # Check if maximum command duration exceeded
                if total_duration_sec >= self.max_command_duration_sec:
                    logger.info(f"[VAD] Max command duration ({self.max_command_duration_sec}s) reached.")
                    return self._finalize_speech()

                # Check if silence duration indicates completion of command
                if silence_duration_ms >= self.silence_timeout_ms:
                    speech_duration_ms = (self._last_speech_time - self._speech_start_time) * 1000
                    if speech_duration_ms >= self.min_speech_duration_ms:
                        logger.debug(f"[VAD] End of speech detected (silence: {silence_duration_ms:.0f}ms).")
                        return self._finalize_speech()
                    else:
                        # Spurious click or noise too short to be speech
                        self.reset()
                        return False, False, None

                return True, False, None

            return False, False, None

    def _finalize_speech(self) -> Tuple[bool, bool, Optional[bytes]]:
        """Packages accumulated speech frames into a complete audio byte buffer and resets state."""
        complete_audio = b"".join(self._speech_frames)
        self.reset()
        return False, True, complete_audio

    def reset(self) -> None:
        """Reset internal speech accumulation state."""
        self._speech_started = False
        self._speech_start_time = 0.0
        self._last_speech_time = 0.0
        self._speech_frames = []


vad_detector = VoiceActivityDetector()
