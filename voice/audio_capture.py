"""Microphone Stream Capture and Hardware Device Management for JARVIS.

Features:
- Device discovery & dynamic selection
- Non-blocking continuous streaming buffer
- Automatic background noise calibration & RMS energy calculation
- Ring buffer for pre-speech capture
- Fault-tolerant auto-recovery on device disconnection/glitches
"""

from __future__ import annotations

import collections

import numpy as np
import threading
import time
from typing import Dict, List, Optional, Any
from loguru import logger

from voice.state import VoiceState, voice_state_manager
from voice.voice_config import voice_config


class AudioCaptureManager:
    """Manages audio input streaming and hardware device health."""

    def __init__(self) -> None:
        self.sample_rate = voice_config.SAMPLE_RATE
        self.channels = voice_config.CHANNELS
        self.chunk_size = voice_config.CHUNK_SIZE
        self.selected_device_index = voice_config.INPUT_DEVICE_INDEX

        self._is_streaming = False
        self._stream = None
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Ring buffer for continuous audio frames (keeps last ~1.5 seconds)
        max_buffer_frames = int((self.sample_rate / self.chunk_size) * 2.0)
        self._ring_buffer: collections.deque = collections.deque(maxlen=max_buffer_frames)

        # Dynamic energy calibration
        self.energy_threshold = float(voice_config.ENERGY_THRESHOLD)
        self.current_energy = 0.0
        self.is_calibrated = False

        # Status & Telemetry
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()

    def list_input_devices(self) -> List[Dict[str, Any]]:
        """List all audio input devices detected on the OS."""
        devices = []
        try:
            import sounddevice as sd
            dev_list = sd.query_devices()
            for idx, dev in enumerate(dev_list):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({
                        "index": idx,
                        "name": dev.get("name", f"Device {idx}"),
                        "channels": dev.get("max_input_channels", 1),
                        "default_samplerate": dev.get("default_samplerate", 16000),
                        "is_default": (idx == sd.default.device[0]),
                    })
        except Exception as e:
            logger.warning(f"[AudioCapture] Failed to query sounddevice devices: {e}")
            try:
                import pyaudio
                p = pyaudio.PyAudio()
                for i in range(p.get_device_count()):
                    dev_info = p.get_device_info_by_index(i)
                    if dev_info.get("maxInputChannels", 0) > 0:
                        devices.append({
                            "index": i,
                            "name": dev_info.get("name", f"Device {i}"),
                            "channels": dev_info.get("maxInputChannels", 1),
                            "default_samplerate": dev_info.get("defaultSampleRate", 16000),
                            "is_default": False,
                        })
                p.terminate()
            except Exception as e2:
                logger.error(f"[AudioCapture] Failed to query pyaudio devices: {e2}")

        return devices

    def set_input_device(self, device_index: int) -> bool:
        """Select a specific microphone device by index."""
        devices = self.list_input_devices()
        matched = any(d["index"] == device_index for d in devices)
        if not matched and device_index is not None:
            logger.warning(f"[AudioCapture] Device index {device_index} not found in available input devices.")
            return False

        self.selected_device_index = device_index
        if self._is_streaming:
            self.restart_stream()
        return True

    def calculate_rms_energy(self, raw_audio_chunk: bytes) -> float:
        """Calculate Root Mean Square (RMS) energy level of an audio frame."""
        if not raw_audio_chunk:
            return 0.0
        try:
            samples = np.frombuffer(raw_audio_chunk[: len(raw_audio_chunk) // 2 * 2], dtype="<i2").astype(np.float64)
            return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
        except Exception:
            return 0.0

    def calibrate_ambient_noise(self, duration_sec: float = 1.0) -> float:
        """Calibrate energy threshold based on ambient background noise."""
        logger.info(f"[AudioCapture] Calibrating ambient noise for {duration_sec}s...")
        energies = []
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_sec:
            chunk = self.read_chunk(timeout=0.2)
            if chunk:
                energy = self.calculate_rms_energy(chunk)
                if energy > 0:
                    energies.append(energy)
            time.sleep(0.02)

        if energies:
            avg_energy = sum(energies) / len(energies)
            # Threshold set to 1.5x of ambient noise + baseline margin
            self.energy_threshold = max(200.0, avg_energy * 1.6)
            self.is_calibrated = True
            logger.info(f"[AudioCapture] Noise calibrated. Ambient avg: {avg_energy:.1f}, Energy threshold: {self.energy_threshold:.1f}")
        else:
            self.energy_threshold = float(voice_config.ENERGY_THRESHOLD)
            self.is_calibrated = True

        return self.energy_threshold

    def start_stream(self) -> bool:
        """Start non-blocking continuous microphone recording stream."""
        if self._is_streaming:
            return True

        voice_state_manager.transition_to(VoiceState.MIC_INITIALIZING, "Starting audio stream")
        self._stop_event.clear()

        try:
            self._stream_thread = threading.Thread(target=self._audio_stream_worker, daemon=True, name="JARVIS-AudioCapture")
            self._stream_thread.start()
            self._is_streaming = True
            voice_state_manager.transition_to(VoiceState.MIC_READY, "Audio stream active")
            logger.info("[AudioCapture] Microphone audio stream started successfully.")
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[AudioCapture] Failed to start audio stream: {e}")
            voice_state_manager.transition_to(VoiceState.ERROR, f"Microphone error: {e}")
            return False

    def stop_stream(self) -> None:
        """Stop microphone recording stream gracefully."""
        if not self._is_streaming:
            return

        self._stop_event.set()
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=1.0)

        self._is_streaming = False
        voice_state_manager.transition_to(VoiceState.MIC_OFF, "Audio stream stopped")
        logger.info("[AudioCapture] Microphone audio stream stopped.")

    def restart_stream(self) -> bool:
        """Gracefully restarts stream on device change or error recovery."""
        self.stop_stream()
        time.sleep(0.1)
        return self.start_stream()

    def read_chunk(self, timeout: float = 0.5) -> Optional[bytes]:
        """Reads latest audio chunk from ring buffer."""
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            with self._lock:
                if self._ring_buffer:
                    return self._ring_buffer[-1]
            time.sleep(0.01)
        return None

    def get_buffered_audio(self, duration_sec: float) -> bytes:
        """Get the last N seconds of raw audio bytes from the circular buffer."""
        frames_needed = int((self.sample_rate / self.chunk_size) * duration_sec)
        with self._lock:
            buffer_copy = list(self._ring_buffer)

        recent_frames = buffer_copy[-frames_needed:] if frames_needed <= len(buffer_copy) else buffer_copy
        return b"".join(recent_frames)

    def _audio_stream_worker(self) -> None:
        """Worker thread handling audio stream acquisition."""
        try:
            import sounddevice as sd
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                device=self.selected_device_index,
                channels=self.channels,
                dtype="int16",
            ) as stream:
                while not self._stop_event.is_set():
                    data, overflowed = stream.read(self.chunk_size)
                    if overflowed:
                        logger.debug("[AudioCapture] Stream overflow occurred.")

                    raw_bytes = bytes(data)
                    energy = self.calculate_rms_energy(raw_bytes)
                    self.current_energy = energy

                    with self._lock:
                        self._ring_buffer.append(raw_bytes)

        except Exception as e:
            logger.warning(f"[AudioCapture] sounddevice error: {e}. Trying pyaudio fallback.")
            self._audio_stream_pyaudio_fallback()

    def _audio_stream_pyaudio_fallback(self) -> None:
        """Fallback stream worker using PyAudio."""
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.selected_device_index,
                frames_per_buffer=self.chunk_size,
            )

            while not self._stop_event.is_set():
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                energy = self.calculate_rms_energy(data)
                self.current_energy = energy

                with self._lock:
                    self._ring_buffer.append(data)

            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"[AudioCapture] PyAudio stream error: {e}")
            voice_state_manager.transition_to(VoiceState.ERROR, f"Audio hardware failure: {e}")


audio_capture = AudioCaptureManager()
