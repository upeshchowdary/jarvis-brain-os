"""Voice Subsystem REST Endpoints for JARVIS."""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from voice import (
    voice_manager,
    voice_state_manager,
    audio_capture,
    tts_engine,
    stt_engine,
    voice_config,
)

router = APIRouter(tags=["Voice"])


class DeviceSelectRequest(BaseModel):
    device_index: int = Field(..., description="Audio input device index")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text to speak")
    priority: bool = Field(default=False, description="Whether to interrupt active speech")


@router.get("/voice/status")
async def get_voice_status() -> Dict[str, Any]:
    """Get current status of Voice subsystem, microphone health, and active state."""
    return {
        "is_running": voice_manager.is_running,
        "state": voice_state_manager.current_state.value,
        "previous_state": voice_state_manager.previous_state.value,
        "state_duration_sec": round(voice_state_manager.state_duration, 2),
        "is_speaking": tts_engine.is_speaking,
        "selected_device_index": audio_capture.selected_device_index,
        "current_energy": round(audio_capture.current_energy, 1),
        "energy_threshold": round(audio_capture.energy_threshold, 1),
        "wake_word_enabled": voice_config.WAKE_WORD_ENABLED,
        "barge_in_enabled": voice_config.ENABLE_BARGE_IN,
        "error": voice_state_manager.error_message,
    }


@router.get("/voice/devices")
async def list_audio_devices() -> Dict[str, Any]:
    """List all detected audio input devices and active microphone."""
    devices = audio_capture.list_input_devices()
    return {
        "selected_device_index": audio_capture.selected_device_index,
        "total_devices": len(devices),
        "devices": devices,
    }


@router.post("/voice/device/select")
async def select_audio_device(request: DeviceSelectRequest) -> Dict[str, Any]:
    """Select active microphone device by index."""
    success = audio_capture.set_input_device(request.device_index)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to set input device {request.device_index}")
    return {"success": True, "selected_device_index": request.device_index}


@router.post("/voice/start")
async def start_voice_pipeline() -> Dict[str, Any]:
    """Start continuous microphone listening loop."""
    started = await voice_manager.start()
    return {"success": started, "state": voice_state_manager.current_state.value}


@router.post("/voice/stop")
async def stop_voice_pipeline() -> Dict[str, Any]:
    """Stop continuous microphone listening loop."""
    await voice_manager.stop()
    return {"success": True, "state": voice_state_manager.current_state.value}


@router.post("/voice/synthesize")
async def synthesize_speech(request: SynthesizeRequest) -> Dict[str, Any]:
    """Synthesize text into speech via TTS."""
    tts_engine.speak(request.text, priority=request.priority)
    return {"success": True, "text": request.text}


@router.post("/voice/interrupt")
async def interrupt_speech() -> Dict[str, Any]:
    """Immediately stop TTS output."""
    tts_engine.stop()
    return {"success": True, "action": "interrupted"}
