# JARVIS Vision System Architecture & Integration Guide

## 1. Executive Summary

The **JARVIS Vision System** provides visual perception, desktop screen capture, OCR, UI element detection, and Vision LLM reasoning capabilities to the JARVIS AI Assistant.

Vision functions as a first-class tool integrated directly into the JARVIS Brain's cognitive decision pipeline without altering existing Memory, Brain Managers, or Provider abstractions.

---

## 2. Architectural Overview

```
                      USER QUERY
                          │
                          ▼
                   ┌──────────────┐
                   │ JARVIS BRAIN │ (BrainManager)
                   └──────┬───────┘
                          │
                   Intent Classification (IntentEngine)
                          │
       Is intent SCREEN_VISION / Visual query?
                 ┌────────┴────────┐
                YES                NO
                 │                 │
                 ▼                 ▼
         ┌──────────────┐   Normal Text Reasoning
         │ VISION TOOL  │   & Memory Retrieval
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │VISION MANAGER│ (VisionManager)
         └──────┬───────┘
                │
  ┌─────────────┼──────────────┬───────────────┐
  │             │              │               │
  ▼             ▼              ▼               ▼
Screen         OCR          UI & Spatial     Vision LLM
Capture       Engine         Detection       Providers
 (MSS)       (Tesseract/   (Bounding Boxes  (Gemini /
              EasyOCR)     [x1,y1,x2,y2])    Ollama)
  │             │              │               │
  └─────────────┴──────────────┴───────────────┘
                          │
                          ▼
            Structured Vision Result (EnvironmentState / JSON)
                          │
                          ▼
                   ┌──────────────┐
                   │ JARVIS BRAIN │ -> Synthesizes final response
                   └──────────────┘    with visual perception context
```

---

## 3. Package Structure

The vision module is organized under `vision/`:

- `vision/capture/`: Desktop, monitor, and region capture via MSS / Pillow.
- `vision/ocr/`: Text extraction engine supporting Tesseract and EasyOCR.
- `vision/analysis/`: UI element detection, object detection, and layout analysis.
- `vision/providers/`: Vision provider abstractions (`base.py`, `gemini.py`, `ollama_vision.py`, `qwen.py`, `llava.py`, `openai.py`).
- `vision/schemas.py`: Pydantic structured output models (`EnvironmentState`, `UIElement`, `OCRTextItem`, `StructuredVisionResult`).
- `vision/manager.py`: Central orchestrator (`VisionManager`).
- `tools/vision_tool.py`: `ScreenVisionTool` registered in JARVIS `ToolRouter`.

---

## 4. Configuration & Supported Models

### Environment Variables (.env)
```env
# Google Gemini Vision
GEMINI_API_KEY=your_gemini_api_key_here

# Local Ollama Multimodal Vision
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3.5:latest
```

### Supported Vision LLM Models
1. **Gemini Vision**: `gemini-2.0-flash`, `gemini-1.5-pro`
2. **Local Ollama Vision**: `qwen3.5:latest`, `qwen2.5-vl`, `llava`, `bakllava`, `moondream`, `llama3.2-vision`
3. **OpenAI Vision**: `gpt-4o-mini`, `gpt-4o`

---

## 5. Vision Execution Modes & Performance Fast-Paths

- **`full` mode**: Takes screenshot, extracts OCR text, identifies UI element bounding boxes, and queries a Vision LLM for narrative scene description.
- **`ocr_only` mode**: Skips expensive LLM calls when the user asks simple text questions (e.g., *"Read the text on screen"*). Runs screenshot → OCR.
- **`ui_only` mode**: Extracts UI controls and spatial bounding box coordinates `[x1, y1, x2, y2]`.

### Screenshot Hash Caching & Privacy
- Screenshots are processed in memory and never saved permanently to disk.
- Hash caching in `screen_memory` avoids re-analyzing identical static frames.
- Image downscaling and JPEG compression minimize network payload and API latency.

---

## 6. How Brain Integrates Vision

When a user query like *"What is on my screen?"* or *"Read the error code"* is received:
1. `IntentEngine.detect_intent(query)` returns `intent="SCREEN_VISION"`.
2. `BrainManager` routes the call to `ToolRouter.route_and_execute(ToolCallSpec(tool="screen_vision", arguments={"query": query}))`.
3. `ScreenVisionTool` calls `VisionManager.capture_environment_state()`.
4. The structured visual context is injected into the Brain's prompt, allowing JARVIS to reason and reply with precision.

---

## 7. Future Desktop Automation Compatibility

Every detected UI element contains normalized pixel bounding box coordinates:

```json
{
  "id": "btn_submit",
  "element_type": "button",
  "label": "Submit",
  "bounds": {
    "x": 820,
    "y": 540,
    "width": 90,
    "height": 40
  },
  "confidence": 0.96
}
```

Future automation modules will consume `bounds.center` (`x=865, y=560`) to perform mouse movement and keyboard click actions.
