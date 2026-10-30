"""Intent Engine for classifying structured user intents and extracted parameters."""

import re
from typing import Dict, Any
from pydantic import BaseModel, Field
from brain.brain_config import brain_config


class StructuredIntent(BaseModel):
    intent: str = Field(..., description="High-level intent classification code")
    confidence: float = Field(default=0.95, ge=0.0, le=1.0)
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Extracted parameters or target arguments")
    summary: str = Field(..., description="Human readable summary of detected intent")


class IntentEngine:
    """Analyzes user inputs to determine structured intent and arguments."""

    @staticmethod
    def classify_visual_query_type(query: str) -> str:
        """Classify visual query into sub-categories."""
        q = query.lower().strip()
        if any(w in q for w in ["read text", "read the text", "visible text", "extract text", "read paragraph"]):
            return "OCR"
        if any(w in q for w in ["error", "exception", "traceback", "crash", "warning message", "stack trace"]):
            return "ERROR_ANALYSIS"
        if any(w in q for w in ["button", "submit", "checkbox", "menu", "where is", "click", "text box", "input field"]):
            return "UI_ANALYSIS"
        if any(w in q for w in ["image", "photo", "picture", "dog", "cat", "animal", "object", "item in picture"]):
            return "IMAGE_DESCRIPTION"
        if any(w in q for w in ["what animal", "what object", "who is in", "recognize object"]):
            return "OBJECT_RECOGNITION"
        if any(w in q for w in ["document", "pdf", "table", "chart", "graph"]):
            return "DOCUMENT_ANALYSIS"
        if any(w in q for w in ["what is on my screen", "describe my screen", "see my screen", "whats on screen", "entire screen"]):
            return "SCREEN_DESCRIPTION"
        return "VISUAL_QUESTION"

    @staticmethod
    def detect_intent(query: str) -> StructuredIntent:
        """Classify intent using heuristic patterns and entity extraction."""
        q_lower = query.lower().strip()

        # 0. Model Switch / Provider Change Intent (typo-tolerant)
        # Check for "ollama vision", "ollma vision", "qwen 2b", etc. first
        _ollama_vision_triggers = [
            "ollama vision", "ollama-vision", "ollama vison", "ollma vision",
            "switch vision", "use vision model", "use ollama vision", "switch to vision",
            "ollama qwen 2b", "ollma qwen 2b", "qwen 2b", "qwen2b", "ollama qwen",
            "switch to ollama vision", "switch to ollama qwen 2b", "switch to ollma vision",
            "switch to ollma qwen 2b", "switch ollama vision", "switch ollma vision",
        ]
        if any(t in q_lower for t in _ollama_vision_triggers) or (
            re.search(r"\b(?:switch|change|set|use|select)\b", q_lower)
            and (re.search(r"\bvision\b", q_lower) or re.search(r"\bqwen\s*2b\b", q_lower))
            and not re.search(r"\bgemini\b|\bgpt\b|\bgroq\b", q_lower)
        ):
            # Always map to the installed Ollama 2B vision model
            target_model = f"ollama/{brain_config.OLLAMA_VISION_MODEL}"
            return StructuredIntent(
                intent="MODEL_SWITCH",
                confidence=0.99,
                arguments={
                    "target_model": target_model,
                    "raw_query": query,
                },
                summary=f"User requested switching to Ollama vision model ({brain_config.OLLAMA_VISION_MODEL})",
            )

        switch_match = re.search(
            r"\b(?:switch|swtich|swich|siwtch|sitch|change|set|use|select)\s+(?:model\s+|provider\s+)?(?:to\s+)?(ollama(?:/[a-zA-Z0-9_\-\.:]+|\s+[a-zA-Z0-9_\-\.:]+)?|ollma(?:/[a-zA-Z0-9_\-\.:]+|\s+[a-zA-Z0-9_\-\.:]+)?|gemini[a-zA-Z0-9_\-\.\s]*|groq|openai|gpt[a-zA-Z0-9_\-\.]*|llama[a-zA-Z0-9_\-\.]*|qwen[a-zA-Z0-9_\-\.:]*|mixtral[a-zA-Z0-9_\-\.]*|gemma[a-zA-Z0-9_\-\.:]*)",
            q_lower,
        )
        if switch_match:
            target_raw = switch_match.group(1).strip()
            if target_raw.startswith("ollama") or target_raw.startswith("ollma"):
                sub = re.sub(r"^oll[a-z]*", "", target_raw).strip("/ ").strip()
                if "vision" in sub or "qwen 2b" in sub or "qwen2b" in sub:
                    target_model = f"ollama/{brain_config.OLLAMA_VISION_MODEL}"
                elif sub:
                    target_model = f"ollama/{sub}"
                else:
                    target_model = f"ollama/{brain_config.OLLAMA_MODEL}"
            elif target_raw.startswith("gemini"):
                target_model = brain_config.GEMINI_MODEL
            elif target_raw == "groq":
                target_model = "llama-3.3-70b-versatile"
            elif target_raw in ("openai", "gpt"):
                target_model = "gpt-4o-mini"
            elif target_raw.startswith("qwen") or target_raw.startswith("gemma"):
                if "2b" in target_raw and "vl" in target_raw:
                    target_model = f"ollama/{brain_config.OLLAMA_VISION_MODEL}"
                else:
                    target_model = f"ollama/{target_raw}"
            else:
                target_model = target_raw

            return StructuredIntent(
                intent="MODEL_SWITCH",
                confidence=0.98,
                arguments={"target_model": target_model, "raw_query": query},
                summary=f"User requested switching active model to '{target_model}'",
            )

        # ── AUTOMATION_TASK: Dynamic computer control, app launching, actions ──
        # 1. Regex match for opening, closing, or controlling any application or action
        app_action_pattern = re.search(
            r"^(?:please\s+)?(?:can\s+you\s+)?(open|launch|start|run|close|quit|kill|terminate|exit)\s+(?:the\s+)?([a-zA-Z0-9_\s\+\-\.]+?)(?:\s+(?:app|browser|application|program|window))?$",
            q_lower,
        )
        if app_action_pattern:
            action_verb = app_action_pattern.group(1).strip()
            target_entity = app_action_pattern.group(2).strip()
            # If target is not a pure question or vision request
            if target_entity and not any(target_entity.startswith(w) for w in ("camera", "battery", "image")):
                return StructuredIntent(
                    intent="AUTOMATION_TASK",
                    confidence=0.99,
                    arguments={"command": query, "action": action_verb, "target": target_entity},
                    summary=f"User requested automation: '{action_verb} {target_entity}'",
                )

        automation_triggers = [
            # System and Window Control
            "minimize", "maximize", "restore window", "unmaximize", "snap window", "snap left", "snap right",
            "show desktop", "hide all windows", "switch window", "list windows", "open windows",
            "close this window", "close active window", "close window", "close this app",
            # Audio and Volume
            "volume up", "volume down", "mute volume", "mute audio", "mute sound", "mute", "unmute", "unmute volume",
            "set volume", "louder", "quieter", "turn up volume", "turn down volume",
            "next track", "previous track", "play music", "pause music",
            # Brightness
            "brightness up", "brightness down", "increase brightness", "decrease brightness", "set brightness",
            "dim screen", "dim the screen", "brighter", "dimmer",
            # Power Management
            "lock screen", "lock the screen", "lock laptop", "lock pc", "lock computer", "lock",
            "sleep pc", "sleep laptop", "put pc to sleep", "put laptop to sleep", "sleep",
            "shutdown pc", "shutdown laptop", "turn off pc", "turn off laptop", "shutdown",
            "restart pc", "restart laptop", "reboot pc", "reboot laptop", "restart", "reboot",
            # System Utilities & Apps
            "task manager", "open task manager", "open taskmgr", "open settings", "windows settings",
            "open file explorer", "open explorer", "open my files", "empty recycle bin", "clear recycle bin", "empty trash",
            # Network & Connectivity
            "turn on wifi", "enable wifi", "wifi on", "turn off wifi", "disable wifi", "wifi off", "wifi status",
            "turn on bluetooth", "enable bluetooth", "bluetooth on", "turn off bluetooth", "disable bluetooth", "bluetooth off",
            # Process & Hardware Management
            "list processes", "running processes", "top processes", "kill process", "kill ", "terminate process", "terminate ",
            "take a screenshot", "take screenshot", "capture my screen", "capture screen", "screenshot",
            # Messaging and communication (full names + abbreviations)
            "message to ", "message ", "msg to ", "msg ", "text to ", "text ", "send message", "send a message",
            "dm to ", "dm ", "send email", "send an email", "send mail", "send a mail", "email to ", "mail to ", "compose email", "compose mail",
            "open whatsapp", "open telegram", "open discord", "open slack", "open gmail",
            "in telegram", "on telegram", "in whatsapp", "on whatsapp",
            # App/action verbs
            "open ", "launch ", "start ", "run ", "close ", "quit ",
            # Mouse/click control
            "click the", "click on", "right click", "double click",
            "drag and drop", "drag from",
            # Keyboard control
            "type in", "type into", "press enter", "press escape", "press tab",
            "hold ctrl", "press ctrl", "press alt", "press shift",
            "copy and paste", "select all and",
            # Browser actions
            "navigate to", "go to website", "go to url", "open website", "open url",
            "search for on google", "search google for", "google search", "search youtube for",
            "new tab", "close tab", "refresh the page", "go back", "go forward",
            "scroll down", "scroll up", "scroll to",
            "fill in the form", "fill out", "fill the field",
            # File operations
            "create a file", "create file", "save file as", "delete file",
            "rename the file", "move the file", "copy the file", "search for file",
            # Workflow
            "start recording", "stop recording", "record what i do",
            "replay ", "run workflow", "play workflow", "learn this",
            "repeat that workflow", "do that again",
            # Automation commands
            "automate", "automation status", "dry run mode", "emergency stop",
            "stop everything", "stop automation",
            "list apps", "show apps", "installed apps", "what apps are installed",
        ]

        automation_regex_patterns = [
            r"\b(?:set\s+)?brightness\s+(?:to\s+)?\d+\b",
            r"\b(?:set\s+)?volume\s+(?:to\s+)?\d+\b",
            r"\b(?:turn\s+up|turn\s+down|increase|decrease|raise|lower)\s+(?:the\s+)?(?:volume|brightness|sound)\b",
            r"\b(?:email|mail)\s+[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            r"\b(?:send|compose|write)\s+(?:a\s+)?(?:formal\s+|casual\s+)?(?:mail|email|message|dm)\b",
            r"\b(?:open|launch)\s+(?:whatsapp|telegram|discord|slack|gmail)\b",
            r"\b(?:dm|msg|text)\s+[a-zA-Z0-9_]+\s+on\s+(?:discord|slack|telegram|whatsapp)\b",
        ]

        if any(t in q_lower for t in automation_triggers) or any(re.search(p, q_lower) for p in automation_regex_patterns):
            return StructuredIntent(
                intent="AUTOMATION_TASK",
                confidence=0.97,
                arguments={"command": query},
                summary=f"User requested computer automation: '{query[:60]}'",
            )


        # ── FILESYSTEM_OPERATION: Native file and folder queries, sizes, contents, operations ──
        fs_patterns = [
            # Size queries
            r"\b(?:size\s+of|how\s+big\s+is|folder\s+size|file\s+size|how\s+much\s+space|how\s+many\s+files)\b",
            # Listing / directory contents
            r"\b(?:list\s+files|show\s+files|contents\s+of|what\s+files|which\s+files|files\s+in|folders\s+in|inside\s+folder|files\s+on\s+desktop|files\s+in\s+desktop|what\s+is\s+on\s+my\s+desktop)\b",
            # File operations and reading
            r"\b(?:read\s+file|view\s+file|cat\s+file|show\s+file|check\s+file|content\s+of\s+file|contents\s+of\s+file|contents?\s+of\s+[a-zA-Z0-9_\-\.]+\s+file|what\s+is\s+in\s+.*file|what's\s+in\s+.*file|whats\s+in\s+.*file|open\s+and\s+read)\b",
            r"\b(?:create\s+file|make\s+file|write\s+file|delete\s+file|remove\s+file)\b",
            r"\b(?:create\s+folder|make\s+folder|make\s+directory|mkdir|delete\s+folder)\b",
            # File presence and existence checks
            r"\b(?:is\s+there\s+a\s+file|is\s+there\s+a\s+folder|file\s+named\s+[a-zA-Z0-9_\-\.]+|file\s+called\s+[a-zA-Z0-9_\-\.]+|is\s+.*(?:file|folder)\s+(?:present|exists?)|(?:file|folder)\s+is\s+(?:present|exists?)|check\s+if\s+.*(?:file|folder|exists?|present)|check\s+whether\s+.*(?:file|folder|exists?|present)|do\s+i\s+have\s+.*(?:file|folder)|exists?\s+(?:in|on)\s+desktop)\b",
            # File inspection and details
            r"\b(?:what\s+is\s+(?:that|this|the)\s+file|inspect\s+(?:file|folder)|file\s+info|details\s+of\s+file|tell\s+me\s+about\s+(?:that|this|the)\s+file|whats?\s+in\s+that\s+file)\b",
        ]
        if any(re.search(p, q_lower) for p in fs_patterns):
            return StructuredIntent(
                intent="FILESYSTEM_OPERATION",
                confidence=0.98,
                arguments={"query": query},
                summary=f"User requested filesystem inspection or operation: '{query[:60]}'",
            )

        # ── SYSTEM_TELEMETRY: Hardware & OS telemetry ──
        if any(re.search(rf"\b{w}\b", q_lower) for w in ["cpu", "ram", "memory usage", "system info", "os version", "disk space", "hardware status"]):
            return StructuredIntent(
                intent="SYSTEM_TELEMETRY",
                confidence=0.95,
                arguments={"query": query},
                summary="User requested system metrics or OS telemetry.",
            )

        # ── SCREEN_VISION: Explicit visual perception queries ──
        refresh_triggers = [
            "look again", "refresh vision", "analyze again", "analyze the screen again",
            "re-analyze screen", "reanalyze screen", "check screen again", "look at screen again",
            "what is on my screen right now", "take another look", "refresh screen"
        ]
        if any(w in q_lower for w in refresh_triggers):
            v_type = IntentEngine.classify_visual_query_type(query)
            return StructuredIntent(
                intent="SCREEN_VISION",
                confidence=0.99,
                arguments={"query": query, "visual_query_type": v_type, "force_refresh": True},
                summary=f"User explicitly requested fresh screen analysis ({v_type}).",
            )

        vision_triggers = [
            # Explicit screen vision queries
            "on my screen", "in my screen", "on the screen", "see the screen",
            "what is on my screen", "whats on my screen", "what's on my screen",
            "what is on screen", "whats on screen", "what's on screen",
            "can you see", "what do you see", "what can you see", "see my", "look at my",
            "look at screen", "describe my screen", "describe the screen", "describe what you see",
            # Image on screen
            "image on screen", "picture on screen", "photo on screen", "image on my screen",
            "picture on my screen", "what is the image", "what's the image", "what image",
            "what inmage", "inmage", "what imge", "what pic", "what picture", "what photo",
            "what animal", "what object", "what is in the image", "whats in the image",
            # Taskbar / UI visual metrics
            "battery %", "battery percentage", "see my battery",
            # Cursor / position
            "cursor position", "where is the cursor", "where is my cursor",
            # Visual highlights / colors
            "red highlight", "blue highlight", "color of the screen", "background colour", "background color",
            # General visual requests
            "look at this", "see this", "what is shown",
        ]
        if any(w in q_lower for w in vision_triggers):
            v_type = IntentEngine.classify_visual_query_type(query)
            return StructuredIntent(
                intent="SCREEN_VISION",
                confidence=0.98,
                arguments={"query": query, "visual_query_type": v_type, "force_refresh": False},
                summary=f"User requested visual perception ({v_type}) of active desktop.",
            )

        # ── REALTIME_KNOWLEDGE_SEARCH: Live web search ──
        realtime_triggers = [
            "today's", "right now", "latest news", "recent news",
            "search internet", "search the web", "search web", "look up online",
            "live price", "current price", "price of", "stock price",
            "weather", "who won", "score", "live update", "breaking news",
            "what is happening", "what happened today", "news today",
        ]
        if any(w in q_lower for w in realtime_triggers):
            return StructuredIntent(
                intent="REALTIME_KNOWLEDGE_SEARCH",
                confidence=0.95,
                arguments={"search_query": query},
                summary="User requested real-time live internet information or current web search.",
            )

        # 5. Open Application Intent — ONLY for explicit launch commands, NOT screen queries
        open_app_match = re.search(r"\b(?:open|launch|start|run)\s+([a-zA-Z0-9_\-\s]+)", q_lower)
        if open_app_match and not any(w in q_lower for w in [
            "code", "script", "file", "url", "browser", "screen", "taskbar", "window", "app", "image"
        ]):
            app_name = open_app_match.group(1).strip()
            return StructuredIntent(
                intent="OPEN_APPLICATION",
                confidence=0.95,
                arguments={"application_name": app_name},
                summary=f"User requested to launch application: '{app_name}'",
            )

        # 5. Open Web / Browser Intent — only if not a memory or email statement
        is_memory_or_email = bool(re.search(r"\b(?:remember|recall|my\s+email|email\s+is|favorite|favourite|who\s+am\s+i)\b", q_lower))
        if not is_memory_or_email and any(w in q_lower for w in ["open url", "go to website", "navigate to", "http://", "https://", "www."]):
            url_match = re.search(r"https?://[^\s]+", query)
            target_url = url_match.group(0) if url_match else query
            return StructuredIntent(
                intent="BROWSER_NAVIGATION",
                confidence=0.90,
                arguments={"target": target_url},
                summary="User requested browser navigation or web page scraping",
            )

        # 6. Coding Request Intent
        code_triggers = ["def ", "class ", "write code", "fix bug", "refactor", "write a script", "write script", "python script", "write a python", "code for", "generate code", "implement function", "create function"]
        if any(w in q_lower for w in code_triggers) and not any(kw in q_lower for kw in ["favorite", "favourite", "what is", "who is", "remember", "recall"]):
            return StructuredIntent(
                intent="CODE_GENERATION",
                confidence=0.92,
                arguments={"task": query},
                summary="User requested software development or code generation",
            )

        # 7. Planning Request Intent
        if any(w in q_lower for w in ["plan", "roadmap", "steps", "how to build", "architecture", "workflow"]):
            return StructuredIntent(
                intent="TASK_PLANNING",
                confidence=0.88,
                arguments={"goal": query},
                summary="User requested multi-step execution plan or roadmap",
            )

        # 8. Knowledge Explanation Request
        if any(w in q_lower for w in ["explain", "what is", "why does", "how does", "compare", "define", "describe"]):
            return StructuredIntent(
                intent="KNOWLEDGE_REQUEST",
                confidence=0.90,
                arguments={"topic": query},
                summary="User requested analytical knowledge explanation",
            )

        # Default General Dialogue
        return StructuredIntent(
            intent="GENERAL_CONVERSATION",
            confidence=0.98,
            arguments={"query": query},
            summary="General conversational interaction",
        )


# Global intent engine singleton
intent_engine = IntentEngine()
