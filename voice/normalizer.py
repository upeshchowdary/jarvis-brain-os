"""Spoken Command Cleaning and Phonetic Normalizer for JARVIS Voice Module.

Features:
- Application & website alias resolution (e.g. "you tube" -> "YouTube", "vs code" -> "Visual Studio Code")
- Spoken acronym and punctuation normalization (e.g. "1080 p" -> "1080p", "wi fi" -> "WiFi")
- Correction of common speech-to-text transcription anomalies
"""

from __future__ import annotations

import re
from typing import List, Tuple
from loguru import logger


class CommandNormalizer:
    """Cleans and standardizes raw spoken transcripts before cognitive processing."""

    def __init__(self) -> None:
        # Common application and site phonetic aliases
        self._app_aliases: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"\b(?:you\s*tube|u\s*tube|yt)\b", re.IGNORECASE), "YouTube"),
            (re.compile(r"\b(?:vs\s*code|v\s*s\s*code|visual\s+studio\s+code)\b", re.IGNORECASE), "Visual Studio Code"),
            (re.compile(r"\b(?:git\s*hub|git\s+hub)\b", re.IGNORECASE), "GitHub"),
            (re.compile(r"\b(?:linked\s*in|link\s*in)\b", re.IGNORECASE), "LinkedIn"),
            (re.compile(r"\b(?:whats\s*app|what's\s*app|wapp)\b", re.IGNORECASE), "WhatsApp"),
            (re.compile(r"\b(?:tele\s*gram|tg)\b", re.IGNORECASE), "Telegram"),
            (re.compile(r"\b(?:google\s+chrome|chrome\s+browser)\b", re.IGNORECASE), "Chrome"),
            (re.compile(r"\b(?:file\s+explorer|my\s+files)\b", re.IGNORECASE), "File Explorer"),
            (re.compile(r"\b(?:task\s+manager|taskmgr)\b", re.IGNORECASE), "Task Manager"),
            (re.compile(r"\b(?:windows\s+settings|system\s+settings)\b", re.IGNORECASE), "Settings"),
            (re.compile(r"\b(?:py\s*thon)\b", re.IGNORECASE), "Python"),
        ]

        # Common phonetic speech-to-text corrections
        self._phonetic_corrections: List[Tuple[re.Pattern, str]] = [
            (re.compile(r"\bcomplier\b", re.IGNORECASE), "compiler"),
            (re.compile(r"\b1080\s*p\b", re.IGNORECASE), "1080p"),
            (re.compile(r"\b720\s*p\b", re.IGNORECASE), "720p"),
            (re.compile(r"\b4\s*k\b", re.IGNORECASE), "4K"),
            (re.compile(r"\bfull\s*screen\b", re.IGNORECASE), "fullscreen"),
            (re.compile(r"\bwi\s*fi\b", re.IGNORECASE), "WiFi"),
            (re.compile(r"\bblue\s*tooth\b", re.IGNORECASE), "Bluetooth"),
            (re.compile(r"\bdot\s*com\b", re.IGNORECASE), ".com"),
            (re.compile(r"\bdot\s*org\b", re.IGNORECASE), ".org"),
            (re.compile(r"\bdot\s*io\b", re.IGNORECASE), ".io"),
            (re.compile(r"\bdot\s*pdf\b", re.IGNORECASE), ".pdf"),
            (re.compile(r"\bdot\s*py\b", re.IGNORECASE), ".py"),
            (re.compile(r"\bdot\s*txt\b", re.IGNORECASE), ".txt"),
            (re.compile(r"\bat\s+the\s+rate\b", re.IGNORECASE), "@"),
            (re.compile(r"\bnew\s+folder\b", re.IGNORECASE), "new folder"),
            (re.compile(r"\bh\s*t\s*t\s*p\s*s?\b", re.IGNORECASE), "https://"),
        ]

        # Spoken filler words at start of command (supports chained e.g. "um please")
        self._filler_prefix = re.compile(
            r"^(?:(?:um+|uh+|ah+|so+|like+|actually+|basically+|please+|can\s+you+|could\s+you+)\s*)+",
            re.IGNORECASE
        )

    def normalize(self, raw_transcript: str) -> str:
        """
        Cleans and standardizes raw spoken transcript.
        
        Example:
            "um please open you tube and search for py thon complier"
            -> "open YouTube and search for Python compiler"
        """
        if not raw_transcript:
            return ""

        text = raw_transcript.strip()

        # 1. Clean leading filler words
        text = self._filler_prefix.sub("", text).strip()

        # 2. Apply application & site aliases
        for pattern, replacement in self._app_aliases:
            text = pattern.sub(replacement, text)

        # 3. Apply phonetic & technical corrections
        for pattern, replacement in self._phonetic_corrections:
            text = pattern.sub(replacement, text)

        # 4. Clean extra whitespaces
        text = re.sub(r"\s+", " ", text).strip()

        if text != raw_transcript.strip():
            logger.debug(f"[Normalizer] Normalized: '{raw_transcript}' -> '{text}'")

        return text


command_normalizer = CommandNormalizer()
