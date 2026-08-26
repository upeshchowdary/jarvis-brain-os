"""Context Memory & Reference Resolution for JARVIS Automation Engine.

Maintains short-term execution memory, tracks active entities (windows, tabs,
videos, contacts, files, search results), and resolves natural language references
(pronouns, spatial, temporal, relative references like 'it', 'that', 'the 2nd one', 'his channel').
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Entity:
    """Represents an active entity tracked in memory."""
    id: str
    name: str
    entity_type: str  # "video", "channel", "contact", "file", "tab", "window", "url", "search_result", "email"
    value: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)


@dataclass
class MemoryState:
    """Snapshot of current computer context state."""
    active_app: str = ""
    active_window_title: str = ""
    active_url: str = ""
    active_tab_title: str = ""
    focused_element: str = ""
    last_downloaded_file: str = ""
    last_created_file: str = ""
    last_search_query: str = ""
    search_results: List[Entity] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class ExecutionMemory:
    """Manages short-term execution state across multi-turn automation."""

    def __init__(self) -> None:
        self.state = MemoryState()
        self._max_history = 50

    def update_app_state(
        self,
        active_app: Optional[str] = None,
        window_title: Optional[str] = None,
        active_url: Optional[str] = None,
        tab_title: Optional[str] = None,
    ) -> None:
        if active_app is not None:
            self.state.active_app = active_app
        if window_title is not None:
            self.state.active_window_title = window_title
        if active_url is not None:
            self.state.active_url = active_url
        if tab_title is not None:
            self.state.active_tab_title = tab_title
        self.state.updated_at = time.time()

    def add_entity(
        self,
        name: str,
        entity_type: str,
        value: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Entity:
        entity_id = f"{entity_type}_{len(self.state.entities) + 1}_{int(time.time())}"
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            value=value or name,
            metadata=metadata or {},
        )
        self.state.entities.insert(0, entity)  # Most recent first
        if len(self.state.entities) > 50:
            self.state.entities = self.state.entities[:50]
        return entity

    def get_latest_entity(self, entity_type: Optional[str] = None) -> Optional[Entity]:
        if not self.state.entities:
            return None
        if entity_type is None:
            return self.state.entities[0]
        for entity in self.state.entities:
            if entity.entity_type.lower() == entity_type.lower():
                return entity
        return None

    def set_search_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        self.state.last_search_query = query
        self.state.search_results = []
        for idx, res in enumerate(results, 1):
            entity = Entity(
                id=f"res_{idx}",
                name=res.get("title", f"Result {idx}"),
                entity_type="search_result",
                value=res.get("url", res.get("value", "")),
                metadata={"index": idx, **res},
            )
            self.state.search_results.append(entity)
            self.state.entities.insert(0, entity)

    def record_action(self, action_type: str, target: str, result: Any, success: bool) -> None:
        entry = {
            "timestamp": time.time(),
            "action_type": action_type,
            "target": target,
            "success": success,
            "result_summary": str(result)[:200],
        }
        self.state.action_history.append(entry)
        if len(self.state.action_history) > self._max_history:
            self.state.action_history = self.state.action_history[-self._max_history:]

    def resolve_reference(self, text: str) -> Dict[str, Any]:
        """
        Resolves implicit pronouns, relative references, and ordinal expressions.
        e.g. 'open the first one', 'like it', 'send it', 'open his channel', 'that file'.
        """
        text_clean = text.strip()
        text_lower = text_clean.lower()
        resolved = {
            "original": text,
            "resolved_text": text,
            "target_entity": None,
            "inferred_action": "",
        }

        # 1. Ordinal References: 'first one', 'second video', '3rd result'
        ordinal_match = re.search(
            r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|last|latest)\b\s*(?:one|result|video|link|item|file|repo|tab)?",
            text_lower,
        )
        if ordinal_match:
            ord_word = ordinal_match.group(1)
            index_map = {
                "first": 1, "1st": 1,
                "second": 2, "2nd": 2,
                "third": 3, "3rd": 3,
                "fourth": 4, "4th": 4,
                "fifth": 5, "5th": 5,
            }
            if ord_word in index_map and self.state.search_results:
                target_idx = index_map[ord_word]
                if target_idx <= len(self.state.search_results):
                    target_ent = self.state.search_results[target_idx - 1]
                    resolved["target_entity"] = target_ent
                    resolved["resolved_text"] = text_clean.replace(ordinal_match.group(0), target_ent.name)
                    return resolved

            if ord_word in ("last", "latest"):
                latest_ent = self.get_latest_entity()
                if latest_ent:
                    resolved["target_entity"] = latest_ent
                    resolved["resolved_text"] = text_clean.replace(ordinal_match.group(0), latest_ent.name)
                    return resolved

        # 2. Pronouns ('it', 'that', 'this', 'them')
        if any(w in text_lower.split() for w in ("it", "that", "this")):
            # If user says "like it", resolve 'it' to current video/post
            if "like" in text_lower:
                vid_entity = self.get_latest_entity("video")
                if vid_entity:
                    resolved["target_entity"] = vid_entity
                    resolved["inferred_action"] = "like_video"
                    return resolved
            elif "subscribe" in text_lower:
                channel_entity = self.get_latest_entity("channel")
                if channel_entity:
                    resolved["target_entity"] = channel_entity
                    resolved["inferred_action"] = "subscribe_channel"
                    return resolved
            elif "download" in text_lower:
                file_entity = self.get_latest_entity("file")
                if file_entity:
                    resolved["target_entity"] = file_entity
                    resolved["inferred_action"] = "download_file"
                    return resolved
            else:
                latest = self.get_latest_entity()
                if latest:
                    resolved["target_entity"] = latest
                    return resolved

        # 3. Channel / Profile References: 'his channel', 'their channel', 'its page'
        if any(p in text_lower for p in ("his channel", "her channel", "its channel", "their channel", "the channel")):
            curr_video = self.get_latest_entity("video")
            if curr_video and "channel" in curr_video.metadata:
                channel_name = curr_video.metadata["channel"]
                resolved["target_entity"] = Entity(
                    id=f"channel_{channel_name}",
                    name=channel_name,
                    entity_type="channel",
                    value=channel_name,
                )
                return resolved

        # 4. File References: 'the downloaded file', 'that pdf', 'the resume'
        if "downloaded file" in text_lower or "that file" in text_lower:
            if self.state.last_downloaded_file:
                resolved["target_entity"] = Entity(
                    id="downloaded_file",
                    name=self.state.last_downloaded_file,
                    entity_type="file",
                    value=self.state.last_downloaded_file,
                )
                return resolved

        return resolved


execution_memory = ExecutionMemory()
