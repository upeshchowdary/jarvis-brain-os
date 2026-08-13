"""JARVIS Screen Delta & Diff Detector Module.

Compares consecutive EnvironmentState frames to report opened/closed windows,
new popups, cursor movement, altered text lines, and UI element changes.
"""

from typing import List, Dict, Any, Optional
from vision.environment import EnvironmentState, DiffSummary


class DiffDetector:
    """Engine for calculating visual and structural deltas between environment states."""

    def compute_diff(
        self,
        prev_state: Optional[EnvironmentState],
        curr_state: EnvironmentState,
    ) -> DiffSummary:
        """Computes structural differences between previous and current EnvironmentState."""
        if prev_state is None:
            return DiffSummary(has_changes=True)

        prev_wins = {w.title for w in prev_state.windows if w.title}
        curr_wins = {w.title for w in curr_state.windows if w.title}

        new_windows = list(curr_wins - prev_wins)
        closed_windows = list(prev_wins - curr_wins)

        # Cursor movement check (>10px threshold)
        cursor_moved = (
            abs(prev_state.cursor.x - curr_state.cursor.x) > 10
            or abs(prev_state.cursor.y - curr_state.cursor.y) > 10
        )

        # Popup check
        popup_appeared = any("popup" in w.title.lower() or "dialog" in w.title.lower() for w in curr_state.windows)

        # Text changes
        prev_text = {t.text for t in prev_state.visible_text}
        curr_text = {t.text for t in curr_state.visible_text}
        changed_text = list(curr_text - prev_text)[:5]

        has_changes = bool(new_windows or closed_windows or cursor_moved or popup_appeared or changed_text)

        return DiffSummary(
            has_changes=has_changes,
            new_windows=new_windows,
            closed_windows=closed_windows,
            popup_appeared=popup_appeared,
            cursor_moved=cursor_moved,
            changed_text=changed_text,
            changed_elements=[],
        )


# Global singleton instance
diff_detector = DiffDetector()
