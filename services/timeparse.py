"""Natural-language time expressions → datetimes and simple recurrences (no dependencies).

Supports: "in 10 minutes", "in an hour", "in 2 hours 30 minutes", "at 6 pm", "at 18:30",
"tomorrow at 9", "tonight at 8", "this evening", "on friday at 5pm", "next monday 10am",
"at noon", "every day at 7am", "every weekday at 9", "every monday at 10:15",
"every 30 minutes", "every hour". ``dateparser`` (if installed) handles anything else.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WD_SHORT = {d[:3]: i for i, d in enumerate(WEEKDAYS)}
_WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10, "fifteen": 15,
             "twenty": 20, "thirty": 30, "forty five": 45, "half an": 0.5, "half a": 0.5, "couple of": 2, "few": 3}
_UNIT_SECONDS = {"second": 1, "sec": 1, "minute": 60, "min": 60, "hour": 3600, "hr": 3600, "day": 86400, "week": 604800}


@dataclass
class ParsedTime:
    due: datetime
    recurrence: Optional[str] = None  # "daily" | "weekdays" | "weekly:<0-6>" | "every:<seconds>"
    matched: str = ""

    def describe(self) -> str:
        when = self.due.strftime("%I:%M %p").lstrip("0")
        day = self.due.strftime("%A %d %b")
        if self.recurrence == "daily":
            return f"every day at {when}"
        if self.recurrence == "weekdays":
            return f"every weekday at {when}"
        if self.recurrence and self.recurrence.startswith("weekly:"):
            return f"every {WEEKDAYS[int(self.recurrence[7:])].title()} at {when}"
        if self.recurrence and self.recurrence.startswith("every:"):
            secs = int(self.recurrence[6:])
            return f"every {secs // 3600} hour(s)" if secs % 3600 == 0 else f"every {secs // 60} minute(s)"
        today = datetime.now().date()
        if self.due.date() == today:
            return f"today at {when}"
        if self.due.date() == today + timedelta(days=1):
            return f"tomorrow at {when}"
        return f"{day} at {when}"


_TIME_RE = re.compile(r"(?P<h>\d{1,2})(?:[:.](?P<m>\d{2}))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)?", re.I)


def _parse_clock(text: str, default_pm_hint: bool = False) -> Optional[tuple[int, int]]:
    t = text.strip().lower()
    if t in ("noon", "midday"):
        return 12, 0
    if t == "midnight":
        return 0, 0
    m = _TIME_RE.fullmatch(t)
    if not m:
        return None
    h, mins = int(m.group("h")), int(m.group("m") or 0)
    ap = (m.group("ap") or "").replace(".", "")
    if h > 23 or mins > 59:
        return None
    if ap == "pm" and h < 12:
        h += 12
    elif ap == "am" and h == 12:
        h = 0
    elif not ap and default_pm_hint and 1 <= h <= 11:
        h += 12
    return h, mins


def _next_at(now: datetime, h: int, m: int, day_offset: int = 0) -> datetime:
    target = (now + timedelta(days=day_offset)).replace(hour=h, minute=m, second=0, microsecond=0)
    if day_offset == 0 and target <= now:
        target += timedelta(days=1)
    return target


def _next_weekday(now: datetime, weekday: int, h: int, m: int, force_next_week: bool = False) -> datetime:
    days = (weekday - now.weekday()) % 7
    target = (now + timedelta(days=days)).replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now or (force_next_week and days == 0):
        target += timedelta(days=7)
    return target


def _next_workday(now: datetime, h: int, m: int) -> datetime:
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target


_CLOCK = r"(?P<clock>\d{1,2}(?:[:.]\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?|noon|midday|midnight)"


def parse_when(text: str, now: Optional[datetime] = None) -> Optional[ParsedTime]:
    """Find the first time expression in ``text``. Returns None if nothing recognisable."""
    now = now or datetime.now()
    t = " " + text.lower().strip() + " "

    # Relative: "in 10 minutes", "in an hour and a half", "in 2 hours 30 minutes"
    m = re.search(r"\bin\s+((?:(?:\d+(?:\.\d+)?|a|an|one|two|three|four|five|ten|fifteen|twenty|thirty|half an|half a|couple of|few)\s*"
                  r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)(?:\s*(?:and\s*)?)?)+)", t)
    if m:
        total = 0.0
        for num, unit in re.findall(r"(\d+(?:\.\d+)?|half an|half a|couple of|an|a|one|two|three|four|five|ten|fifteen|twenty|thirty|few)\s*"
                                    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)", m.group(1)):
            n = float(num) if num[0].isdigit() else _WORD_NUM.get(num, 1)
            total += n * _UNIT_SECONDS[unit.rstrip("s")]
        half = re.search(r"\b(hours?|hrs?|minutes?|mins?|days?)\s+and\s+a\s+half\b", t)
        if half:
            total += _UNIT_SECONDS[half.group(1).rstrip("s")] / 2
        if total > 0:
            return ParsedTime(now + timedelta(seconds=total), matched=m.group(0).strip())

    # Recurring interval: "every 30 minutes", "every hour"
    m = re.search(r"\bevery\s+(?:(\d+)\s*)?(minutes?|mins?|hours?|hrs?)\b", t)
    if m:
        n = int(m.group(1) or 1)
        secs = n * _UNIT_SECONDS[m.group(2).rstrip("s")]
        return ParsedTime(now + timedelta(seconds=secs), f"every:{secs}", m.group(0).strip())

    clock_m = re.search(r"\b(?:at\s+)?" + _CLOCK + r"(?=\s|$|,|\.)", t)
    has_at = bool(re.search(r"\bat\s+" + _CLOCK, t)) or bool(re.search(r"\d\s*(?:am|pm|a\.m|p\.m)|noon|midnight|\d:\d\d", t))
    evening = bool(re.search(r"\b(tonight|this evening|evening)\b", t))
    clock = None
    if clock_m and has_at:
        clock = _parse_clock(clock_m.group("clock"), default_pm_hint=evening)
    elif evening:
        clock = (20, 0)
    elif re.search(r"\b(this morning|morning)\b", t):
        clock = (9, 0)
    elif re.search(r"\bthis afternoon|afternoon\b", t):
        clock = (15, 0)
    h, mins = clock if clock else (9, 0)

    # Recurring by day: "every day", "daily", "every weekday", "every monday"
    if re.search(r"\b(every\s*day|everyday|daily|each day)\b", t):
        return ParsedTime(_next_at(now, h, mins), "daily", "every day")
    if re.search(r"\b(every\s+weekday|weekdays|every\s+working\s+day|on\s+weekdays)\b", t):
        return ParsedTime(_next_workday(now, h, mins), "weekdays", "every weekday")
    m = re.search(r"\bevery\s+(" + "|".join(WEEKDAYS) + r"|" + "|".join(_WD_SHORT) + r")\b", t)
    if m:
        wd = _WD_SHORT[m.group(1)[:3]]
        return ParsedTime(_next_weekday(now, wd, h, mins), f"weekly:{wd}", m.group(0))

    # One-off day references
    if re.search(r"\btomorrow\b", t):
        return ParsedTime(_next_at(now, h, mins, 1), matched="tomorrow")
    m = re.search(r"\b(?:on\s+|next\s+|this\s+)?(" + "|".join(WEEKDAYS) + r")\b", t)
    if m:
        force_next = "next " in m.group(0)
        return ParsedTime(_next_weekday(now, WEEKDAYS.index(m.group(1)), h, mins, force_next), matched=m.group(0).strip())
    if clock is not None:
        return ParsedTime(_next_at(now, h, mins), matched=(clock_m.group(0).strip() if clock_m else "tonight"))

    try:  # optional, broader coverage
        import dateparser  # type: ignore

        dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": now})
        if dt and dt > now:
            return ParsedTime(dt.replace(tzinfo=None), matched=text)
    except Exception:
        pass
    return None


def next_occurrence(recurrence: str, last_due: datetime, now: Optional[datetime] = None) -> Optional[datetime]:
    now = now or datetime.now()
    if recurrence == "daily":
        nxt = last_due + timedelta(days=1)
        while nxt <= now:
            nxt += timedelta(days=1)
        return nxt
    if recurrence == "weekdays":
        return _next_workday(max(now, last_due), last_due.hour, last_due.minute)
    if recurrence.startswith("weekly:"):
        return _next_weekday(max(now, last_due) + timedelta(minutes=1), int(recurrence[7:]), last_due.hour, last_due.minute)
    if recurrence.startswith("every:"):
        step = timedelta(seconds=int(recurrence[6:]))
        nxt = last_due + step
        while nxt <= now:
            nxt += step
        return nxt
    return None


def strip_time_phrase(text: str, parsed: ParsedTime) -> str:
    """Remove the time expression and filler from a reminder text."""
    out = text
    if parsed.matched:
        out = re.sub(re.escape(parsed.matched), " ", out, flags=re.I)
    out = re.sub(r"\b(?:at\s+)?" + _CLOCK + r"(?=\s|$)", " ", out, flags=re.I) if parsed.matched else out
    out = re.sub(r"\b(today|tonight|tomorrow|this evening|this morning|this afternoon|every|at|on)\b\s*$", " ", out.strip(), flags=re.I)
    out = re.sub(r"^\s*(?:to|that|about|for|at|in|on)\s+", "", out.strip(), flags=re.I)
    return re.sub(r"\s+", " ", out).strip(" ,.-")
