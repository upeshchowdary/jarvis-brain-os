"""In-process metrics: per-stage latency, provider usage and error counters."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict


class Metrics:
    def __init__(self, window: int = 500) -> None:
        self._lat: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._counters: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self.started = time.time()

    def observe(self, stage: str, ms: float) -> None:
        with self._lock:
            self._lat[stage].append(float(ms))

    def incr(self, name: str, by: int = 1) -> None:
        with self._lock:
            self._counters[name] += by

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            stages = {}
            for stage, values in self._lat.items():
                ordered = sorted(values)
                n = len(ordered)
                if not n:
                    continue
                stages[stage] = {
                    "count": n,
                    "avg_ms": round(sum(ordered) / n, 1),
                    "p50_ms": round(ordered[n // 2], 1),
                    "p95_ms": round(ordered[min(n - 1, int(n * 0.95))], 1),
                    "max_ms": round(ordered[-1], 1),
                }
            return {"uptime_s": round(time.time() - self.started), "latency": stages, "counters": dict(self._counters)}


metrics = Metrics()


class timed:
    """``with timed("router"):`` records elapsed milliseconds under that stage."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.ms = 0.0

    def __enter__(self) -> "timed":
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = (time.perf_counter() - self._t) * 1000
        metrics.observe(self.stage, self.ms)
