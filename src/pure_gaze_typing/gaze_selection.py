from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class DwellUpdate:
    target_id: str | None
    progress: float
    triggered_target_id: str | None
    armed: bool


class DwellSelector:
    def __init__(
        self,
        dwell_seconds: float,
        *,
        leave_grace_seconds: float = 0.2,
        rearm_seconds: float = 0.3,
        target_dwell_seconds: Mapping[str, float] | None = None,
    ) -> None:
        if not math.isfinite(dwell_seconds) or dwell_seconds <= 0:
            raise ValueError("dwell_seconds must be positive and finite")
        self.dwell_seconds = float(dwell_seconds)
        self.leave_grace_seconds = float(leave_grace_seconds)
        self.rearm_seconds = float(rearm_seconds)
        self.target_dwell_seconds = dict(target_dwell_seconds or {})
        self.reset()

    def reset(self) -> None:
        self._target_id: str | None = None
        self._accumulated = 0.0
        self._last_timestamp: float | None = None
        self._away_since: float | None = None
        self._locked_target: str | None = None
        self._rearm_out_since: float | None = None

    def update(
        self,
        timestamp: float,
        target_id: str | None,
        *,
        valid: bool,
        blink: bool,
    ) -> DwellUpdate:
        now = float(timestamp)
        if self._last_timestamp is not None and now < self._last_timestamp:
            raise ValueError("timestamps must be monotonic")
        delta = 0.0 if self._last_timestamp is None else now - self._last_timestamp
        self._last_timestamp = now

        if self._locked_target is not None:
            if valid and not blink and target_id == self._locked_target:
                self._rearm_out_since = None
                return DwellUpdate(None, 0.0, None, False)
            if self._rearm_out_since is None:
                self._rearm_out_since = now
            if now - self._rearm_out_since < self.rearm_seconds:
                return DwellUpdate(None, 0.0, None, False)
            self._locked_target = None
            self._rearm_out_since = None

        if blink:
            return self._snapshot()
        effective_target = target_id if valid else None
        if effective_target is None:
            if self._target_id is None:
                return self._snapshot()
            if delta > self.leave_grace_seconds:
                self._clear_progress()
                return self._snapshot()
            if self._away_since is None:
                self._away_since = now
            elif now - self._away_since > self.leave_grace_seconds:
                self._clear_progress()
            return self._snapshot()

        if effective_target != self._target_id:
            self._target_id = effective_target
            self._accumulated = 0.0
            self._away_since = None
            return self._snapshot()

        if self._away_since is not None:
            if now - self._away_since > self.leave_grace_seconds:
                self._accumulated = 0.0
            self._away_since = None
        else:
            self._accumulated += delta

        required = self.target_dwell_seconds.get(effective_target, self.dwell_seconds)
        progress = min(1.0, self._accumulated / required)
        if self._accumulated >= required:
            triggered = effective_target
            self._locked_target = triggered
            self._target_id = None
            self._accumulated = 0.0
            return DwellUpdate(triggered, 1.0, triggered, False)
        return DwellUpdate(effective_target, progress, None, True)

    def _clear_progress(self) -> None:
        self._target_id = None
        self._accumulated = 0.0
        self._away_since = None

    def _snapshot(self) -> DwellUpdate:
        required = self.target_dwell_seconds.get(self._target_id or "", self.dwell_seconds)
        progress = 0.0 if self._target_id is None else min(1.0, self._accumulated / required)
        return DwellUpdate(self._target_id, progress, None, self._locked_target is None)


@dataclass(frozen=True)
class BlinkUpdate:
    count: int
    triple_blink: bool


class TripleBlinkDetector:
    def __init__(
        self,
        *,
        window_seconds: float = 2.5,
        min_closed_seconds: float = 0.05,
        max_closed_seconds: float = 0.6,
        cooldown_seconds: float = 1.0,
    ) -> None:
        self.window_seconds = float(window_seconds)
        self.min_closed_seconds = float(min_closed_seconds)
        self.max_closed_seconds = float(max_closed_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self._blink_times: deque[float] = deque()
        self._closed_since: float | None = None
        self._invalid_closure = False
        self._cooldown_until = float("-inf")

    def reset(self) -> None:
        self._blink_times.clear()
        self._closed_since = None
        self._invalid_closure = False

    def update(self, timestamp: float, *, face_detected: bool, blink: bool) -> BlinkUpdate:
        now = float(timestamp)
        if not face_detected:
            self.reset()
            return BlinkUpdate(0, False)
        while self._blink_times and now - self._blink_times[0] > self.window_seconds:
            self._blink_times.popleft()
        if now < self._cooldown_until:
            return BlinkUpdate(0, False)

        if blink:
            if self._closed_since is None:
                self._closed_since = now
                self._invalid_closure = False
            elif now - self._closed_since > self.max_closed_seconds:
                self._invalid_closure = True
            return BlinkUpdate(len(self._blink_times), False)

        if self._closed_since is None:
            return BlinkUpdate(len(self._blink_times), False)
        duration = now - self._closed_since
        valid_blink = (
            not self._invalid_closure
            and self.min_closed_seconds <= duration <= self.max_closed_seconds
        )
        self._closed_since = None
        self._invalid_closure = False
        if not valid_blink:
            return BlinkUpdate(len(self._blink_times), False)

        self._blink_times.append(now)
        count = len(self._blink_times)
        if count >= 3:
            self._blink_times.clear()
            self._cooldown_until = now + self.cooldown_seconds
            return BlinkUpdate(3, True)
        return BlinkUpdate(count, False)
