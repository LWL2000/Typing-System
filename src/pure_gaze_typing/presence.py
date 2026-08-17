from __future__ import annotations

from enum import Enum


class PresenceEvent(str, Enum):
    NONE = "none"
    LEFT = "left"
    RETURNED = "returned"


class SeatReturnDetector:
    def __init__(self, *, absence_seconds: float = 1.5) -> None:
        if absence_seconds <= 0:
            raise ValueError("absence_seconds must be positive")
        self.absence_seconds = float(absence_seconds)
        self.reset()

    def reset(self) -> None:
        self._absent_since: float | None = None
        self._away = False

    def update(
        self,
        timestamp: float,
        *,
        face_detected: bool,
        active: bool,
    ) -> PresenceEvent:
        now = float(timestamp)
        if not active:
            self.reset()
            return PresenceEvent.NONE
        if face_detected:
            self._absent_since = None
            if self._away:
                self._away = False
                return PresenceEvent.RETURNED
            return PresenceEvent.NONE
        if self._away:
            return PresenceEvent.NONE
        if self._absent_since is None:
            self._absent_since = now
            return PresenceEvent.NONE
        if now - self._absent_since >= self.absence_seconds:
            self._away = True
            return PresenceEvent.LEFT
        return PresenceEvent.NONE
