from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class TypingSettings:
    show_gaze_point: bool = True
    dwell_seconds: float = 1.0
    validate_calibration: bool = False
    adaptive_correction_enabled: bool = True
    blink_return_count: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.show_gaze_point, bool):
            raise ValueError("show_gaze_point must be a boolean")
        if not isinstance(self.validate_calibration, bool):
            raise ValueError("validate_calibration must be a boolean")
        if not isinstance(self.adaptive_correction_enabled, bool):
            raise ValueError("adaptive_correction_enabled must be a boolean")
        if isinstance(self.blink_return_count, bool) or not 2 <= int(self.blink_return_count) <= 5:
            raise ValueError("blink_return_count must be between 2 and 5")
        if not math.isfinite(self.dwell_seconds) or not 0.5 <= self.dwell_seconds <= 3.0:
            raise ValueError("dwell_seconds must be between 0.5 and 3.0")
        object.__setattr__(self, "dwell_seconds", round(float(self.dwell_seconds), 1))
        object.__setattr__(self, "blink_return_count", int(self.blink_return_count))


def load_settings(path: Path) -> TypingSettings:
    path = Path(path)
    if not path.exists():
        return TypingSettings()
    required_fields = {"show_gaze_point", "dwell_seconds", "validate_calibration"}
    known_fields = required_fields | {"adaptive_correction_enabled", "blink_return_count"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("settings file must be a JSON object")
        keys = set(payload.keys())
        if not required_fields.issubset(keys) or not keys.issubset(known_fields):
            raise ValueError("settings fields do not match the current schema")
        payload.setdefault("adaptive_correction_enabled", True)
        payload.setdefault("blink_return_count", 3)
        return TypingSettings(**payload)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        invalid = path.with_name(f"{path.stem}.invalid{path.suffix}")
        invalid.unlink(missing_ok=True)
        path.replace(invalid)
        return TypingSettings()


def save_settings(path: Path, settings: TypingSettings) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
