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

    def __post_init__(self) -> None:
        if not isinstance(self.show_gaze_point, bool):
            raise ValueError("show_gaze_point must be a boolean")
        if not isinstance(self.validate_calibration, bool):
            raise ValueError("validate_calibration must be a boolean")
        if not isinstance(self.adaptive_correction_enabled, bool):
            raise ValueError("adaptive_correction_enabled must be a boolean")
        if not math.isfinite(self.dwell_seconds) or not 0.5 <= self.dwell_seconds <= 3.0:
            raise ValueError("dwell_seconds must be between 0.5 and 3.0")
        object.__setattr__(self, "dwell_seconds", round(float(self.dwell_seconds), 1))


def load_settings(path: Path) -> TypingSettings:
    path = Path(path)
    if not path.exists():
        return TypingSettings()
    old_fields = {"show_gaze_point", "dwell_seconds", "validate_calibration"}
    new_fields = old_fields | {"adaptive_correction_enabled"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("settings file must be a JSON object")
        keys = set(payload.keys())
        if keys == old_fields:
            payload["adaptive_correction_enabled"] = True
        elif keys != new_fields:
            raise ValueError("settings fields do not match the current schema")
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
