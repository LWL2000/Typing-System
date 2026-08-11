from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    calibration_dir: Path
    logs_dir: Path
    sessions_dir: Path
    settings_file: Path

    @classmethod
    def for_root(cls, root: Path) -> "AppPaths":
        resolved = Path(root).expanduser().resolve()
        return cls(
            root=resolved,
            calibration_dir=resolved / "calibration",
            logs_dir=resolved / "logs",
            sessions_dir=resolved / "sessions",
            settings_file=resolved / "settings.json",
        )

    @classmethod
    def default(cls) -> "AppPaths":
        app_data = os.environ.get("APPDATA")
        if not app_data:
            app_data = str(Path.home() / "AppData" / "Roaming")
        return cls.for_root(Path(app_data) / "PureGazeTyping")

    def ensure(self) -> None:
        for directory in (self.root, self.calibration_dir, self.logs_dir, self.sessions_dir):
            directory.mkdir(parents=True, exist_ok=True)
