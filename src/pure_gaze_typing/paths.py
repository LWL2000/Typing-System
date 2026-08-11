from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import sys


PACKAGE_DIR_NAME = "纯眼动打字系统"
PORTABLE_DATA_DIR_NAME = "纯眼动打字系统数据"


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
    def default(
        cls,
        *,
        executable: Path | None = None,
        frozen: bool | None = None,
    ) -> "AppPaths":
        is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else bool(frozen)
        if is_frozen:
            executable_path = Path(sys.executable if executable is None else executable).resolve()
            package_dir = next(
                (parent for parent in executable_path.parents if parent.name == PACKAGE_DIR_NAME),
                None,
            )
            release_dir = package_dir.parent if package_dir is not None else executable_path.parent
            return cls.for_root(release_dir / PORTABLE_DATA_DIR_NAME)

        app_data = os.environ.get("APPDATA")
        if not app_data:
            app_data = str(Path.home() / "AppData" / "Roaming")
        return cls.for_root(Path(app_data) / "PureGazeTyping")

    def migrate_legacy(self, legacy_root: Path | None = None) -> bool:
        source = self._legacy_root() if legacy_root is None else Path(legacy_root).resolve()
        if source == self.root or not source.is_dir():
            return False
        if self.root.exists() and any(self.root.iterdir()):
            return False

        copied = False
        self.root.mkdir(parents=True, exist_ok=True)
        for source_path in source.rglob("*"):
            relative = source_path.relative_to(source)
            target_path = self.root / relative
            if source_path.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
            elif not target_path.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                copied = True
        return copied

    @staticmethod
    def _legacy_root() -> Path:
        app_data = os.environ.get("APPDATA")
        if not app_data:
            app_data = str(Path.home() / "AppData" / "Roaming")
        return (Path(app_data) / "PureGazeTyping").resolve()

    def ensure(self) -> None:
        for directory in (self.root, self.calibration_dir, self.logs_dir, self.sessions_dir):
            directory.mkdir(parents=True, exist_ok=True)
