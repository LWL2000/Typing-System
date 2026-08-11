from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import secrets
from typing import Callable, Mapping

from .paths import AppPaths
from .protocol import GazeSample
from .settings import TypingSettings


class SessionRecorder:
    GAZE_FIELDS = (
        "timestamp",
        "valid",
        "face_detected",
        "blink",
        "quality",
        "fps",
        "calibration_id",
        "layout_version",
        "screen_x",
        "screen_y",
        "raw_x",
        "raw_y",
        "target_id",
        "dwell_progress",
    )
    EVENT_FIELDS = ("timestamp", "event", "payload_json")

    def __init__(
        self,
        session_dir: Path,
        settings: TypingSettings,
        context: Mapping[str, object],
        error_callback: Callable[[str], None],
    ) -> None:
        self.session_dir = session_dir
        self.settings = settings
        self.context = dict(context)
        self._error_callback = error_callback
        self.degraded = False
        self.gaze_path = session_dir / "gaze_samples.csv"
        self.events_path = session_dir / "events.csv"
        self.result_path = session_dir / "result.txt"
        self.session_path = session_dir / "session.json"
        self._gaze_handle = self.gaze_path.open("w", encoding="utf-8-sig", newline="")
        self._events_handle = self.events_path.open("w", encoding="utf-8-sig", newline="")
        self._gaze_writer = csv.DictWriter(self._gaze_handle, fieldnames=self.GAZE_FIELDS)
        self._event_writer = csv.DictWriter(self._events_handle, fieldnames=self.EVENT_FIELDS)
        self._gaze_writer.writeheader()
        self._event_writer.writeheader()
        self._gaze_handle.flush()
        self._events_handle.flush()
        self._finished = False

    @classmethod
    def start(
        cls,
        paths: AppPaths,
        settings: TypingSettings,
        context: Mapping[str, object],
        *,
        error_callback: Callable[[str], None] | None = None,
    ) -> "SessionRecorder":
        paths.sessions_dir.mkdir(parents=True, exist_ok=True)
        callback = error_callback or (lambda _message: None)
        while True:
            name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{secrets.token_hex(2)}"
            session_dir = paths.sessions_dir / name
            try:
                session_dir.mkdir()
                break
            except FileExistsError:
                continue
        return cls(session_dir, settings, context, callback)

    def record_gaze(
        self,
        sample: GazeSample,
        target_id: str | None,
        dwell_progress: float,
    ) -> bool:
        row = asdict(sample)
        row.update(target_id=target_id or "", dwell_progress=float(dwell_progress))
        return self._write(self._gaze_writer.writerow, row, self._gaze_handle)

    def record_event(self, event: str, payload: Mapping[str, object]) -> bool:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload_json": json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":")),
        }
        return self._write(self._event_writer.writerow, row, self._events_handle)

    def finish(self, final_text: str) -> bool:
        if self._finished:
            return not self.degraded
        result_tmp = self.result_path.with_suffix(".txt.tmp")
        session_tmp = self.session_path.with_suffix(".json.tmp")
        try:
            result_tmp.write_text(final_text, encoding="utf-8")
            result_tmp.replace(self.result_path)
            payload = {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "settings": asdict(self.settings),
                "context": self.context,
                "recording_degraded": self.degraded,
                "result_file": self.result_path.name,
            }
            session_tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            session_tmp.replace(self.session_path)
        except OSError as error:
            self._report_error(f"保存实验结果失败：{error}")
        finally:
            self._gaze_handle.close()
            self._events_handle.close()
            self._finished = True
        return not self.degraded

    def _write(self, writer: Callable[[dict[str, object]], object], row: dict[str, object], handle) -> bool:
        if self._finished or self.degraded:
            return False
        try:
            writer(row)
            handle.flush()
            return True
        except (OSError, ValueError) as error:
            self._report_error(f"实验记录写入失败：{error}")
            return False

    def _report_error(self, message: str) -> None:
        self.degraded = True
        self._error_callback(message)
