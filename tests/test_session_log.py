import csv
from pathlib import Path

from pure_gaze_typing.paths import AppPaths
from pure_gaze_typing.protocol import GazeSample
from pure_gaze_typing.session_log import SessionRecorder
from pure_gaze_typing.settings import TypingSettings


def test_session_recorder_creates_isolated_parseable_files(tmp_path: Path):
    errors: list[str] = []
    recorder = SessionRecorder.start(
        AppPaths.for_root(tmp_path),
        TypingSettings(),
        {"calibration_id": "cal-1"},
        error_callback=errors.append,
    )
    recorder.record_gaze(
        GazeSample(
            1.0,
            True,
            True,
            False,
            0.9,
            30.0,
            "cal-1",
            "gaze-grid-v1",
            100.0,
            200.0,
        ),
        "target_0",
        0.5,
    )
    recorder.record_event("selection", {"target_id": "letter_A"})
    assert recorder.finish("A")
    assert recorder.result_path.read_text(encoding="utf-8") == "A"
    with recorder.events_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["event"] == "selection"
    with recorder.gaze_path.open(encoding="utf-8-sig", newline="") as handle:
        gaze_rows = list(csv.DictReader(handle))
    assert gaze_rows[0]["target_id"] == "target_0"
    assert not errors


def test_two_sessions_never_share_a_directory(tmp_path: Path):
    paths = AppPaths.for_root(tmp_path)
    first = SessionRecorder.start(paths, TypingSettings(), {})
    second = SessionRecorder.start(paths, TypingSettings(), {})
    assert first.session_dir != second.session_dir
    first.finish("")
    second.finish("")
