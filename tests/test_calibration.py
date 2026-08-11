from pathlib import Path

import numpy as np

from pure_gaze_typing.calibration import (
    CalibrationEnvironment,
    CalibrationMetadata,
    CalibrationPoint,
    CalibrationSession,
    CalibrationStore,
    score_validation,
)


class FakeModel:
    def save_model(self, path: Path) -> None:
        Path(path).write_bytes(b"model")


def test_fast_session_advances_after_duration_and_minimum_frames():
    session = CalibrationSession(
        (CalibrationPoint("target_0", 100.0, 200.0, 0.8),),
        min_valid_frames=12,
    )
    for index in range(11):
        session.add_frame(
            index * 0.08,
            np.array([index], dtype=float),
            blink=False,
            face_detected=True,
        )
    assert session.current_point_id == "target_0"
    session.add_frame(0.88, np.array([12.0]), blink=False, face_detected=True)
    assert session.complete
    features, labels = session.training_data()
    assert features.shape == (12, 1)
    assert np.all(labels == np.array([100.0, 200.0]))


def test_invalid_frames_are_ignored_and_point_blocks_after_three_seconds():
    session = CalibrationSession(
        (CalibrationPoint("target_0", 100.0, 200.0, 0.8),),
        min_valid_frames=2,
        max_point_seconds=3.0,
    )
    session.add_frame(0.0, None, blink=False, face_detected=False)
    session.add_frame(3.1, np.array([1.0]), blink=True, face_detected=True)
    assert session.blocked_reason == "insufficient_valid_frames"
    session.resume_current_point(4.0)
    assert session.blocked_reason is None


def test_validation_is_optional_and_reports_failed_regions():
    result = score_validation(
        {
            "target_0": 4,
            "target_1": 0,
            "target_2": 3,
            "target_3": 5,
            "target_4": 2,
            "target_5": 4,
        }
    )
    assert result.hit_count == 5
    assert result.passed
    assert result.failed_target_ids == ("target_1",)


def test_store_round_trip_checks_environment(tmp_path: Path):
    environment = CalibrationEnvironment(1920, 1080, 1.25, 0, "gaze-grid-v1")
    metadata = CalibrationMetadata(
        "cal-1",
        "2026-08-11T12:00:00+08:00",
        environment,
        (0.0, 1.0),
        (2.0, 3.0),
    )
    store = CalibrationStore(tmp_path)
    store.save(FakeModel(), metadata)
    loaded = store.load(environment)
    assert loaded is not None
    assert loaded.metadata == metadata
    assert loaded.model_path.read_bytes() == b"model"
    mismatch = CalibrationEnvironment(1280, 720, 1.25, 0, "gaze-grid-v1")
    assert store.load(mismatch) is None
    assert "分辨率" in store.compatibility(mismatch).reason
