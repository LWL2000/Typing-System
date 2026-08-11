from pathlib import Path

import numpy as np
import pytest

from pure_gaze_typing.calibration import (
    CalibrationEnvironment,
    CalibrationMode,
    CalibrationMetadata,
    CalibrationPoint,
    CalibrationSession,
    CalibrationStore,
    apply_screen_affine,
    balance_point_samples,
    calibration_profile,
    filter_stable_features,
    fit_screen_affine,
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
    for index in range(24):
        session.add_frame(
            index / 30.0,
            np.array([index], dtype=float),
            blink=False,
            face_detected=True,
        )
    assert session.current_point_id == "target_0"
    session.add_frame(0.8, np.array([24.0]), blink=False, face_detected=True)
    assert session.complete
    features, labels = session.training_data()
    assert features.shape == (17, 1)
    assert features[0, 0] == 8.0
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


def test_calibration_profiles_offer_short_single_pass_and_reference_precision():
    fast = calibration_profile(CalibrationMode.FAST)
    precise = calibration_profile(CalibrationMode.PRECISE)

    assert fast.target_passes == 1
    assert fast.estimated_seconds == 30
    assert precise.target_passes == 2
    assert precise.reverse_second_pass
    assert precise.estimated_seconds == 55
    assert fast.capture_seconds < precise.capture_seconds


def test_filter_stable_features_rejects_outlier_and_caps_temporal_samples():
    cluster = np.column_stack((np.linspace(0.0, 0.2, 30), np.linspace(1.0, 1.2, 30)))
    samples = np.vstack((cluster, np.array([[100.0, -100.0]])))

    stable = filter_stable_features(samples, max_samples=12, min_samples=8)

    assert stable.shape == (12, 2)
    assert np.max(np.abs(stable)) < 10.0
    assert stable[0, 0] < stable[-1, 0]


def test_balance_point_samples_uses_equal_counts_for_each_target():
    groups = (
        ((100.0, 200.0), np.arange(10, dtype=float).reshape(5, 2)),
        ((300.0, 400.0), np.arange(6, dtype=float).reshape(3, 2)),
    )

    features, labels = balance_point_samples(groups)

    assert features.shape == (6, 2)
    assert labels.tolist() == [[100.0, 200.0]] * 3 + [[300.0, 400.0]] * 3


def test_screen_affine_recovers_translation_scale_and_tilt():
    predictions = np.array(
        [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]],
        dtype=float,
    )
    expected_coefficients = np.array(
        [[1.2, 0.1], [-0.05, 0.9], [40.0, -25.0]],
        dtype=float,
    )
    targets = np.column_stack((predictions, np.ones(4))) @ expected_coefficients

    coefficients = fit_screen_affine(predictions, targets)

    assert coefficients == pytest.approx(expected_coefficients)
    assert apply_screen_affine(coefficients, 25.0, 75.0) == pytest.approx(
        tuple(np.array([25.0, 75.0, 1.0]) @ expected_coefficients)
    )


def test_store_round_trip_preserves_reference_pipeline_metadata(tmp_path: Path):
    environment = CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v3-reference")
    metadata = CalibrationMetadata(
        "cal-reference",
        "2026-08-11T12:00:00+08:00",
        environment,
        (0.0, 1.0),
        (2.0, 3.0),
        calibration_mode="precise",
        pipeline_version="reference-v1",
        feature_range_threshold=4.5,
        screen_affine=((1.0, 0.0), (0.0, 1.0), (12.0, -8.0)),
        validation_hits=5,
        validation_total=6,
    )

    store = CalibrationStore(tmp_path)
    store.save(FakeModel(), metadata)

    assert store.load(environment).metadata == metadata


def test_session_prepares_stable_balanced_point_samples():
    session = CalibrationSession(
        (
            CalibrationPoint("first", 100.0, 200.0, 0.1),
            CalibrationPoint("second", 300.0, 400.0, 0.1),
        ),
        settle_seconds=0.0,
        min_valid_frames=3,
    )
    for timestamp, value in ((0.0, 0.0), (0.04, 0.1), (0.08, 99.0), (0.11, 0.2)):
        session.add_frame(timestamp, np.array([value]), blink=False, face_detected=True)
    for timestamp, value in ((0.2, 1.0), (0.24, 1.1), (0.28, 1.2), (0.31, 1.3)):
        session.add_frame(timestamp, np.array([value]), blink=False, face_detected=True)

    features, labels = session.prepared_training_data(max_samples_per_point=3, min_samples=3)

    assert features.shape == (6, 1)
    assert np.max(features[:3]) < 10.0
    assert labels.tolist() == [[100.0, 200.0]] * 3 + [[300.0, 400.0]] * 3
