from pathlib import Path

import numpy as np
import pytest

from pure_gaze_typing.calibration import (
    CalibrationEnvironment,
    CalibrationMetadata,
    StoredCalibration,
)
from pure_gaze_typing.eyetrax_runtime import CenterDriftCorrector, EyeTraxRuntime


class FakeEstimator:
    def __init__(self, features=None, blink=False, prediction=(50.0, 60.0)):
        self.features = features
        self.blink = blink
        self.prediction = prediction
        self.predict_calls = 0
        self.trained = None
        self.loaded_path = None
        self.closed = False

    def extract_features(self, _frame):
        return self.features, self.blink

    def predict(self, _features):
        self.predict_calls += 1
        return np.asarray([self.prediction], dtype=float)

    def train(self, features, labels):
        self.trained = (features.copy(), labels.copy())

    def load_model(self, path):
        self.loaded_path = Path(path)

    def save_model(self, path):
        Path(path).write_bytes(b"fake")

    def close(self):
        self.closed = True


class IdentitySmoother:
    def step(self, x, y):
        return x, y


def make_metadata() -> CalibrationMetadata:
    return CalibrationMetadata(
        "cal-1",
        "2026-08-11T12:00:00+08:00",
        CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v1"),
        (0.0, 0.0),
        (2.0, 2.0),
    )


def test_blink_frame_is_invalid_and_never_predicted():
    estimator = FakeEstimator(features=np.array([1.0, 2.0]), blink=True)
    runtime = EyeTraxRuntime(
        Path("face.task"),
        1920,
        1080,
        estimator_factory=lambda **_kwargs: estimator,
        smoother_factory=IdentitySmoother,
    )
    estimate = runtime.process_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    assert not estimate.valid
    assert estimate.blink
    assert estimator.predict_calls == 0


def test_in_range_prediction_is_smoothed_and_clamped():
    estimator = FakeEstimator(features=np.array([1.0, 1.0]), prediction=(2000.0, -10.0))
    runtime = EyeTraxRuntime(
        Path("face.task"),
        1920,
        1080,
        estimator_factory=lambda **_kwargs: estimator,
        smoother_factory=IdentitySmoother,
    )
    runtime.set_metadata(make_metadata())
    estimate = runtime.process_frame(np.zeros((2, 2, 3), dtype=np.uint8))
    assert estimate.valid
    assert estimate.quality == 1.0
    assert (estimate.screen_x, estimate.screen_y) == (1919.0, 0.0)


def test_low_quality_feature_is_rejected():
    estimator = FakeEstimator(features=np.array([3.6, 1.0]))
    runtime = EyeTraxRuntime(
        Path("face.task"),
        1920,
        1080,
        estimator_factory=lambda **_kwargs: estimator,
        smoother_factory=IdentitySmoother,
    )
    runtime.set_metadata(make_metadata())
    estimate = runtime.process_frame(np.zeros((2, 2, 3), dtype=np.uint8))
    assert not estimate.valid
    assert estimate.quality < 0.25
    assert estimator.predict_calls == 0


def test_load_and_train_delegate_to_estimator(tmp_path: Path):
    estimator = FakeEstimator()
    runtime = EyeTraxRuntime(
        Path("face.task"),
        1920,
        1080,
        estimator_factory=lambda **_kwargs: estimator,
        smoother_factory=IdentitySmoother,
    )
    stored = StoredCalibration(tmp_path / "model.pkl", make_metadata())
    runtime.load_calibration(stored)
    assert estimator.loaded_path == stored.model_path
    runtime.train(np.ones((2, 2)), np.ones((2, 2)))
    assert estimator.trained is not None


def test_center_drift_is_median_based_and_capped():
    corrector = CenterDriftCorrector(1920, 1080, max_x_ratio=0.08, max_y_ratio=0.05)
    for point in [(1200.0, 700.0), (1204.0, 698.0), (5000.0, 5000.0)]:
        corrector.collect(*point)
    offset = corrector.finish((960.0, 540.0))
    assert offset == pytest.approx((-153.6, -54.0))
    assert corrector.apply(1000.0, 600.0) == pytest.approx((846.4, 546.0))
