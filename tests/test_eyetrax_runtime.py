from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import pure_gaze_typing.eyetrax_runtime as runtime_module

from pure_gaze_typing.calibration import (
    CalibrationEnvironment,
    CalibrationMetadata,
    StoredCalibration,
)
from pure_gaze_typing.eyetrax_runtime import (
    CenterDriftCorrector,
    EyeTraxRuntime,
    FrameObservation,
)


class IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class FakeEstimator:
    def __init__(
        self,
        features=None,
        blink=False,
        prediction=(50.0, 60.0),
        scaler=None,
    ):
        self.features = features
        self.blink = blink
        self.prediction = prediction
        self.predict_calls = 0
        self.trained = None
        self.loaded_path = None
        self.closed = False
        self.model = SimpleNamespace(scaler=scaler)

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


def make_reference_metadata(
    *,
    threshold: float = 4.0,
    affine=((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)),
) -> CalibrationMetadata:
    return CalibrationMetadata(
        "cal-reference",
        "2026-08-11T12:00:00+08:00",
        CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v3-reference"),
        (0.0, 0.0),
        (2.0, 2.0),
        calibration_mode="precise",
        pipeline_version="reference-v1",
        feature_range_threshold=threshold,
        screen_affine=affine,
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
    estimator = FakeEstimator(features=np.array([1.0, 1.0]), prediction=(1950.0, -10.0))
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


def test_extreme_offscreen_prediction_is_clamped_after_smoothing():
    estimator = FakeEstimator(features=np.array([1.0, 1.0]), prediction=(-500.0, 500.0))
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
    assert (estimate.screen_x, estimate.screen_y) == (0.0, 500.0)


def test_default_runtime_matches_reference_ema_smoothing():
    estimator = FakeEstimator(features=np.array([1.0, 1.0]))
    runtime = EyeTraxRuntime(
        Path("face.task"),
        1920,
        1080,
        estimator_factory=lambda **_kwargs: estimator,
    )

    assert runtime._smoother.ema_alpha == pytest.approx(0.35)


def test_reference_affine_is_applied_before_smoothing():
    estimator = FakeEstimator(
        features=np.array([1.0, 1.0]),
        prediction=(100.0, 200.0),
        scaler=IdentityScaler(),
    )

    class RecordingSmoother:
        def __init__(self):
            self.input = None

        def step(self, x, y):
            self.input = (x, y)
            return x, y

    smoother = RecordingSmoother()
    runtime = EyeTraxRuntime(
        Path("face.task"),
        1920,
        1080,
        estimator_factory=lambda **_kwargs: estimator,
        smoother_factory=lambda: smoother,
    )
    runtime.set_metadata(
        make_reference_metadata(
            affine=((2.0, 0.0), (0.0, 0.5), (10.0, -20.0)),
        )
    )

    estimate = runtime.estimate(FrameObservation(estimator.features, True, False), timestamp=1.0)

    assert smoother.input == (210, 80)
    assert (estimate.raw_x, estimate.raw_y) == (100.0, 200.0)
    assert (estimate.screen_x, estimate.screen_y) == (210.0, 80.0)


def test_reference_feature_range_softens_quality_then_hard_rejects():
    estimator = FakeEstimator(prediction=(100.0, 200.0), scaler=IdentityScaler())
    runtime = EyeTraxRuntime(
        Path("face.task"),
        1920,
        1080,
        estimator_factory=lambda **_kwargs: estimator,
        smoother_factory=IdentitySmoother,
    )
    runtime.set_metadata(make_reference_metadata(threshold=4.0))

    soft = runtime.estimate(FrameObservation(np.array([6.0, 6.0]), True, False), timestamp=1.0)
    hard = runtime.estimate(FrameObservation(np.array([9.0, 9.0]), True, False), timestamp=2.0)

    assert soft.valid
    assert soft.quality == pytest.approx(0.5)
    assert not hard.valid
    assert hard.quality == 0.0


def test_smoother_resets_after_reference_invalid_interval():
    estimator = FakeEstimator(prediction=(100.0, 200.0), scaler=IdentityScaler())
    created = []

    class NumberedSmoother(IdentitySmoother):
        def __init__(self):
            created.append(self)

    runtime = EyeTraxRuntime(
        Path("face.task"),
        1920,
        1080,
        estimator_factory=lambda **_kwargs: estimator,
        smoother_factory=NumberedSmoother,
    )
    runtime.set_metadata(make_reference_metadata())
    runtime.estimate(FrameObservation(np.array([1.0, 1.0]), True, False), timestamp=1.0)
    runtime.estimate(FrameObservation(None, False, False), timestamp=1.1)
    runtime.estimate(FrameObservation(None, False, False), timestamp=2.0)
    runtime.estimate(FrameObservation(np.array([1.0, 1.0]), True, False), timestamp=2.1)

    assert len(created) == 2


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


def test_default_center_drift_allows_practical_vertical_correction():
    corrector = CenterDriftCorrector(1920, 1080)
    corrector.collect(960.0, 300.0)

    offset = corrector.finish((960.0, 540.0))

    assert offset == pytest.approx((0.0, 162.0))


def test_default_runtime_loads_face_model_from_unicode_path(tmp_path: Path):
    source = Path.home() / ".cache" / "eyetrax" / "mediapipe" / "face_landmarker.task"
    if not source.is_file():
        pytest.skip("EyeTrax FaceLandmarker model is not available")
    model = tmp_path / "中文模型" / "face_landmarker.task"
    model.parent.mkdir()
    shutil.copyfile(source, model)

    runtime = EyeTraxRuntime(model, 1920, 1080)
    runtime.close()


def test_default_runtime_does_not_use_eyetrax_model_directory_discovery(
    monkeypatch, tmp_path: Path
):
    from eyetrax import gaze

    class FakeLandmarker:
        def close(self):
            return None

    monkeypatch.setattr(
        runtime_module,
        "_create_face_landmarker_from_buffer",
        lambda **_kwargs: (object(), FakeLandmarker()),
    )
    monkeypatch.setattr(
        gaze,
        "create_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("eyetrax/models is not a physical directory")
        ),
    )

    runtime = EyeTraxRuntime(tmp_path / "face_landmarker.task", 1920, 1080)
    runtime.close()
