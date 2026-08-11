from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
import time
from typing import Callable

import numpy as np

from .calibration import (
    CalibrationMetadata,
    StoredCalibration,
    apply_screen_affine,
)


LOGGER = logging.getLogger("pure_gaze_typing.capture")


def _create_face_landmarker_from_buffer(*, model_path):
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    task_path = Path(model_path).expanduser().resolve()
    model_bytes = task_path.read_bytes()
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def _create_default_estimator(face_model_path: Path):
    from eyetrax import gaze
    from eyetrax.models.ridge import RidgeModel

    original_landmarker_factory = gaze._create_face_landmarker
    original_model_factory = gaze.create_model
    gaze._create_face_landmarker = _create_face_landmarker_from_buffer
    gaze.create_model = lambda name, **kwargs: (
        RidgeModel(**kwargs)
        if name == "ridge"
        else original_model_factory(name, **kwargs)
    )
    try:
        return gaze.GazeEstimator(
            model_name="ridge",
            face_landmarker_model=str(face_model_path),
        )
    finally:
        gaze._create_face_landmarker = original_landmarker_factory
        gaze.create_model = original_model_factory


@dataclass(frozen=True)
class FrameObservation:
    features: np.ndarray | None
    face_detected: bool
    blink: bool


@dataclass(frozen=True)
class GazeEstimate:
    valid: bool
    face_detected: bool
    blink: bool
    quality: float
    raw_x: float | None = None
    raw_y: float | None = None
    screen_x: float | None = None
    screen_y: float | None = None


def resource_path(relative: str) -> Path:
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS"))
    else:
        base = Path(__file__).resolve().parents[2]
    return base / "resources" / relative


class EyeTraxRuntime:
    def __init__(
        self,
        face_model_path: Path,
        screen_width: int,
        screen_height: int,
        *,
        estimator_factory: Callable[..., object] | None = None,
        smoother_factory: Callable[[], object] | None = None,
        min_quality: float = 0.25,
    ) -> None:
        if estimator_factory is None:
            LOGGER.info("eyetrax_import_gaze_begin")
            from eyetrax import gaze as _gaze
            LOGGER.info("eyetrax_import_gaze_complete")
        if smoother_factory is None:
            LOGGER.info("eyetrax_import_smoother_begin")
            from eyetrax.filters import KalmanEMASmoother, make_kalman

            smoother_factory = lambda: KalmanEMASmoother(make_kalman(), ema_alpha=0.35)
            LOGGER.info("eyetrax_import_smoother_complete")
        LOGGER.info("eyetrax_estimator_create_begin model=%s", face_model_path)
        if estimator_factory is None:
            self._estimator = _create_default_estimator(face_model_path)
        else:
            self._estimator = estimator_factory(
                model_name="ridge",
                face_landmarker_model=str(face_model_path),
            )
        LOGGER.info("eyetrax_estimator_create_complete")
        self._smoother_factory = smoother_factory
        self._smoother = smoother_factory()
        LOGGER.info("eyetrax_smoother_create_complete")
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.min_quality = float(min_quality)
        self.metadata: CalibrationMetadata | None = None
        self._invalid_since: float | None = None
        self._reset_smoother_on_valid = False

    @property
    def estimator(self) -> object:
        return self._estimator

    def extract(self, frame: np.ndarray) -> FrameObservation:
        features, blink = self._estimator.extract_features(frame)
        return FrameObservation(
            None if features is None else np.asarray(features, dtype=float).reshape(-1),
            features is not None,
            bool(blink),
        )

    def set_metadata(self, metadata: CalibrationMetadata) -> None:
        self.metadata = metadata
        model = getattr(self._estimator, "model", None)
        if model is not None:
            model.calibration_kind = metadata.pipeline_version
            model.feature_range_threshold = metadata.feature_range_threshold
            model.screen_affine = (
                None if not metadata.screen_affine else np.asarray(metadata.screen_affine, dtype=float)
            )

    def estimate(
        self,
        observation: FrameObservation,
        *,
        timestamp: float | None = None,
    ) -> GazeEstimate:
        now = time.monotonic() if timestamp is None else float(timestamp)
        if not observation.face_detected or observation.features is None:
            self._mark_invalid(now)
            return GazeEstimate(False, False, False, 0.0)
        if observation.blink:
            self._mark_invalid(now)
            return GazeEstimate(False, True, True, 0.0)
        if self.metadata is None:
            return GazeEstimate(False, True, False, 0.0)

        self._mark_valid()
        quality, hard_reject = self._quality_and_range(observation.features)
        if hard_reject:
            return GazeEstimate(False, True, False, quality)
        prediction = np.asarray(
            self._estimator.predict(observation.features.reshape(1, -1)),
            dtype=float,
        ).reshape(-1, 2)[0]
        raw_x, raw_y = map(float, prediction)
        if not np.isfinite(raw_x) or not np.isfinite(raw_y):
            return GazeEstimate(False, True, False, quality)
        corrected_x, corrected_y = raw_x, raw_y
        if self.metadata.screen_affine:
            corrected_x, corrected_y = apply_screen_affine(
                self.metadata.screen_affine,
                raw_x,
                raw_y,
            )
        smooth_x, smooth_y = self._smoother.step(round(corrected_x), round(corrected_y))
        screen_x = min(max(float(smooth_x), 0.0), float(self.screen_width - 1))
        screen_y = min(max(float(smooth_y), 0.0), float(self.screen_height - 1))
        return GazeEstimate(
            True,
            True,
            False,
            quality,
            raw_x,
            raw_y,
            screen_x,
            screen_y,
        )

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> GazeEstimate:
        return self.estimate(self.extract(frame), timestamp=timestamp)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._estimator.train(np.asarray(features, dtype=float), np.asarray(labels, dtype=float))
        self._reset_smoothing()

    def load_calibration(self, stored: StoredCalibration) -> None:
        self._estimator.load_model(stored.model_path)
        self.set_metadata(stored.metadata)
        self._reset_smoothing()

    def predict_raw(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self._estimator.predict(np.asarray(features, dtype=float)), dtype=float)

    def feature_range_threshold(self, training_features: np.ndarray) -> float:
        model = getattr(self._estimator, "model", None)
        scaler = getattr(model, "scaler", None)
        if scaler is None:
            raise RuntimeError("trained gaze model does not expose a feature scaler")
        scaled = scaler.transform(np.asarray(training_features, dtype=float))
        per_sample = np.percentile(np.abs(scaled), 95, axis=1)
        return max(4.0, float(np.percentile(per_sample, 99)) * 1.5)

    def save_model(self, path: Path) -> None:
        self._estimator.save_model(path)

    def close(self) -> None:
        self._estimator.close()

    def _quality_and_range(self, features: np.ndarray) -> tuple[float, bool]:
        assert self.metadata is not None
        threshold = self.metadata.feature_range_threshold
        model = getattr(self._estimator, "model", None)
        scaler = getattr(model, "scaler", None)
        if threshold is not None and scaler is not None:
            scaled = scaler.transform(np.asarray(features, dtype=float).reshape(1, -1))[0]
            score = float(np.percentile(np.abs(scaled), 95))
            if score > threshold * 2.0:
                return 0.0, True
            if score <= threshold:
                return 1.0, False
            quality = max(0.0, 1.0 - (score - threshold) / max(threshold, 1e-6))
            return float(quality), False

        lower = np.asarray(self.metadata.feature_min, dtype=float)
        upper = np.asarray(self.metadata.feature_max, dtype=float)
        vector = np.asarray(features, dtype=float).reshape(-1)
        if vector.size != lower.size or lower.size != upper.size or not np.all(np.isfinite(vector)):
            return 0.0, True
        span = np.maximum(upper - lower, 1e-6)
        below = np.maximum(lower - vector, 0.0) / span
        above = np.maximum(vector - upper, 0.0) / span
        excess = float(np.max(np.maximum(below, above), initial=0.0))
        quality = max(0.0, min(1.0, 1.0 - excess))
        return quality, quality < self.min_quality

    def _mark_invalid(self, timestamp: float) -> None:
        if self._invalid_since is None:
            self._invalid_since = timestamp
        elif timestamp - self._invalid_since >= 0.8:
            self._reset_smoother_on_valid = True

    def _mark_valid(self) -> None:
        if self._reset_smoother_on_valid:
            self._reset_smoothing()
        self._invalid_since = None
        self._reset_smoother_on_valid = False

    def _reset_smoothing(self) -> None:
        self._smoother = self._smoother_factory()
        self._invalid_since = None
        self._reset_smoother_on_valid = False


class CenterDriftCorrector:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        *,
        max_x_ratio: float = 0.12,
        max_y_ratio: float = 0.15,
    ) -> None:
        self._max_x = float(screen_width) * max_x_ratio
        self._max_y = float(screen_height) * max_y_ratio
        self._points: list[tuple[float, float]] = []
        self.offset = (0.0, 0.0)
        self.measured_center: tuple[float, float] | None = None

    def collect(self, x: float, y: float) -> None:
        if np.isfinite(x) and np.isfinite(y):
            self._points.append((float(x), float(y)))

    def finish(self, expected_center: tuple[float, float]) -> tuple[float, float]:
        if not self._points:
            self.offset = (0.0, 0.0)
            return self.offset
        values = np.asarray(self._points, dtype=float)
        measured = tuple(map(float, np.median(values, axis=0)))
        self.measured_center = measured
        raw_x = expected_center[0] - measured[0]
        raw_y = expected_center[1] - measured[1]
        self.offset = (
            min(max(raw_x, -self._max_x), self._max_x),
            min(max(raw_y, -self._max_y), self._max_y),
        )
        return self.offset

    def apply(self, x: float, y: float) -> tuple[float, float]:
        return x + self.offset[0], y + self.offset[1]
