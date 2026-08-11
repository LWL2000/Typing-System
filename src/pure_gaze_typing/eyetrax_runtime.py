from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np

from .calibration import CalibrationMetadata, StoredCalibration


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
            from eyetrax.gaze import GazeEstimator

            estimator_factory = GazeEstimator
        if smoother_factory is None:
            from eyetrax.filters import KalmanEMASmoother

            smoother_factory = KalmanEMASmoother
        self._estimator = estimator_factory(
            model_name="ridge",
            face_landmarker_model=str(face_model_path),
        )
        self._smoother_factory = smoother_factory
        self._smoother = smoother_factory()
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.min_quality = float(min_quality)
        self.metadata: CalibrationMetadata | None = None

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

    def estimate(self, observation: FrameObservation) -> GazeEstimate:
        if not observation.face_detected or observation.features is None:
            return GazeEstimate(False, False, False, 0.0)
        if observation.blink:
            return GazeEstimate(False, True, True, 0.0)
        if self.metadata is None:
            return GazeEstimate(False, True, False, 0.0)
        quality = self._quality(observation.features)
        if quality < self.min_quality:
            return GazeEstimate(False, True, False, quality)
        prediction = np.asarray(
            self._estimator.predict(observation.features.reshape(1, -1)),
            dtype=float,
        ).reshape(-1, 2)[0]
        raw_x, raw_y = map(float, prediction)
        smooth_x, smooth_y = self._smoother.step(round(raw_x), round(raw_y))
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

    def process_frame(self, frame: np.ndarray) -> GazeEstimate:
        return self.estimate(self.extract(frame))

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        self._estimator.train(np.asarray(features, dtype=float), np.asarray(labels, dtype=float))
        self._smoother = self._smoother_factory()

    def load_calibration(self, stored: StoredCalibration) -> None:
        self._estimator.load_model(stored.model_path)
        self.metadata = stored.metadata
        self._smoother = self._smoother_factory()

    def save_model(self, path: Path) -> None:
        self._estimator.save_model(path)

    def close(self) -> None:
        self._estimator.close()

    def _quality(self, features: np.ndarray) -> float:
        assert self.metadata is not None
        lower = np.asarray(self.metadata.feature_min, dtype=float)
        upper = np.asarray(self.metadata.feature_max, dtype=float)
        vector = np.asarray(features, dtype=float).reshape(-1)
        if vector.size != lower.size or lower.size != upper.size or not np.all(np.isfinite(vector)):
            return 0.0
        span = np.maximum(upper - lower, 1e-6)
        below = np.maximum(lower - vector, 0.0) / span
        above = np.maximum(vector - upper, 0.0) / span
        excess = float(np.max(np.maximum(below, above), initial=0.0))
        return max(0.0, min(1.0, 1.0 - excess))


class CenterDriftCorrector:
    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        *,
        max_x_ratio: float = 0.08,
        max_y_ratio: float = 0.05,
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
