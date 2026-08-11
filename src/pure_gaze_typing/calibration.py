from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
from pathlib import Path
import shutil
from typing import Mapping, Protocol, Sequence

import numpy as np


REFERENCE_PIPELINE_VERSION = "reference-v1"


class CalibrationMode(str, Enum):
    FAST = "fast"
    PRECISE = "precise"


@dataclass(frozen=True)
class CalibrationProfile:
    mode: CalibrationMode
    pose_hold_seconds: float
    settle_seconds: float
    capture_seconds: float
    max_samples_per_point: int
    target_passes: int
    reverse_second_pass: bool
    validation_settle_seconds: float
    validation_capture_seconds: float
    estimated_seconds: int


def calibration_profile(mode: CalibrationMode | str) -> CalibrationProfile:
    selected = CalibrationMode(mode)
    if selected is CalibrationMode.PRECISE:
        return CalibrationProfile(
            mode=selected,
            pose_hold_seconds=1.5,
            settle_seconds=0.65,
            capture_seconds=0.90,
            max_samples_per_point=24,
            target_passes=2,
            reverse_second_pass=True,
            validation_settle_seconds=0.45,
            validation_capture_seconds=0.60,
            estimated_seconds=55,
        )
    return CalibrationProfile(
        mode=selected,
        pose_hold_seconds=0.8,
        settle_seconds=0.35,
        capture_seconds=0.55,
        max_samples_per_point=18,
        target_passes=1,
        reverse_second_pass=False,
        validation_settle_seconds=0.25,
        validation_capture_seconds=0.45,
        estimated_seconds=30,
    )


@dataclass(frozen=True)
class CalibrationEnvironment:
    screen_width: int
    screen_height: int
    scale_factor: float
    camera_index: int
    layout_version: str

    def __post_init__(self) -> None:
        if self.screen_width <= 0 or self.screen_height <= 0:
            raise ValueError("screen dimensions must be positive")
        if not math.isfinite(self.scale_factor) or self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive and finite")


@dataclass(frozen=True)
class CalibrationMetadata:
    calibration_id: str
    created_at: str
    environment: CalibrationEnvironment
    feature_min: tuple[float, ...]
    feature_max: tuple[float, ...]
    calibration_mode: str = CalibrationMode.FAST.value
    pipeline_version: str = REFERENCE_PIPELINE_VERSION
    feature_range_threshold: float | None = None
    screen_affine: tuple[tuple[float, float], ...] = ()
    validation_hits: int | None = None
    validation_total: int | None = None

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id is required")
        if len(self.feature_min) != len(self.feature_max):
            raise ValueError("feature bounds must have equal lengths")
        CalibrationMode(self.calibration_mode)
        if self.feature_range_threshold is not None and self.feature_range_threshold <= 0:
            raise ValueError("feature range threshold must be positive")
        if self.screen_affine and (len(self.screen_affine) != 3 or any(len(row) != 2 for row in self.screen_affine)):
            raise ValueError("screen affine coefficients must be 3x2")


@dataclass(frozen=True)
class CalibrationPoint:
    target_id: str
    screen_x: float
    screen_y: float
    duration_seconds: float

    def __post_init__(self) -> None:
        if not self.target_id or self.duration_seconds <= 0:
            raise ValueError("calibration point requires an id and positive duration")


@dataclass(frozen=True)
class ValidationResult:
    hit_count: int
    total_count: int
    failed_target_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.hit_count >= max(1, self.total_count - 1)


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    reason: str
    calibration_id: str | None = None


@dataclass(frozen=True)
class StoredCalibration:
    model_path: Path
    metadata: CalibrationMetadata


class SavableModel(Protocol):
    def save_model(self, path: Path) -> None: ...


class CalibrationSession:
    def __init__(
        self,
        points: tuple[CalibrationPoint, ...],
        *,
        center_seconds: float = 1.0,
        target_seconds: float = 0.8,
        settle_seconds: float = 0.25,
        min_valid_frames: int = 12,
        max_point_seconds: float = 3.0,
    ) -> None:
        del center_seconds, target_seconds
        if not points:
            raise ValueError("calibration requires at least one point")
        if min_valid_frames < 1 or max_point_seconds <= 0 or settle_seconds < 0:
            raise ValueError("invalid calibration timing")
        self.points = tuple(points)
        self.settle_seconds = float(settle_seconds)
        self.min_valid_frames = int(min_valid_frames)
        self.max_point_seconds = float(max_point_seconds)
        self._point_index = 0
        self._point_started_at: float | None = None
        self._features: list[list[np.ndarray]] = [[] for _ in self.points]
        self._feature_width: int | None = None
        self.blocked_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self._point_index >= len(self.points)

    @property
    def current_point_id(self) -> str | None:
        return None if self.complete else self.points[self._point_index].target_id

    def add_frame(
        self,
        timestamp: float,
        features: np.ndarray | None,
        *,
        blink: bool,
        face_detected: bool,
    ) -> None:
        if self.complete or self.blocked_reason:
            return
        now = float(timestamp)
        if self._point_started_at is None:
            self._point_started_at = now
        elapsed = now - self._point_started_at
        accepted = False
        if (
            elapsed >= self.settle_seconds
            and face_detected
            and not blink
            and features is not None
        ):
            vector = np.asarray(features, dtype=float).reshape(-1)
            if vector.size and np.all(np.isfinite(vector)):
                if self._feature_width is None:
                    self._feature_width = int(vector.size)
                if vector.size != self._feature_width:
                    raise ValueError("feature width changed during calibration")
                self._features[self._point_index].append(vector.copy())
                accepted = True
        point = self.points[self._point_index]
        if elapsed >= point.duration_seconds and len(self._features[self._point_index]) >= self.min_valid_frames:
            self._point_index += 1
            self._point_started_at = None
            return
        if elapsed >= self.max_point_seconds and not accepted:
            self.blocked_reason = "insufficient_valid_frames"

    def resume_current_point(self, timestamp: float) -> None:
        if self.complete:
            return
        self._features[self._point_index].clear()
        self._point_started_at = float(timestamp)
        self.blocked_reason = None

    def training_data(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.complete:
            raise RuntimeError("calibration is not complete")
        vectors: list[np.ndarray] = []
        labels: list[tuple[float, float]] = []
        for point, point_features in zip(self.points, self._features):
            vectors.extend(point_features)
            labels.extend((point.screen_x, point.screen_y) for _ in point_features)
        return np.vstack(vectors), np.asarray(labels, dtype=float)

    def feature_bounds(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        features, _labels = self.training_data()
        lower = np.percentile(features, 2, axis=0)
        upper = np.percentile(features, 98, axis=0)
        return tuple(map(float, lower)), tuple(map(float, upper))


def filter_stable_features(
    features: Sequence[object] | np.ndarray,
    *,
    max_samples: int = 24,
    min_samples: int = 8,
) -> np.ndarray:
    values = np.asarray(features, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("features must be a 2D array")
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.shape[0] == 0:
        return values

    center = np.median(values, axis=0)
    mad = np.median(np.abs(values - center), axis=0)
    std_fallback = np.std(values, axis=0) * 0.6745
    scale = 1.4826 * np.maximum(mad, std_fallback * 0.25)
    scale = np.maximum(scale, 1e-6)
    robust_z = np.abs((values - center) / scale)
    row_score = np.percentile(robust_z, 80, axis=1)

    keep = np.flatnonzero(row_score <= 3.5)
    wanted_min = min(max(1, int(min_samples)), values.shape[0])
    if keep.size < wanted_min:
        keep = np.argsort(row_score)[:wanted_min]
    keep = np.sort(keep)
    cap = max(1, int(max_samples))
    if keep.size > cap:
        positions = np.linspace(0, keep.size - 1, cap).round().astype(int)
        keep = keep[positions]
    return values[keep]


def balance_point_samples(
    groups: Sequence[tuple[tuple[float, float], np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if not groups:
        raise ValueError("at least one calibration point group is required")
    prepared = [(target, np.asarray(features, dtype=float)) for target, features in groups]
    if any(features.ndim != 2 or features.shape[0] == 0 for _target, features in prepared):
        raise ValueError("each calibration point requires a non-empty 2D feature array")
    feature_widths = {features.shape[1] for _target, features in prepared}
    if len(feature_widths) != 1:
        raise ValueError("feature widths must match")

    balanced_count = min(features.shape[0] for _target, features in prepared)
    feature_rows: list[np.ndarray] = []
    target_rows: list[tuple[float, float]] = []
    for target, features in prepared:
        balanced = features[:balanced_count]
        feature_rows.append(balanced)
        target_rows.extend((float(target[0]), float(target[1])) for _ in range(balanced_count))
    return np.vstack(feature_rows), np.asarray(target_rows, dtype=float)


def fit_screen_affine(predicted_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted_points, dtype=float)
    targets = np.asarray(target_points, dtype=float)
    if predicted.shape != targets.shape or predicted.ndim != 2 or predicted.shape[1] != 2:
        raise ValueError("predicted_points and target_points must both be Nx2")
    design = np.column_stack((predicted, np.ones(predicted.shape[0], dtype=float)))
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design, targets, rcond=None)
    if rank < 3:
        raise RuntimeError("calibration validation points are degenerate")
    return coefficients


def apply_screen_affine(
    coefficients: Sequence[Sequence[float]] | np.ndarray,
    x: float,
    y: float,
) -> tuple[float, float]:
    matrix = np.asarray(coefficients, dtype=float)
    if matrix.shape != (3, 2):
        raise ValueError("screen affine coefficients must be 3x2")
    corrected = np.asarray([float(x), float(y), 1.0]) @ matrix
    return float(corrected[0]), float(corrected[1])


def score_validation(
    hit_counts: Mapping[str, int],
    *,
    min_hits_per_target: int = 2,
) -> ValidationResult:
    if min_hits_per_target < 1:
        raise ValueError("min_hits_per_target must be positive")
    failed = tuple(
        target_id
        for target_id, hit_count in hit_counts.items()
        if int(hit_count) < min_hits_per_target
    )
    return ValidationResult(len(hit_counts) - len(failed), len(hit_counts), failed)


class CalibrationStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def save(self, model: SavableModel, metadata: CalibrationMetadata) -> StoredCalibration:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / metadata.calibration_id
        temporary = self.root / f".{metadata.calibration_id}.tmp"
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True)
        model_path = temporary / "model.pkl"
        model.save_model(model_path)
        if not model_path.is_file() or model_path.stat().st_size <= 0:
            raise RuntimeError("calibration model was not saved")
        (temporary / "metadata.json").write_text(
            json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        shutil.rmtree(target, ignore_errors=True)
        temporary.replace(target)
        current_tmp = self.root / "current.json.tmp"
        current_tmp.write_text(
            json.dumps({"calibration_id": metadata.calibration_id}),
            encoding="utf-8",
        )
        current_tmp.replace(self.root / "current.json")
        return StoredCalibration(target / "model.pkl", metadata)

    def compatibility(self, environment: CalibrationEnvironment) -> CompatibilityResult:
        try:
            current = json.loads((self.root / "current.json").read_text(encoding="utf-8"))
            calibration_id = str(current["calibration_id"])
            metadata = self._read_metadata(self.root / calibration_id / "metadata.json")
            if not (self.root / calibration_id / "model.pkl").is_file():
                return CompatibilityResult(False, "校准模型文件缺失", calibration_id)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return CompatibilityResult(False, "尚无可用校准")

        saved = metadata.environment
        if (saved.screen_width, saved.screen_height) != (
            environment.screen_width,
            environment.screen_height,
        ):
            return CompatibilityResult(False, "显示器分辨率不匹配", calibration_id)
        if round(saved.scale_factor, 2) != round(environment.scale_factor, 2):
            return CompatibilityResult(False, "系统缩放比例不匹配", calibration_id)
        if saved.camera_index != environment.camera_index:
            return CompatibilityResult(False, "摄像头编号不匹配", calibration_id)
        if saved.layout_version != environment.layout_version:
            return CompatibilityResult(False, "界面布局版本不匹配", calibration_id)
        return CompatibilityResult(True, "校准可用", calibration_id)

    def load(self, environment: CalibrationEnvironment) -> StoredCalibration | None:
        compatibility = self.compatibility(environment)
        if not compatibility.compatible or compatibility.calibration_id is None:
            return None
        directory = self.root / compatibility.calibration_id
        return StoredCalibration(
            directory / "model.pkl",
            self._read_metadata(directory / "metadata.json"),
        )

    @staticmethod
    def _read_metadata(path: Path) -> CalibrationMetadata:
        payload = json.loads(path.read_text(encoding="utf-8"))
        environment = CalibrationEnvironment(**payload["environment"])
        return CalibrationMetadata(
            calibration_id=payload["calibration_id"],
            created_at=payload["created_at"],
            environment=environment,
            feature_min=tuple(payload["feature_min"]),
            feature_max=tuple(payload["feature_max"]),
            calibration_mode=payload.get("calibration_mode", CalibrationMode.FAST.value),
            pipeline_version=payload.get("pipeline_version", "legacy"),
            feature_range_threshold=payload.get("feature_range_threshold"),
            screen_affine=tuple(tuple(row) for row in payload.get("screen_affine", ())),
            validation_hits=payload.get("validation_hits"),
            validation_total=payload.get("validation_total"),
        )
