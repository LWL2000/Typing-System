from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import math
import numpy as np


class AdaptiveState(str, Enum):
    READY = "READY"
    ADAPTING = "ADAPTING"
    SUSPENDED = "SUSPENDED"


@dataclass(frozen=True)
class AdaptiveObservation:
    timestamp: float
    valid: bool
    blink: bool
    x: Optional[float]
    y: Optional[float]
    quality: float = 0.0


@dataclass(frozen=True)
class StableWindowStats:
    accepted: bool
    reason: str
    sample_count: int
    valid_count: int
    invalid_ratio: float
    median_xy: Optional[tuple[float, float]]
    mad_px: float
    max_quality: Optional[float]


@dataclass(frozen=True)
class RlsConfig:
    forgetting_factor: float = 0.995
    huber_delta_px: float = 90.0
    max_step_px: float = 12.0
    initial_covariance: float = 60.0

    def __post_init__(self) -> None:
        if not 0.90 <= float(self.forgetting_factor) <= 1.0:
            raise ValueError("forgetting_factor must be between 0.90 and 1.0")
        if min(
            float(self.huber_delta_px),
            float(self.max_step_px),
            float(self.initial_covariance),
        ) <= 0.0:
            raise ValueError("RLS parameters must be positive")


@dataclass(frozen=True)
class AffineConstraints:
    min_scale: float = 0.80
    max_scale: float = 1.25
    max_rotation_deg: float = 10.0
    max_shear: float = 0.12
    max_translation_ratio: float = 0.18
    max_condition: float = 20.0


@dataclass(frozen=True)
class AdaptiveDecision:
    accepted: bool
    reason: str
    residual_before: Optional[float]
    residual_after: Optional[float]
    matrix_version: int
    rollback_performed: bool = False
    unsafe: bool = False


@dataclass(frozen=True)
class AdaptiveSnapshot:
    state: AdaptiveState
    matrix_version: int
    accepted_updates: int
    rejected_updates: int
    rollback_count: int
    recent_residual: Optional[float]
    consecutive_unsafe_rejections: int
    suspended: bool
    enabled: bool


def identity_affine() -> np.ndarray:
    return np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=float)


def apply_affine(points: np.ndarray | list[float] | list[list[float]], matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("points must contain at least x,y")
    matrix = np.asarray(matrix, dtype=float).reshape(3, 2)
    design = np.column_stack((values[:, :2], np.ones(len(values), dtype=float)))
    return design @ matrix


def _matrix_to_normalized(matrix: np.ndarray, screen_size: tuple[int, int]) -> np.ndarray:
    width, height = map(float, screen_size)
    matrix = np.asarray(matrix, dtype=float).reshape(3, 2)
    return np.asarray(
        [
            [matrix[0, 0], matrix[0, 1] * width / height],
            [matrix[1, 0] * height / width, matrix[1, 1]],
            [matrix[2, 0] / width, matrix[2, 1] / height],
        ],
        dtype=float,
    )


def _normalized_to_matrix(theta: np.ndarray, screen_size: tuple[int, int]) -> np.ndarray:
    width, height = map(float, screen_size)
    theta = np.asarray(theta, dtype=float)
    return np.asarray(
        [
            [theta[0, 0], theta[0, 1] * height / width],
            [theta[1, 0] * width / height, theta[1, 1]],
            [theta[2, 0] * width, theta[2, 1] * height],
        ],
        dtype=float,
    )


def inspect_affine(
    matrix: np.ndarray,
    screen_size: tuple[int, int],
    constraints: AffineConstraints,
) -> tuple[bool, str]:
    coefficients = np.asarray(matrix, dtype=float).reshape(3, 2)
    if coefficients.shape != (3, 2) or not np.isfinite(coefficients).all():
        return False, "non_finite"

    linear = coefficients[:2, :].T
    determinant = float(np.linalg.det(linear))
    scale_x = float(np.linalg.norm(linear[:, 0]))
    scale_y = float(np.linalg.norm(linear[:, 1]))
    rotation_deg = math.degrees(math.atan2(linear[1, 0], linear[0, 0]))
    denominator = max(scale_x * scale_y, 1e-12)
    shear = float(np.dot(linear[:, 0], linear[:, 1]) / denominator)
    condition = float(np.linalg.cond(linear))
    width, height = map(float, screen_size)
    translation_x = float(coefficients[2, 0])
    translation_y = float(coefficients[2, 1])

    if determinant <= 0.0:
        return False, "invalid_affine"
    if not (constraints.min_scale <= scale_x <= constraints.max_scale and constraints.min_scale <= scale_y <= constraints.max_scale):
        return False, "scale"
    if abs(rotation_deg) > constraints.max_rotation_deg:
        return False, "rotation"
    if abs(shear) > constraints.max_shear:
        return False, "shear"
    if abs(translation_x) > width * constraints.max_translation_ratio or abs(
        translation_y,
    ) > height * constraints.max_translation_ratio:
        return False, "translation"
    if not math.isfinite(condition) or condition > constraints.max_condition:
        return False, "condition"

    return True, "accepted"


def inner_core_contains(
    point: tuple[float, float],
    rect: tuple[float, float, float, float],
    *,
    core_ratio: float = 0.65,
) -> bool:
    x, y = map(float, point)
    left, top, width, height = map(float, rect)
    if not all(math.isfinite(value) for value in (x, y, left, top, width, height)):
        return False
    if width <= 0.0 or height <= 0.0:
        return False
    ratio = max(0.05, min(1.0, float(core_ratio)))
    center_x = left + width / 2.0
    center_y = top + height / 2.0
    return (
        abs(x - center_x) <= width * ratio / 2.0
        and abs(y - center_y) <= height * ratio / 2.0
    )


class AdaptiveGazeSession:
    _UNSAFE_REASONS = {"residual_too_large", "invalid_affine", "residual_worsened"}

    def __init__(
        self,
        *,
        screen_size: tuple[int, int],
        origin: tuple[float, float] = (0.0, 0.0),
        enabled: bool = True,
        constraints: AffineConstraints | None = None,
        rls_config: RlsConfig | None = None,
        max_window_seconds: float = 0.55,
        min_valid_samples: int = 8,
        max_invalid_ratio: float = 0.25,
        max_mad_px: float = 38.0,
        max_quality: float = 1.5,
        max_consecutive_unsafe_rejections: int = 3,
        history_limit: int = 8,
    ) -> None:
        width, height = int(screen_size[0]), int(screen_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("screen size must be positive")
        origin_x, origin_y = float(origin[0]), float(origin[1])
        if not math.isfinite(origin_x) or not math.isfinite(origin_y):
            raise ValueError("session origin must be finite")
        if int(min_valid_samples) <= 0:
            raise ValueError("min_valid_samples must be positive")
        if float(max_window_seconds) <= 0.0:
            raise ValueError("max_window_seconds must be positive")
        if float(max_mad_px) <= 0.0:
            raise ValueError("max_mad_px must be positive")
        if float(max_quality) <= 0.0:
            raise ValueError("max_quality must be positive")
        self.screen_size = (width, height)
        self.origin = (origin_x, origin_y)
        self.constraints = constraints or AffineConstraints()
        self.rls_config = rls_config or RlsConfig()
        self.max_window_seconds = float(max_window_seconds)
        self.min_valid_samples = int(min_valid_samples)
        self.max_invalid_ratio = float(max_invalid_ratio)
        self.max_mad_px = float(max_mad_px)
        self.max_quality = float(max_quality)
        self.max_consecutive_unsafe_rejections = max(1, int(max_consecutive_unsafe_rejections))
        self.history_limit = max(2, int(history_limit))

        self._observations: deque[AdaptiveObservation] = deque()
        self._enabled = bool(enabled)
        self._matrix = identity_affine()
        self._theta = _matrix_to_normalized(self._matrix, self.screen_size)
        self._covariance = self._initial_covariance()
        self._history: list[np.ndarray] = [self._matrix.copy()]

        self.state = AdaptiveState.READY
        self.matrix_version = 0
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.rollback_count = 0
        self.recent_residual: Optional[float] = None
        self._suspended = False
        self._consecutive_unsafe_rejections = 0
        self.last_target_id: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        if bool(enabled):
            self._enabled = True
            self._suspended = False
            if self.state == AdaptiveState.SUSPENDED:
                self.state = AdaptiveState.READY
            return
        self._enabled = False
        self._suspended = False
        self.clear_window()

    def snapshot(self) -> AdaptiveSnapshot:
        return AdaptiveSnapshot(
            state=self.state,
            matrix_version=self.matrix_version,
            accepted_updates=self.accepted_updates,
            rejected_updates=self.rejected_updates,
            rollback_count=self.rollback_count,
            recent_residual=self.recent_residual,
            consecutive_unsafe_rejections=self._consecutive_unsafe_rejections,
            suspended=self._suspended,
            enabled=self._enabled,
        )

    def apply(self, x: float, y: float) -> tuple[float, float]:
        if not self.enabled:
            return float(x), float(y)
        values = np.asarray([[float(x), float(y)]], dtype=float)
        point = values - np.asarray(self.origin, dtype=float)
        corrected = apply_affine(point, self._matrix)[0] + np.asarray(
            self.origin,
            dtype=float,
        )
        corrected = np.asarray(corrected, dtype=float).reshape(-1)
        return float(corrected[0]), float(corrected[1])

    def observe(
        self,
        timestamp: float,
        *,
        valid: bool,
        blink: bool,
        x: float | None = None,
        y: float | None = None,
        quality: float = 0.0,
    ) -> None:
        now = float(timestamp)
        safe_x = float(x) if x is not None else None
        safe_y = float(y) if y is not None else None
        if valid and (safe_x is None or safe_y is None):
            raise ValueError("valid samples require gaze coordinates")
        if (safe_x is not None and not math.isfinite(safe_x)) or (
            safe_y is not None and not math.isfinite(safe_y)
        ):
            safe_x = None
            safe_y = None
            valid = False
            blink = True
        self._observations.append(
            AdaptiveObservation(
                timestamp=now,
                valid=bool(valid),
                blink=bool(blink),
                x=safe_x,
                y=safe_y,
                quality=max(0.0, min(float(quality), 1.0)),
            ),
        )
        self._prune(now)

    def clear_window(self) -> None:
        self._observations.clear()

    def reset(self, reason: str = "reset") -> AdaptiveDecision:
        self._matrix = identity_affine()
        self._theta = _matrix_to_normalized(self._matrix, self.screen_size)
        self._covariance = self._initial_covariance()
        self._history = [self._matrix.copy()]
        self.clear_window()
        self.state = AdaptiveState.READY
        self.matrix_version = 0
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.rollback_count = 0
        self.recent_residual = None
        self._suspended = False
        self._consecutive_unsafe_rejections = 0
        self.last_target_id = None
        return AdaptiveDecision(
            accepted=False,
            reason=str(reason),
            residual_before=None,
            residual_after=None,
            matrix_version=self.matrix_version,
            rollback_performed=False,
        )

    def consider_anchor(
        self,
        target_id: str,
        target_rect: tuple[float, float, float, float],
    ) -> AdaptiveDecision:
        if not self._enabled:
            return AdaptiveDecision(
                accepted=False,
                reason="disabled",
                residual_before=None,
                residual_after=None,
                matrix_version=self.matrix_version,
            )
        if self.state == AdaptiveState.SUSPENDED:
            return AdaptiveDecision(
                accepted=False,
                reason="suspended",
                residual_before=None,
                residual_after=None,
                matrix_version=self.matrix_version,
                unsafe=False,
            )

        stats = self._stable_window()
        if not stats.accepted:
            return AdaptiveDecision(
                accepted=False,
                reason=stats.reason,
                residual_before=None,
                residual_after=None,
                matrix_version=self.matrix_version,
            )
        assert stats.median_xy is not None

        target_x = float(target_rect[0]) + float(target_rect[2]) / 2.0
        target_y = float(target_rect[1]) + float(target_rect[3]) / 2.0
        target = np.asarray([target_x, target_y], dtype=float)
        median = np.asarray(stats.median_xy, dtype=float)
        current = apply_affine(median - np.asarray(self.origin, dtype=float), self._matrix)[0] + np.asarray(
            self.origin,
            dtype=float,
        )
        if not inner_core_contains((float(current[0]), float(current[1])), target_rect):
            return AdaptiveDecision(
                accepted=False,
                reason="outside_target_core",
                residual_before=None,
                residual_after=None,
                matrix_version=self.matrix_version,
            )

        error = target - current
        residual_before = float(np.linalg.norm(error))
        diagonal = math.hypot(*self.screen_size)
        if residual_before > diagonal * 0.18:
            return self._reject(
                "residual_too_large",
                residual_before,
                None,
                target_id,
            )

        width, height = self.screen_size
        fixed_local = median - np.asarray(self.origin, dtype=float)
        design = np.asarray([fixed_local[0] / width, fixed_local[1] / height, 1.0], dtype=float)
        clipped_error = error.copy()
        if residual_before > 0.0 and residual_before > self.rls_config.max_step_px:
            clipped_error *= self.rls_config.max_step_px / residual_before
        clipped_normalized = np.asarray(
            [clipped_error[0] / width, clipped_error[1] / height],
            dtype=float,
        )
        huber_weight = min(
            1.0,
            self.rls_config.huber_delta_px / max(residual_before, 1e-9),
        )

        covariance_design = self._covariance @ design
        denominator = self.rls_config.forgetting_factor + float(design @ covariance_design)
        gain = covariance_design / max(denominator, 1e-12)
        candidate_theta = self._theta + np.outer(gain, clipped_normalized * huber_weight)
        candidate_covariance = (
            self._covariance - np.outer(gain, design) @ self._covariance
        ) / self.rls_config.forgetting_factor
        candidate = _normalized_to_matrix(candidate_theta, self.screen_size)
        valid, reason = inspect_affine(candidate, self.screen_size, self.constraints)
        if not valid:
            return self._reject(
                reason,
                residual_before,
                None,
                target_id,
                unsafe=reason in {"invalid_affine", "shear", "scale", "rotation", "translation", "condition"},
            )

        candidate_point = (
            apply_affine(np.asarray([fixed_local]), candidate)[0]
            + np.asarray(self.origin, dtype=float)
        )
        residual_after = float(np.linalg.norm(candidate_point - target))
        if residual_after > residual_before + 1e-6:
            return self._reject(
                "residual_worsened",
                residual_before,
                residual_after,
                target_id,
            )

        self._commit(candidate, candidate_covariance, target_id)
        self.recent_residual = residual_after
        return AdaptiveDecision(
            accepted=True,
            reason="accepted",
            residual_before=residual_before,
            residual_after=residual_after,
            matrix_version=self.matrix_version,
        )

    def _stable_window(self) -> StableWindowStats:
        observations = list(self._observations)
        total = len(observations)
        if total == 0:
            return StableWindowStats(
                accepted=False,
                reason="insufficient_samples",
                sample_count=0,
                valid_count=0,
                invalid_ratio=1.0,
                median_xy=None,
                mad_px=math.inf,
                max_quality=None,
            )

        valid_coordinates: list[tuple[float, float, float]] = []
        max_quality: float | None = None
        for item in observations:
            if item.quality is not None:
                max_quality = (
                    item.quality if max_quality is None else max(max_quality, item.quality)
                )
            if not item.valid or item.blink:
                continue
            if item.x is None or item.y is None:
                continue
            if not (math.isfinite(item.x) and math.isfinite(item.y)):
                continue
            valid_coordinates.append((float(item.x), float(item.y), float(item.quality)))

        valid_count = len(valid_coordinates)
        invalid_ratio = (total - valid_count) / float(total) if total else 1.0
        median_xy: Optional[tuple[float, float]] = None
        mad_px = math.inf
        if valid_coordinates:
            points = np.asarray([(x, y) for x, y, _ in valid_coordinates], dtype=float)
            median = np.median(points, axis=0)
            median_xy = (float(median[0]), float(median[1]))
            mad_px = float(np.median(np.linalg.norm(points - median, axis=1)))
        if valid_count < self.min_valid_samples:
            return StableWindowStats(
                accepted=False,
                reason="insufficient_samples",
                sample_count=total,
                valid_count=valid_count,
                invalid_ratio=invalid_ratio,
                median_xy=median_xy,
                mad_px=mad_px,
                max_quality=max_quality,
            )
        if invalid_ratio > self.max_invalid_ratio:
            return StableWindowStats(
                accepted=False,
                reason="invalid_ratio",
                sample_count=total,
                valid_count=valid_count,
                invalid_ratio=invalid_ratio,
                median_xy=median_xy,
                mad_px=mad_px,
                max_quality=max_quality,
            )
        if max_quality is not None and max_quality > self.max_quality:
            return StableWindowStats(
                accepted=False,
                reason="invalid_quality",
                sample_count=total,
                valid_count=valid_count,
                invalid_ratio=invalid_ratio,
                median_xy=median_xy,
                mad_px=mad_px,
                max_quality=max_quality,
            )
        if not mad_px <= self.max_mad_px or not math.isfinite(mad_px):
            return StableWindowStats(
                accepted=False,
                reason="high_mad",
                sample_count=total,
                valid_count=valid_count,
                invalid_ratio=invalid_ratio,
                median_xy=median_xy,
                mad_px=mad_px,
                max_quality=max_quality,
            )
        if median_xy is None:
            return StableWindowStats(
                accepted=False,
                reason="invalid_coordinate",
                sample_count=total,
                valid_count=valid_count,
                invalid_ratio=invalid_ratio,
                median_xy=None,
                mad_px=mad_px,
                max_quality=max_quality,
            )
        return StableWindowStats(
            accepted=True,
            reason="accepted",
            sample_count=total,
            valid_count=valid_count,
            invalid_ratio=invalid_ratio,
            median_xy=median_xy,
            mad_px=mad_px,
            max_quality=max_quality,
        )

    def _reject(
        self,
        reason: str,
        residual_before: Optional[float],
        residual_after: Optional[float],
        target_id: str | None,
        *,
        unsafe: bool = False,
    ) -> AdaptiveDecision:
        self.rejected_updates += 1
        self.last_target_id = target_id
        is_unsafe = reason in self._UNSAFE_REASONS if not unsafe else unsafe
        rollback_performed = False
        if is_unsafe:
            self._consecutive_unsafe_rejections += 1
            if self._consecutive_unsafe_rejections >= self.max_consecutive_unsafe_rejections:
                if self.rollback():
                    rollback_performed = True
                self._suspended = True
                self.state = AdaptiveState.SUSPENDED
        else:
            self._consecutive_unsafe_rejections = 0

        if residual_after is not None:
            self.recent_residual = residual_after
        elif residual_before is not None:
            self.recent_residual = residual_before
        return AdaptiveDecision(
            accepted=False,
            reason=reason,
            residual_before=residual_before,
            residual_after=residual_after,
            matrix_version=self.matrix_version,
            rollback_performed=rollback_performed,
            unsafe=is_unsafe,
        )

    def rollback(self) -> bool:
        if len(self._history) < 2:
            return False
        self._history.pop()
        self._matrix = self._history[-1].copy()
        self._theta = _matrix_to_normalized(self._matrix, self.screen_size)
        self._covariance = self._initial_covariance()
        self.matrix_version += 1
        self.rollback_count += 1
        self._consecutive_unsafe_rejections = 0
        self.state = AdaptiveState.READY
        return True

    def _commit(self, candidate: np.ndarray, covariance: np.ndarray, target_id: str | None = None) -> None:
        self._matrix = np.asarray(candidate, dtype=float).reshape(3, 2)
        self._theta = _matrix_to_normalized(self._matrix, self.screen_size)
        self._covariance = np.asarray(covariance, dtype=float).reshape(3, 3)
        self._history.append(self._matrix.copy())
        if len(self._history) > self.history_limit:
            self._history.pop(0)
        self.matrix_version += 1
        self.accepted_updates += 1
        self.last_target_id = target_id
        self.state = AdaptiveState.ADAPTING
        self._consecutive_unsafe_rejections = 0

    def _prune(self, now: float) -> None:
        while self._observations:
            if now - self._observations[0].timestamp > self.max_window_seconds:
                self._observations.popleft()
            else:
                break

    def _initial_covariance(self) -> np.ndarray:
        value = self.rls_config.initial_covariance
        return np.diag([value * 0.002, value * 0.002, value]).astype(float)
