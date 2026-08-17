from __future__ import annotations

import numpy as np
import pytest

from pure_gaze_typing.adaptive_gaze import AdaptiveGazeSession


SCREEN_SIZE = (1920, 1080)
TARGET_RECT = (200.0, 150.0, 300.0, 240.0)


def fill_stable_window(
    session: AdaptiveGazeSession,
    point: tuple[float, float],
    *,
    start: float = 0.0,
    count: int = 8,
) -> None:
    session.clear_window()
    for index in range(count):
        session.observe(
            start + index * 0.05,
            valid=True,
            blink=False,
            x=point[0] + (index % 2) * 0.4,
            y=point[1] - (index % 2) * 0.4,
            quality=0.9,
        )


def test_disabled_session_stays_identity() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE, enabled=False)
    fill_stable_window(session, (300.0, 220.0))

    assert session.apply(320.0, 240.0) == pytest.approx((320.0, 240.0))
    decision = session.consider_anchor("target_0", TARGET_RECT)

    assert not decision.accepted
    assert decision.reason == "disabled"


def test_window_requires_eight_valid_samples() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE)
    fill_stable_window(session, (300.0, 220.0), count=7)

    decision = session.consider_anchor("target_0", TARGET_RECT)

    assert not decision.accepted
    assert decision.reason == "insufficient_samples"


def test_accepted_update_reduces_residual() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE)
    observed = (300.0, 270.0)
    target = np.asarray((350.0, 270.0))
    fill_stable_window(session, observed)
    before = np.linalg.norm(np.asarray(session.apply(*observed)) - target)

    decision = session.consider_anchor("target_0", TARGET_RECT)

    after = np.linalg.norm(np.asarray(session.apply(*observed)) - target)
    assert decision.accepted
    assert decision.matrix_version == 1
    assert after < before


def test_anchor_core_gate_uses_currently_corrected_point() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE)
    for cycle in range(2):
        fill_stable_window(session, (300.0, 270.0), start=float(cycle))
        assert session.consider_anchor("target_0", TARGET_RECT).accepted

    raw_point_outside_core = (245.0, 270.0)
    assert session.apply(*raw_point_outside_core)[0] > 252.5
    fill_stable_window(session, raw_point_outside_core, start=3.0)

    decision = session.consider_anchor("target_0", TARGET_RECT)

    assert decision.accepted


def test_invalid_ratio_and_blinks_reject_window() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE, min_valid_samples=6)
    for index in range(6):
        session.observe(
            index * 0.05,
            valid=True,
            blink=False,
            x=300.0,
            y=220.0,
            quality=0.9,
        )
    for index in range(3):
        session.observe(
            0.30 + index * 0.05,
            valid=False,
            blink=index == 0,
            quality=0.0,
        )

    decision = session.consider_anchor("target_0", TARGET_RECT)

    assert not decision.accepted
    assert decision.reason == "invalid_ratio"


def test_high_mad_rejects_window() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE)
    for index, x in enumerate((260, 440, 270, 430, 280, 420, 290, 410)):
        session.observe(
            index * 0.05,
            valid=True,
            blink=False,
            x=float(x),
            y=270.0,
            quality=0.9,
        )

    decision = session.consider_anchor("target_0", TARGET_RECT)

    assert not decision.accepted
    assert decision.reason == "high_mad"


def test_time_window_prunes_old_samples() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE)
    fill_stable_window(session, (300.0, 270.0), start=0.0)
    session.observe(
        1.0,
        valid=True,
        blink=False,
        x=300.0,
        y=270.0,
        quality=0.9,
    )

    decision = session.consider_anchor("target_0", TARGET_RECT)

    assert not decision.accepted
    assert decision.reason == "insufficient_samples"


def test_unsafe_rejections_roll_back_and_keep_reliable_output() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE)
    for cycle in range(2):
        fill_stable_window(session, (300.0, 270.0), start=float(cycle))
        assert session.consider_anchor("target_0", TARGET_RECT).accepted

    for cycle in range(3):
        fill_stable_window(session, (400.0, 270.0), start=3.0 + cycle)
        decision = session.consider_anchor(
            "target_far",
            (0.0, 0.0, float(SCREEN_SIZE[0]), float(SCREEN_SIZE[1])),
        )
        assert not decision.accepted
        assert decision.reason == "residual_too_large"

    snapshot = session.snapshot()
    assert snapshot.rollback_count == 1
    assert snapshot.suspended
    assert session.apply(300.0, 270.0)[0] > 305.0


def test_reset_restores_fresh_identity_learning() -> None:
    session = AdaptiveGazeSession(screen_size=SCREEN_SIZE)
    fill_stable_window(session, (300.0, 270.0))
    assert session.consider_anchor("target_0", TARGET_RECT).accepted

    decision = session.reset("calibration_changed")

    assert decision.reason == "calibration_changed"
    assert decision.matrix_version == 0
    assert session.apply(300.0, 270.0) == pytest.approx((300.0, 270.0))
    snapshot = session.snapshot()
    assert snapshot.matrix_version == 0
    assert snapshot.accepted_updates == 0
    assert snapshot.rejected_updates == 0
    assert snapshot.rollback_count == 0
    assert not snapshot.suspended
