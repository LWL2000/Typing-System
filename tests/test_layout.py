import pytest

from pure_gaze_typing.layout import build_layout, calibration_points, hit_test


def test_layout_has_six_stable_targets_and_back_region():
    layout = build_layout(1920, 1080)
    assert layout.version == "gaze-grid-v1"
    assert len(layout.targets) == 6
    expected_centers = [
        (326.4, 356.4),
        (960.0, 356.4),
        (1593.6, 356.4),
        (326.4, 788.4),
        (960.0, 788.4),
        (1593.6, 788.4),
    ]
    for actual, expected in zip((rect.center for rect in layout.targets), expected_centers):
        assert actual == pytest.approx(expected)
    assert all(
        not first.intersects(second)
        for index, first in enumerate(layout.targets)
        for second in layout.targets[index + 1 :]
    )


def test_calibration_and_hit_testing_use_the_same_centers():
    layout = build_layout(1280, 720)
    points = dict(calibration_points(layout))
    assert points["center"] == (640.0, 360.0)
    for index, rect in enumerate(layout.targets):
        x, y = points[f"target_{index}"]
        assert hit_test(layout, x, y, include_back=False) == f"target_{index}"
    assert hit_test(layout, *points["back"], include_back=True) == "back"


def test_gaps_and_hidden_back_region_do_not_select_targets():
    layout = build_layout(1000, 1000)
    assert hit_test(layout, 335.0, 330.0, include_back=False) is None
    assert hit_test(layout, *layout.back_target.center, include_back=False) is None
