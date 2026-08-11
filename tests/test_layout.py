import pytest

from pure_gaze_typing.layout import (
    build_layout,
    calibration_points,
    hit_test,
    uniform_grid_calibration_points,
)


def test_layout_has_six_stable_targets_and_back_region():
    layout = build_layout(1920, 1080)
    assert layout.version == "gaze-grid-v3-reference"
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


def test_reference_grid_uses_nine_cell_centers_in_snake_order():
    layout = build_layout(900, 600)

    points = uniform_grid_calibration_points(layout)

    assert [name for name, _point in points] == [
        "grid_0_0",
        "grid_0_1",
        "grid_0_2",
        "grid_1_2",
        "grid_1_1",
        "grid_1_0",
        "grid_2_0",
        "grid_2_1",
        "grid_2_2",
    ]
    assert points[0][1] == pytest.approx((150.0, 100.0))
    assert points[-1][1] == pytest.approx((750.0, 500.0))
