import pytest

from pure_gaze_typing.layout import (
    build_layout,
    calibration_points,
    hit_test,
    uniform_grid_calibration_points,
    validation_points,
)


def test_layout_matches_reference_parent_and_submenu_geometry():
    layout = build_layout(1920, 1080)

    assert layout.version == "gaze-keyboard-v4-brain-layout"
    assert len(layout.main_targets) == 8
    assert len(layout.submenu_targets) == 6
    expected_main = [
        (268.8, 248.4),
        (268.8, 540.0),
        (268.8, 831.6),
        (960.0, 248.4),
        (960.0, 831.6),
        (1651.2, 248.4),
        (1651.2, 540.0),
        (1651.2, 831.6),
    ]
    for actual, expected in zip((rect.center for rect in layout.main_targets), expected_main):
        assert actual == pytest.approx(expected)
    expected_submenu = [
        (288.0, 302.4),
        (960.0, 302.4),
        (1632.0, 302.4),
        (288.0, 777.6),
        (960.0, 777.6),
        (1632.0, 777.6),
    ]
    for actual, expected in zip((rect.center for rect in layout.submenu_targets), expected_submenu):
        assert actual == pytest.approx(expected)


def test_hit_testing_uses_the_active_page_geometry():
    layout = build_layout(1280, 720)

    for index, rect in enumerate(layout.main_targets):
        assert hit_test(layout, *rect.center, target_count=8) == f"target_{index}"
    for index, rect in enumerate(layout.submenu_targets):
        assert hit_test(layout, *rect.center, target_count=6) == f"target_{index}"


def test_calibration_covers_parent_targets_and_validation_covers_six_regions():
    layout = build_layout(1280, 720)
    training = dict(calibration_points(layout))
    validation = dict(validation_points(layout))

    assert training["center"] == (640.0, 360.0)
    assert len([name for name in training if name.startswith("target_")]) == 8
    assert len(validation) == 6
    for index, rect in enumerate(layout.submenu_targets):
        assert validation[f"validation_{index}"] == rect.center


def test_gaps_do_not_select_targets():
    layout = build_layout(1000, 1000)
    assert hit_test(layout, 500.0, 500.0, target_count=8) is None


def test_reference_grid_uses_nine_cell_centers_in_snake_order():
    layout = build_layout(900, 600)
    points = uniform_grid_calibration_points(layout)
    assert [name for name, _point in points] == [
        "grid_0_0", "grid_0_1", "grid_0_2",
        "grid_1_2", "grid_1_1", "grid_1_0",
        "grid_2_0", "grid_2_1", "grid_2_2",
    ]
    assert points[0][1] == pytest.approx((150.0, 100.0))
    assert points[-1][1] == pytest.approx((750.0, 500.0))
