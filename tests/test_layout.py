import pytest

from pure_gaze_typing.layout import (
    build_layout,
    calibration_points,
    hit_test,
    reentry_calibration_points,
    uniform_grid_calibration_points,
    validation_points,
)


def test_keyboard_area_is_an_equal_three_by_three_grid():
    layout = build_layout(1920, 1080)

    assert layout.version == "gaze-keyboard-v5-equal-grid"
    assert len(layout.grid_cells) == 9
    assert len(layout.main_targets) == 8
    assert len(layout.submenu_targets) == 6
    widths = {round(rect.width, 6) for rect in layout.grid_cells}
    heights = {round(rect.height, 6) for rect in layout.grid_cells}
    assert len(widths) == len(heights) == 1
    assert layout.grid_cells[0].left == pytest.approx(0.0)
    assert layout.grid_cells[0].top == pytest.approx(layout.keyboard_bounds.top)
    assert layout.grid_cells[-1].right == pytest.approx(1920.0)
    assert layout.grid_cells[-1].bottom == pytest.approx(1080.0)


def test_parent_target_order_preserves_reference_groups_and_center_is_safe():
    layout = build_layout(900, 600)

    expected_cell_indices = (0, 3, 6, 1, 7, 2, 5, 8)
    for target, cell_index in zip(layout.main_targets, expected_cell_indices):
        assert target == layout.grid_cells[cell_index]
    assert hit_test(layout, *layout.grid_cells[4].center, target_count=8) is None


def test_hit_testing_uses_the_active_page_geometry():
    layout = build_layout(1280, 720)

    for index, rect in enumerate(layout.main_targets):
        assert hit_test(layout, *rect.center, target_count=8) == f"target_{index}"
    for index, rect in enumerate(layout.submenu_targets):
        assert hit_test(layout, *rect.center, target_count=6) == f"target_{index}"


def test_hit_testing_clamps_outer_edges_and_holds_previous_target_near_boundaries():
    layout = build_layout(1200, 900)
    top_left = layout.main_targets[0]
    top_center = layout.main_targets[3]

    assert hit_test(
        layout,
        -18.0,
        top_left.center[1],
        target_count=8,
        outer_tolerance_px=24.0,
    ) == "target_0"
    assert hit_test(
        layout,
        top_left.right + 12.0,
        top_left.center[1],
        target_count=8,
        preferred_target_id="target_0",
        boundary_tolerance_px=24.0,
    ) == "target_0"
    assert hit_test(
        layout,
        top_center.center[0],
        top_center.center[1],
        target_count=8,
        preferred_target_id="target_0",
        boundary_tolerance_px=24.0,
    ) == "target_3"


def test_calibration_covers_parent_targets_and_validation_covers_six_regions():
    layout = build_layout(1280, 720)
    training = dict(calibration_points(layout))
    validation = dict(validation_points(layout))

    assert training["center"] == (640.0, 360.0)
    assert len([name for name in training if name.startswith("target_")]) == 8
    assert len(validation) == 6
    for index, rect in enumerate(layout.submenu_targets):
        assert validation[f"validation_{index}"] == rect.center


def test_top_input_area_does_not_select_targets():
    layout = build_layout(1000, 1000)
    assert hit_test(layout, 500.0, layout.top_bar.center[1], target_count=8) is None


def test_reference_grid_supports_variable_rows_and_columns_in_snake_order():
    layout = build_layout(900, 600)
    points = uniform_grid_calibration_points(layout, rows=4, columns=5)
    assert [name for name, _point in points] == [
        "grid_0_0", "grid_0_1", "grid_0_2", "grid_0_3", "grid_0_4",
        "grid_1_4", "grid_1_3", "grid_1_2", "grid_1_1", "grid_1_0",
        "grid_2_0", "grid_2_1", "grid_2_2", "grid_2_3", "grid_2_4",
        "grid_3_4", "grid_3_3", "grid_3_2", "grid_3_1", "grid_3_0",
    ]
    assert points[0][1] == pytest.approx((90.0, 75.0))
    assert points[-1][1] == pytest.approx((90.0, 525.0))


def test_reentry_calibration_uses_four_grid_corners_and_center():
    layout = build_layout(900, 600)

    points = reentry_calibration_points(layout)

    assert [name for name, _point in points] == [
        "reentry_top_left",
        "reentry_top_right",
        "reentry_center",
        "reentry_bottom_left",
        "reentry_bottom_right",
    ]
    assert [point for _name, point in points] == [
        layout.grid_cells[index].center for index in (0, 2, 4, 6, 8)
    ]
