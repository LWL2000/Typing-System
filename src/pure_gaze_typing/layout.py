from __future__ import annotations

from dataclasses import dataclass
import math


LAYOUT_VERSION = "gaze-keyboard-v5-equal-grid"
MAIN_GRID_CELL_INDICES = (0, 3, 6, 1, 7, 2, 5, 8)
SUBMENU_GRID_CELL_INDICES = (0, 1, 2, 3, 4, 5)


@dataclass(frozen=True)
class PixelRect:
    left: float
    top: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.left + self.width

    @property
    def bottom(self) -> float:
        return self.top + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def expanded(self, amount: float) -> "PixelRect":
        margin = max(0.0, float(amount))
        return PixelRect(
            self.left - margin,
            self.top - margin,
            self.width + margin * 2.0,
            self.height + margin * 2.0,
        )

    def intersects(self, other: "PixelRect") -> bool:
        return not (
            self.right <= other.left
            or other.right <= self.left
            or self.bottom <= other.top
            or other.bottom <= self.top
        )

    def scaled(self, width: float, height: float) -> "PixelRect":
        return PixelRect(
            self.left * width,
            self.top * height,
            self.width * width,
            self.height * height,
        )


@dataclass(frozen=True)
class LayoutSpec:
    version: str
    screen_width: int
    screen_height: int
    top_bar: PixelRect
    keyboard_bounds: PixelRect
    grid_cells: tuple[PixelRect, ...]
    main_targets: tuple[PixelRect, ...]
    submenu_targets: tuple[PixelRect, ...]

    @property
    def targets(self) -> tuple[PixelRect, ...]:
        return self.main_targets

    def targets_for(self, target_count: int) -> tuple[PixelRect, ...]:
        if target_count == 8:
            return self.main_targets
        if target_count == 6:
            return self.submenu_targets
        raise ValueError("typing pages must contain exactly six or eight targets")


def build_layout(width: int, height: int) -> LayoutSpec:
    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    width_f, height_f = float(width), float(height)
    top_height = max(124.0, height_f * 0.14)
    top_height = min(top_height, height_f * 0.25)
    top_bar = PixelRect(0.0, 0.0, width_f, top_height)
    keyboard_bounds = PixelRect(0.0, top_height, width_f, height_f - top_height)
    cell_width = keyboard_bounds.width / 3.0
    cell_height = keyboard_bounds.height / 3.0
    grid_cells = tuple(
        PixelRect(
            column * cell_width,
            keyboard_bounds.top + row * cell_height,
            cell_width,
            cell_height,
        )
        for row in range(3)
        for column in range(3)
    )
    main_targets = tuple(grid_cells[index] for index in MAIN_GRID_CELL_INDICES)
    submenu_targets = tuple(grid_cells[index] for index in SUBMENU_GRID_CELL_INDICES)
    return LayoutSpec(
        version=LAYOUT_VERSION,
        screen_width=int(width),
        screen_height=int(height),
        top_bar=top_bar,
        keyboard_bounds=keyboard_bounds,
        grid_cells=grid_cells,
        main_targets=main_targets,
        submenu_targets=submenu_targets,
    )


def calibration_points(layout: LayoutSpec) -> tuple[tuple[str, tuple[float, float]], ...]:
    points = [("center", (layout.screen_width / 2.0, layout.screen_height / 2.0))]
    points.extend((f"target_{index}", rect.center) for index, rect in enumerate(layout.main_targets))
    return tuple(points)


def validation_points(layout: LayoutSpec) -> tuple[tuple[str, tuple[float, float]], ...]:
    return tuple(
        (f"validation_{index}", rect.center)
        for index, rect in enumerate(layout.submenu_targets)
    )


def reentry_calibration_points(
    layout: LayoutSpec,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    names = (
        "reentry_top_left",
        "reentry_top_right",
        "reentry_center",
        "reentry_bottom_left",
        "reentry_bottom_right",
    )
    return tuple(
        (name, layout.grid_cells[index].center)
        for name, index in zip(names, (0, 2, 4, 6, 8))
    )


def uniform_grid_calibration_points(
    layout: LayoutSpec,
    *,
    rows: int = 3,
    columns: int = 3,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    row_count = int(rows)
    column_count = int(columns)
    if not 2 <= row_count <= 5 or not 2 <= column_count <= 5:
        raise ValueError("calibration grid rows and columns must be between 2 and 5")
    points: list[tuple[str, tuple[float, float]]] = []
    for row in range(row_count):
        column_order = range(column_count) if row % 2 == 0 else reversed(range(column_count))
        for column in column_order:
            x = (float(column) + 0.5) * float(layout.screen_width) / float(column_count)
            y = (float(row) + 0.5) * float(layout.screen_height) / float(row_count)
            points.append((f"grid_{row}_{column}", (x, y)))
    return tuple(points)


def hit_test(
    layout: LayoutSpec,
    x: float,
    y: float,
    *,
    target_count: int,
    preferred_target_id: str | None = None,
    boundary_tolerance_px: float = 0.0,
    outer_tolerance_px: float = 32.0,
) -> str | None:
    point_x, point_y = float(x), float(y)
    if not math.isfinite(point_x) or not math.isfinite(point_y):
        return None
    bounds = layout.keyboard_bounds
    outer = max(0.0, float(outer_tolerance_px))
    if not bounds.expanded(outer).contains(point_x, point_y):
        return None
    point_x = min(max(point_x, bounds.left), math.nextafter(bounds.right, bounds.left))
    point_y = min(max(point_y, bounds.top), math.nextafter(bounds.bottom, bounds.top))
    targets = layout.targets_for(target_count)

    if preferred_target_id and preferred_target_id.startswith("target_"):
        try:
            preferred_index = int(preferred_target_id.removeprefix("target_"))
            preferred = targets[preferred_index]
        except (ValueError, IndexError):
            preferred = None
        if preferred is not None and preferred.expanded(boundary_tolerance_px).contains(point_x, point_y):
            return preferred_target_id

    for index, rect in enumerate(targets):
        if rect.contains(point_x, point_y):
            return f"target_{index}"
    return None
