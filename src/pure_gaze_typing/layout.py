from __future__ import annotations

from dataclasses import dataclass


LAYOUT_VERSION = "gaze-keyboard-v4-brain-layout"


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
    main_targets: tuple[PixelRect, ...]
    submenu_targets: tuple[PixelRect, ...]

    @property
    def targets(self) -> tuple[PixelRect, ...]:
        """Parent targets retained as the canonical calibration geometry."""
        return self.main_targets

    def targets_for(self, target_count: int) -> tuple[PixelRect, ...]:
        if target_count == 8:
            return self.main_targets
        if target_count == 6:
            return self.submenu_targets
        raise ValueError("typing pages must contain exactly six or eight targets")


def _rect_at(center_x: float, center_y: float, width: float, height: float) -> PixelRect:
    return PixelRect(center_x - width / 2.0, center_y - height / 2.0, width, height)


def build_layout(width: int, height: int) -> LayoutSpec:
    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    width_f, height_f = float(width), float(height)

    main_cells = (
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 2),
        (2, 0),
        (2, 1),
        (2, 2),
    )
    main_x = (width_f * 0.14, width_f * 0.50, width_f * 0.86)
    main_y = (height_f * 0.23, height_f * 0.50, height_f * 0.77)
    main_targets = tuple(
        _rect_at(main_x[column], main_y[row], width_f * 0.26, height_f * 0.19)
        for column, row in main_cells
    )

    submenu_cells = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    submenu_x = (width_f * 0.15, width_f * 0.50, width_f * 0.85)
    submenu_y = (height_f * 0.28, height_f * 0.72)
    submenu_targets = tuple(
        _rect_at(submenu_x[column], submenu_y[row], width_f * 0.28, height_f * 0.28)
        for column, row in submenu_cells
    )

    return LayoutSpec(
        version=LAYOUT_VERSION,
        screen_width=int(width),
        screen_height=int(height),
        top_bar=PixelRect(width_f * 0.025, height_f * 0.02, width_f * 0.95, height_f * 0.08),
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


def uniform_grid_calibration_points(
    layout: LayoutSpec,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    points: list[tuple[str, tuple[float, float]]] = []
    for row in range(3):
        columns = range(3) if row % 2 == 0 else reversed(range(3))
        for column in columns:
            x = (float(column) + 0.5) * float(layout.screen_width) / 3.0
            y = (float(row) + 0.5) * float(layout.screen_height) / 3.0
            points.append((f"grid_{row}_{column}", (x, y)))
    return tuple(points)


def hit_test(
    layout: LayoutSpec,
    x: float,
    y: float,
    *,
    target_count: int,
) -> str | None:
    for index, rect in enumerate(layout.targets_for(target_count)):
        if rect.contains(x, y):
            return f"target_{index}"
    return None
