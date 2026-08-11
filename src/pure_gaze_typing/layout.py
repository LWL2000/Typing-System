from __future__ import annotations

from dataclasses import dataclass


LAYOUT_VERSION = "gaze-grid-v1"


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
    back_target: PixelRect
    targets: tuple[PixelRect, ...]


_NORMALIZED_TARGETS = tuple(
    PixelRect(left, top, 0.26, 0.26)
    for top in (0.20, 0.60)
    for left in (0.04, 0.37, 0.70)
)


def build_layout(width: int, height: int) -> LayoutSpec:
    if width <= 0 or height <= 0:
        raise ValueError("screen dimensions must be positive")
    return LayoutSpec(
        version=LAYOUT_VERSION,
        screen_width=int(width),
        screen_height=int(height),
        top_bar=PixelRect(0.02, 0.02, 0.96, 0.10).scaled(width, height),
        back_target=PixelRect(0.02, 0.02, 0.14, 0.10).scaled(width, height),
        targets=tuple(rect.scaled(width, height) for rect in _NORMALIZED_TARGETS),
    )


def calibration_points(
    layout: LayoutSpec,
) -> tuple[tuple[str, tuple[float, float]], ...]:
    points = [
        ("center", (layout.screen_width / 2.0, layout.screen_height / 2.0)),
        ("back", layout.back_target.center),
    ]
    points.extend((f"target_{index}", rect.center) for index, rect in enumerate(layout.targets))
    return tuple(points)


def hit_test(
    layout: LayoutSpec,
    x: float,
    y: float,
    *,
    include_back: bool,
) -> str | None:
    if include_back and layout.back_target.contains(x, y):
        return "back"
    for index, rect in enumerate(layout.targets):
        if rect.contains(x, y):
            return f"target_{index}"
    return None
