from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

from PyQt6.QtWidgets import QApplication

from .app_logging import configure_logging
from .calibration import CalibrationEnvironment
from .capture_window import CaptureController, CaptureWindow
from .eyetrax_runtime import resource_path
from .layout import LAYOUT_VERSION, build_layout
from .paths import AppPaths
from .protocol import Heartbeat, decode_message, encode_message


def find_face_model() -> Path:
    bundled = resource_path("face_landmarker.task")
    if bundled.is_file():
        return bundled
    return Path.home() / ".cache" / "eyetrax" / "mediapipe" / "face_landmarker.task"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="纯眼动采集与校准")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication.instance() or QApplication(["眼动采集校准"])
    paths = AppPaths.default()
    configure_logging(paths, "capture")
    if args.self_test:
        model = find_face_model()
        if not model.is_file() or model.stat().st_size <= 0:
            logging.getLogger("pure_gaze_typing").error("FaceLandmarker 模型缺失：%s", model)
            return 2
        assert len(build_layout(1920, 1080).targets) == 6
        heartbeat = Heartbeat(1.0, True, True, "self-test", LAYOUT_VERSION, 30.0)
        assert decode_message(encode_message(heartbeat)) == heartbeat
        return 0

    screen = app.primaryScreen()
    if screen is None:
        return 2
    geometry = screen.geometry()
    scale = screen.devicePixelRatio()

    def environment_factory(camera_index: int) -> CalibrationEnvironment:
        return CalibrationEnvironment(
            geometry.width(),
            geometry.height(),
            scale,
            camera_index,
            LAYOUT_VERSION,
        )

    controller = CaptureController(paths, environment_factory, find_face_model())
    window = CaptureWindow(controller, paths)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
