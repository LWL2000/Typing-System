from __future__ import annotations

import argparse
import logging
import sys
import tempfile

from PyQt6.QtWidgets import QApplication

from .app_logging import configure_logging
from .layout import LAYOUT_VERSION
from .paths import AppPaths
from .protocol import GazeSample, Heartbeat, UdpReceiver, decode_message, encode_message
from .settings import TypingSettings, load_settings
from .typing_controller import TypingController
from .typing_window import StartupWindow, TypingWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="纯眼动打字器")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    app = QApplication.instance() or QApplication(["纯眼动打字器"])
    paths = AppPaths.default()
    paths.migrate_legacy()
    configure_logging(paths, "typing")
    screen = app.primaryScreen()
    if screen is None:
        return 2
    geometry = screen.geometry()
    if args.self_test:
        with tempfile.TemporaryDirectory() as directory:
            receiver = UdpReceiver(port=0)
            controller = TypingController(
                AppPaths.for_root(directory),
                TypingSettings(),
                geometry.width(),
                geometry.height(),
                receiver=receiver,
            )
            window = StartupWindow(controller, TypingSettings(), AppPaths.for_root(directory))
            heartbeat = Heartbeat(1.0, True, True, "self-test", LAYOUT_VERSION, 30.0)
            assert decode_message(encode_message(heartbeat)) == heartbeat
            assert window.dwell_spin.value() == 1.0
            controller.close()
            window.close()
        return 0

    try:
        controller = TypingController(
            paths,
            load_settings(paths.settings_file),
            geometry.width(),
            geometry.height(),
        )
    except OSError as error:
        logging.getLogger("pure_gaze_typing").exception("UDP 端口绑定失败")
        from PyQt6.QtWidgets import QMessageBox

        QMessageBox.critical(None, "纯眼动打字器", f"端口 9101 无法使用：{error}")
        return 2

    startup = StartupWindow(controller, controller.settings, paths)
    typing_window: TypingWindow | None = None

    def start_typing(settings: TypingSettings) -> None:
        nonlocal typing_window
        controller.start_session()
        typing_window = TypingWindow(controller, settings)
        typing_window.closed.connect(startup.show)
        startup.hide()
        typing_window.showFullScreen()

    startup.start_requested.connect(start_typing)
    startup.show()
    exit_code = app.exec()
    controller.close()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
