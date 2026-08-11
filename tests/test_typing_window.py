from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal

from pure_gaze_typing.layout import LAYOUT_VERSION
from pure_gaze_typing.paths import AppPaths
from pure_gaze_typing.protocol import GazeSample, Heartbeat
from pure_gaze_typing.settings import TypingSettings
from pure_gaze_typing.typing_controller import ConnectionStatus, TypingController
from pure_gaze_typing.typing_engine import PageKind
from pure_gaze_typing.typing_window import StartupWindow, TypingWindow


class FakeReceiver:
    def __init__(self):
        self.online = True

    def poll(self):
        return []

    def is_online(self, _now, timeout=2.0):
        return self.online

    def close(self):
        return None


class FakeTypingController(QObject):
    status_changed = pyqtSignal(object)
    update_ready = pyqtSignal(object)
    session_finished = pyqtSignal(str)

    def update_settings(self, _settings):
        return None

    def tick(self, _now):
        return None

    def end_session(self):
        return None


def valid_sample(timestamp: float, x: float, y: float, *, blink: bool = False) -> GazeSample:
    return GazeSample(
        timestamp,
        not blink,
        True,
        blink,
        0.9 if not blink else 0.0,
        30.0,
        "cal-1",
        LAYOUT_VERSION,
        x if not blink else None,
        y if not blink else None,
    )


def make_controller(root: Path, show_gaze_point: bool) -> TypingController:
    controller = TypingController(
        AppPaths.for_root(root),
        TypingSettings(show_gaze_point, 1.0, False),
        1920,
        1080,
        receiver=FakeReceiver(),
    )
    controller.process_message(
        Heartbeat(0.0, True, True, "cal-1", LAYOUT_VERSION, 30.0),
        0.0,
    )
    for index in range(5):
        controller.process_message(valid_sample(index * 0.05, 268.8, 248.4), index * 0.05)
    return controller


def test_compatible_heartbeat_marks_capture_online_without_valid_gaze(tmp_path: Path):
    controller = TypingController(
        AppPaths.for_root(tmp_path),
        TypingSettings(),
        1920,
        1080,
        receiver=FakeReceiver(),
    )

    controller.process_message(
        Heartbeat(0.0, True, True, "cal-1", LAYOUT_VERSION, 30.0),
        0.0,
    )

    assert controller.status.online
    assert controller.status.calibration_compatible
    assert not controller.status.gaze_ready
    assert "等待有效眼动数据" in controller.status.message


def test_valid_gaze_marks_stream_ready(tmp_path: Path):
    controller = TypingController(
        AppPaths.for_root(tmp_path),
        TypingSettings(),
        1920,
        1080,
        receiver=FakeReceiver(),
    )
    controller.process_message(
        Heartbeat(0.0, True, True, "cal-1", LAYOUT_VERSION, 30.0),
        0.0,
    )

    controller.process_message(valid_sample(0.1, 960.0, 540.0), 0.1)

    assert controller.status.gaze_ready
    assert controller.status.message == "眼动数据已就绪"


def test_hidden_gaze_point_does_not_change_selection(tmp_path: Path):
    visible = make_controller(tmp_path / "visible", True)
    hidden = make_controller(tmp_path / "hidden", False)
    for timestamp in (0.7, 1.2):
        sample = valid_sample(timestamp, 268.8, 248.4)
        visible.process_message(sample, timestamp)
        hidden.process_message(sample, timestamp)
    assert visible.engine.page_kind == hidden.engine.page_kind == PageKind.SUBMENU
    assert visible.last_triggered_target == hidden.last_triggered_target == "main_group_0"


def test_center_prepare_waits_for_initial_eye_movement(tmp_path: Path):
    controller = make_controller(tmp_path, True)
    controller.start_session()
    for offset in (0.0, 0.1, 0.2):
        controller.process_message(
            valid_sample(1.0 + offset, 100.0, 100.0),
            1.0 + offset,
        )

    assert controller.drift._points == []
    controller.end_session()


def test_start_button_requires_live_gaze_in_addition_to_compatible_calibration(qtbot, tmp_path: Path):
    controller = FakeTypingController()
    window = StartupWindow(controller, TypingSettings(), AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    assert not window.start_button.isEnabled()
    controller.status_changed.emit(ConnectionStatus(True, True, "等待有效眼动数据"))
    assert not window.start_button.isEnabled()
    controller.status_changed.emit(ConnectionStatus(True, True, "眼动数据已就绪", True))
    assert window.start_button.isEnabled()
    assert window.gaze_checkbox.isChecked()
    assert window.dwell_spin.value() == 1.0


def test_startup_window_polls_for_udp_before_session_starts(qtbot, tmp_path: Path):
    class PollingController(FakeTypingController):
        def __init__(self):
            super().__init__()
            self.tick_count = 0

        def tick(self, _now):
            self.tick_count += 1
            self.status_changed.emit(ConnectionStatus(True, True, "connected", True))

    controller = PollingController()
    window = StartupWindow(controller, TypingSettings(), AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.show()

    qtbot.waitUntil(lambda: controller.tick_count > 0, timeout=300)

    assert window.start_button.isEnabled()


def test_typing_window_hides_only_the_gaze_dot(qtbot, tmp_path: Path):
    controller = make_controller(tmp_path, False)
    window = TypingWindow(controller, controller.settings)
    qtbot.addWidget(window)
    update = controller.current_update()
    controller.update_ready.emit(update)
    assert not window.canvas.show_gaze_point
    assert window.canvas.target_labels


def test_send_commits_a_newline_without_finishing_the_session(tmp_path: Path):
    controller = make_controller(tmp_path, True)
    controller.start_session(skip_prepare=True)
    controller.engine.current_line = "HELLO"
    recorder = controller.recorder

    controller._activate("main_send")

    assert controller.recorder is recorder
    assert controller.engine.current_line == ""
    assert controller.engine.full_text() == "HELLO\n"
    assert not recorder.result_path.exists()
    controller.end_session()
    assert recorder.result_path.read_text(encoding="utf-8") == "HELLO\n"


def test_typing_window_does_not_close_after_send_signal(qtbot, tmp_path: Path):
    controller = make_controller(tmp_path, True)
    controller.start_session(skip_prepare=True)
    window = TypingWindow(controller, controller.settings)
    qtbot.addWidget(window)
    window.show()

    controller.session_finished.emit("HELLO")
    qtbot.wait(550)

    assert window.isVisible()
    window.close()


def test_history_view_is_read_only_scrollable_and_at_most_three_lines_high(qtbot, tmp_path: Path):
    controller = make_controller(tmp_path, True)
    window = TypingWindow(controller, controller.settings)
    qtbot.addWidget(window)
    window.resize(1920, 1080)
    window.show()
    window._timer.stop()
    update = replace(controller.current_update(), current_text="ONE\nTWO\nTHREE\nFOUR\nFIVE")

    window.canvas.set_controller_update(update)
    qtbot.wait(50)

    history = window.canvas.history_view
    assert history.isReadOnly()
    assert history.toPlainText() == update.current_text
    assert history.verticalScrollBar().maximum() == 2
    assert history.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded


def test_three_blinks_return_from_letter_page(tmp_path: Path):
    controller = make_controller(tmp_path, True)
    controller.engine.activate("main_group_0") if controller.engine.page_kind is PageKind.MAIN else None
    for start in (2.0, 2.6, 3.2):
        controller.process_message(valid_sample(start, 960.0, 540.0, blink=True), start)
        controller.process_message(valid_sample(start + 0.12, 960.0, 540.0), start + 0.12)
    assert controller.engine.page_kind is PageKind.MAIN
