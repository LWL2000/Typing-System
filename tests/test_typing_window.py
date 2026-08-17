import csv
import json
from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal
import pytest

from pure_gaze_typing.layout import LAYOUT_VERSION
from pure_gaze_typing.paths import AppPaths
from pure_gaze_typing.protocol import GazeSample, Heartbeat
from pure_gaze_typing.settings import TypingSettings, load_settings
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


def ready_session_controller(
    root: Path,
    settings: TypingSettings | None = None,
) -> TypingController:
    controller = TypingController(
        AppPaths.for_root(root),
        settings or TypingSettings(),
        1920,
        1080,
        receiver=FakeReceiver(),
    )
    controller.process_message(
        Heartbeat(0.0, True, True, "cal-1", LAYOUT_VERSION, 30.0),
        0.0,
    )
    controller.process_message(valid_sample(0.1, 960.0, 540.0), 0.1)
    controller.start_session(skip_prepare=True)
    return controller


def feed_dwell(
    controller: TypingController,
    point: tuple[float, float],
    *,
    start: float,
) -> None:
    for index in range(24):
        timestamp = start + index * 0.05
        controller.process_message(valid_sample(timestamp, *point), timestamp)


def read_events(controller: TypingController) -> list[dict[str, str]]:
    assert controller.recorder is not None
    with controller.recorder.events_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


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
    assert visible.adaptive.snapshot().matrix_version == hidden.adaptive.snapshot().matrix_version


def test_disabled_adaptation_preserves_center_only_coordinates(tmp_path: Path):
    controller = ready_session_controller(
        tmp_path,
        TypingSettings(adaptive_correction_enabled=False),
    )
    expected = controller.drift.apply(300.0, 220.0)

    update = controller.process_message(valid_sample(1.0, 300.0, 220.0), 1.0)

    assert update.gaze_point == pytest.approx(expected)
    assert controller.adaptive.snapshot().matrix_version == 0
    controller.end_session()


def test_core_dwell_updates_affine_and_records_event(tmp_path: Path):
    controller = ready_session_controller(tmp_path)

    feed_dwell(controller, (250.0, 235.0), start=1.0)

    assert controller.last_triggered_target == "main_group_0"
    assert controller.adaptive.snapshot().matrix_version == 1
    rows = read_events(controller)
    accepted = next(row for row in rows if row["event"] == "adaptive_update_accepted")
    payload = json.loads(accepted["payload_json"])
    assert payload["matrix_version"] == 1
    assert payload["huber_weight"] > 0.0
    controller.end_session()


def test_visible_and_hidden_gaze_use_the_same_adaptive_selection(tmp_path: Path):
    visible = ready_session_controller(
        tmp_path / "visible_adaptive",
        TypingSettings(show_gaze_point=True),
    )
    hidden = ready_session_controller(
        tmp_path / "hidden_adaptive",
        TypingSettings(show_gaze_point=False),
    )

    feed_dwell(visible, (250.0, 235.0), start=1.0)
    feed_dwell(hidden, (250.0, 235.0), start=1.0)

    assert visible.last_triggered_target == hidden.last_triggered_target == "main_group_0"
    assert visible.adaptive.snapshot().matrix_version == 1
    assert hidden.adaptive.snapshot().matrix_version == 1
    visible.end_session()
    hidden.end_session()


def test_new_session_replaces_the_adaptive_matrix(tmp_path: Path):
    controller = ready_session_controller(tmp_path)
    feed_dwell(controller, (250.0, 235.0), start=1.0)
    previous_adaptive = controller.adaptive
    assert previous_adaptive.snapshot().matrix_version == 1

    controller.start_session(skip_prepare=True)

    assert controller.adaptive is not previous_adaptive
    assert controller.adaptive.snapshot().matrix_version == 0
    assert controller.adaptive.apply(300.0, 220.0) == pytest.approx((300.0, 220.0))
    controller.end_session()


def test_adaptive_runtime_error_falls_back_without_closing_session(
    monkeypatch,
    tmp_path: Path,
):
    controller = ready_session_controller(tmp_path)

    def fail_observe(*_args, **_kwargs):
        raise RuntimeError("simulated adaptive failure")

    monkeypatch.setattr(controller.adaptive, "observe", fail_observe)
    expected = controller.drift.apply(300.0, 220.0)

    update = controller.process_message(valid_sample(1.0, 300.0, 220.0), 1.0)

    assert update.gaze_point == pytest.approx(expected)
    assert controller.recorder is not None
    assert not controller.adaptive.snapshot().enabled
    assert "adaptive_disabled" in {row["event"] for row in read_events(controller)}
    controller.end_session()


def test_calibration_change_resets_adaptive_matrix(tmp_path: Path):
    controller = ready_session_controller(tmp_path)
    feed_dwell(controller, (250.0, 235.0), start=1.0)
    assert controller.adaptive.snapshot().matrix_version == 1

    controller.process_message(
        Heartbeat(3.0, True, True, "cal-2", LAYOUT_VERSION, 30.0),
        3.0,
    )

    assert controller.adaptive.snapshot().matrix_version == 0
    assert controller.adaptive.apply(300.0, 220.0) == pytest.approx((300.0, 220.0))
    assert "adaptive_reset" in {row["event"] for row in read_events(controller)}
    controller.end_session()


def test_gaze_calibration_mismatch_resets_adaptive_matrix(tmp_path: Path):
    controller = ready_session_controller(tmp_path)
    feed_dwell(controller, (250.0, 235.0), start=1.0)
    assert controller.adaptive.snapshot().matrix_version == 1
    mismatched = replace(valid_sample(3.0, 300.0, 220.0), calibration_id="cal-2")

    controller.process_message(mismatched, 3.0)

    assert controller.adaptive.snapshot().matrix_version == 0
    assert controller.adaptive.apply(300.0, 220.0) == pytest.approx((300.0, 220.0))
    controller.end_session()


def test_submenu_anchor_uses_submenu_rectangle(tmp_path: Path):
    controller = ready_session_controller(tmp_path)
    feed_dwell(controller, (250.0, 235.0), start=1.0)
    assert controller.engine.page_kind is PageKind.SUBMENU
    controller.process_message(valid_sample(2.3, 960.0, 540.0), 2.3)
    controller.process_message(valid_sample(2.7, 960.0, 540.0), 2.7)

    feed_dwell(controller, controller.layout.submenu_targets[3].center, start=3.0)

    assert controller.last_triggered_target == "letter_D"
    assert controller.adaptive.last_target_id == "target_3"
    assert controller.adaptive.snapshot().matrix_version == 2
    controller.end_session()


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
    assert window.adaptive_checkbox.text() == "实时自适应校正（推荐）"
    assert window.adaptive_checkbox.isChecked()
    assert window.dwell_spin.value() == 1.0
    assert window.blink_count_spin.value() == 3


def test_startup_saves_disabled_adaptive_choice(qtbot, tmp_path: Path):
    paths = AppPaths.for_root(tmp_path)
    window = StartupWindow(FakeTypingController(), TypingSettings(), paths)
    qtbot.addWidget(window)
    window.adaptive_checkbox.setChecked(False)
    window.blink_count_spin.setValue(2)

    window._start()

    assert load_settings(paths.settings_file).adaptive_correction_enabled is False
    assert load_settings(paths.settings_file).blink_return_count == 2


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


def test_configured_two_blinks_return_from_letter_page(tmp_path: Path):
    controller = ready_session_controller(
        tmp_path,
        TypingSettings(blink_return_count=2),
    )
    controller.engine.activate("main_group_0")

    for start in (2.0, 2.6):
        controller.process_message(valid_sample(start, 960.0, 540.0, blink=True), start)
        controller.process_message(valid_sample(start + 0.12, 960.0, 540.0), start + 0.12)

    assert controller.engine.page_kind is PageKind.MAIN
    assert controller.current_update().blink_required == 2


def test_paused_capture_heartbeat_is_visible_and_not_gaze_ready(tmp_path: Path):
    controller = TypingController(
        AppPaths.for_root(tmp_path),
        TypingSettings(),
        1920,
        1080,
        receiver=FakeReceiver(),
    )

    controller.process_message(
        Heartbeat(
            0.0,
            True,
            True,
            "cal-1",
            LAYOUT_VERSION,
            30.0,
            streaming=False,
        ),
        0.0,
    )

    assert controller.status.online
    assert controller.status.calibration_compatible
    assert not controller.status.gaze_ready
    assert "暂停" in controller.status.message
