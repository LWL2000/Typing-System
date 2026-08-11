from pathlib import Path

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

import pure_gaze_typing.capture_window as capture_module
from pure_gaze_typing.calibration import CalibrationEnvironment
from pure_gaze_typing.capture_window import CaptureController, CaptureWindow
from pure_gaze_typing.layout import build_layout
from pure_gaze_typing.paths import AppPaths


class FakeCaptureController(QObject):
    camera_state_changed = pyqtSignal(bool, str)
    calibration_point_changed = pyqtSignal(str)
    calibration_finished = pyqtSignal(bool, str, object)
    stream_state_changed = pyqtSignal(bool, str)
    preview_ready = pyqtSignal(object)

    def start_camera(self, _index):
        return None

    def start_calibration(self, _validate):
        return None

    def start_streaming(self):
        return None

    def stop_streaming(self):
        return None

    def retry_failed_regions(self):
        return None

    def save_anyway(self):
        return None

    def stop(self):
        return None


class FakePublisher:
    def send(self, _message):
        return None

    def close(self):
        return None


def test_validation_is_off_by_default_and_calibration_button_tracks_camera(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    assert not window.validation_checkbox.isChecked()
    assert not window.calibrate_button.isEnabled()
    controller.camera_state_changed.emit(True, "摄像头 0 已连接")
    assert window.calibrate_button.isEnabled()
    assert window.camera_status_label.text() == "摄像头 0 已连接"


def test_calibration_scene_uses_shared_layout_centers(qtbot, tmp_path: Path):
    window = CaptureWindow(FakeCaptureController(), AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.resize(1280, 720)
    window.show_calibration_point("target_3")
    assert window.highlight_center() == pytest.approx(
        build_layout(1280, 720).targets[3].center
    )


def test_failed_validation_exposes_retry_and_save_actions(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    controller.calibration_finished.emit(False, "命中 4/6", ("target_1", "target_4"))
    assert window.retry_failed_button.isVisibleTo(window)
    assert window.save_anyway_button.isVisibleTo(window)


def test_runtime_initialization_failure_is_reported_without_escaping(tmp_path: Path):
    def failing_runtime(*_args, **_kwargs):
        raise FileNotFoundError("模型无法加载")

    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda camera_index: CalibrationEnvironment(
            1920, 1080, 1.0, camera_index, "gaze-grid-v1"
        ),
        tmp_path / "face_landmarker.task",
        runtime_factory=failing_runtime,
        publisher_factory=FakePublisher,
    )
    states = []
    controller.camera_state_changed.connect(
        lambda ready, message: states.append((ready, message))
    )

    controller.start_camera(0)

    assert states == [(False, "眼动运行时初始化失败：模型无法加载")]
    controller.stop()


def test_stop_camera_stops_worker_before_quitting_thread(monkeypatch, tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda camera_index: CalibrationEnvironment(
            1920, 1080, 1.0, camera_index, "gaze-grid-v1"
        ),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    events = []

    class FakeWorker:
        stopped = False

    worker = FakeWorker()

    class FakeThread:
        def isRunning(self):
            return True

        def quit(self):
            assert worker.stopped
            events.append("thread_quit")

        def wait(self, timeout):
            events.append(("thread_wait", timeout))
            return True

    class FakeMetaObject:
        @staticmethod
        def invokeMethod(target, method, connection):
            assert target is worker
            assert method == "stop"
            worker.stopped = True
            events.append(("worker_stop", connection))
            return True

    monkeypatch.setattr(capture_module, "QMetaObject", FakeMetaObject, raising=False)
    controller.worker = worker
    controller.thread = FakeThread()

    controller.stop_camera()

    assert events[0][0] == "worker_stop"
    assert events[1:] == ["thread_quit", ("thread_wait", 5000)]
