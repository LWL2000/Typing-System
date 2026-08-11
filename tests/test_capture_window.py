from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal

import pure_gaze_typing.capture_window as capture_module
from pure_gaze_typing.calibration import CalibrationEnvironment
from pure_gaze_typing.calibration import CalibrationMetadata
from pure_gaze_typing.capture_worker import CameraWorker
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


class ProvisionalRuntime:
    def __init__(self):
        self.metadata = None
        self.trained = None

    def train(self, features, labels):
        self.trained = (features.copy(), labels.copy())

    def feature_range_threshold(self, _features):
        return 4.5

    def set_metadata(self, metadata):
        self.metadata = metadata

    def save_model(self, path):
        Path(path).write_bytes(b"provisional-model")


class RecordingStore:
    def __init__(self, root: Path):
        self.saved = []
        self.root = root

    def save(self, model, metadata):
        self.saved.append(metadata)
        path = self.root / "model.pkl"
        model.save_model(path)
        return object()


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
    assert not window.save_anyway_button.isVisibleTo(window)


def test_worker_training_remains_provisional_until_explicit_promotion(tmp_path: Path):
    runtime = ProvisionalRuntime()
    store = RecordingStore(tmp_path)
    worker = CameraWorker(0, runtime, store)
    metadata = CalibrationMetadata(
        "cal-provisional",
        "2026-08-11T12:00:00+08:00",
        CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v3-reference"),
        (0.0, 0.0),
        (2.0, 2.0),
        calibration_mode="precise",
    )
    trained = []
    saved = []
    worker.model_trained.connect(trained.append)
    worker.model_saved.connect(saved.append)

    worker.train_model(np.ones((8, 2)), np.ones((8, 2)), metadata)

    assert runtime.trained is not None
    assert trained[0].feature_range_threshold == 4.5
    assert store.saved == []

    worker.save_current(trained[0])

    assert store.saved == [trained[0]]
    assert len(saved) == 1


def test_validation_gate_does_not_promote_when_two_regions_fail(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda index: CalibrationEnvironment(1920, 1080, 1.0, index, "gaze-grid-v3-reference"),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    metadata = CalibrationMetadata(
        "cal-provisional",
        "2026-08-11T12:00:00+08:00",
        CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v3-reference"),
        (0.0, 0.0),
        (2.0, 2.0),
    )
    controller._provisional_metadata = metadata
    controller._validation_hits = {
        "target_0": 1,
        "target_1": 1,
        "target_2": 1,
        "target_3": 1,
        "target_4": 0,
        "target_5": 0,
    }
    promoted = []
    outcomes = []
    controller._save_requested.connect(promoted.append)
    controller.calibration_finished.connect(
        lambda passed, message, failed: outcomes.append((passed, message, failed))
    )

    controller._finish_validation()

    assert promoted == []
    assert outcomes[-1][0] is False
    assert outcomes[-1][2] == ("target_4", "target_5")
    controller.stop()


def test_validation_gate_promotes_at_five_of_six(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda index: CalibrationEnvironment(1920, 1080, 1.0, index, "gaze-grid-v3-reference"),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    metadata = CalibrationMetadata(
        "cal-provisional",
        "2026-08-11T12:00:00+08:00",
        CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v3-reference"),
        (0.0, 0.0),
        (2.0, 2.0),
    )
    controller._provisional_metadata = metadata
    controller._validation_hits = {
        "target_0": 1,
        "target_1": 1,
        "target_2": 1,
        "target_3": 1,
        "target_4": 1,
        "target_5": 0,
    }
    promoted = []
    controller._save_requested.connect(promoted.append)

    controller._finish_validation()

    assert len(promoted) == 1
    assert promoted[0].validation_hits == 5
    assert promoted[0].validation_total == 6
    controller.stop()


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


def test_stop_disables_heartbeat_before_closing_resources(qtbot, tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda camera_index: CalibrationEnvironment(
            1920, 1080, 1.0, camera_index, "gaze-grid-v1"
        ),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    assert controller._heartbeat.isActive()

    controller.stop()

    assert not controller._heartbeat.isActive()


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

    class FakeRuntime:
        def close(self):
            assert events[-1] == ("thread_wait", 5000)
            events.append("runtime_close")

    class FakeWorker:
        stopped = False

    worker = FakeWorker()
    runtime = FakeRuntime()

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
    controller.runtime = runtime

    controller.stop_camera()

    assert events[0][0] == "worker_stop"
    assert events[1:] == ["thread_quit", ("thread_wait", 5000), "runtime_close"]
