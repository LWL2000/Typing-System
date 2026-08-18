from pathlib import Path
import time

import numpy as np
import pytest
from PyQt6.QtCore import QObject, Qt, pyqtSignal

import pure_gaze_typing.capture_window as capture_module
import pure_gaze_typing.capture_worker as worker_module
from pure_gaze_typing.calibration import CalibrationEnvironment, stable_median_prediction
from pure_gaze_typing.calibration import CalibrationMetadata, CalibrationMode
from pure_gaze_typing.capture_worker import CameraWorker
from pure_gaze_typing.capture_worker import CapturePacket
from pure_gaze_typing.capture_window import CaptureController, CaptureWindow
from pure_gaze_typing.eyetrax_runtime import FrameObservation, GazeEstimate
from pure_gaze_typing.layout import build_layout
from pure_gaze_typing.layout import reentry_calibration_points
from pure_gaze_typing.paths import AppPaths


class FakeCaptureController(QObject):
    camera_state_changed = pyqtSignal(bool, str)
    calibration_point_changed = pyqtSignal(str)
    calibration_finished = pyqtSignal(bool, str, object)
    stream_state_changed = pyqtSignal(bool, str)
    preview_ready = pyqtSignal(object)
    shutdown_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.calibration_calls = []
        self.cancel_calls = 0
        self.stop_calls = 0
        self.camera_calls = []
        self.stop_stream_calls = 0
        self.start_stream_calls = 0
        self.retry_failed_calls = 0

    def start_camera(self, index):
        self.camera_calls.append(index)

    def start_calibration(self, mode, validate=False, grid_rows=3, grid_columns=3):
        self.calibration_calls.append((mode, validate, grid_rows, grid_columns))

    def cancel_calibration(self):
        self.cancel_calls += 1

    def start_streaming(self):
        self.start_stream_calls += 1

    def stop_streaming(self):
        self.stop_stream_calls += 1

    def retry_failed_regions(self):
        self.retry_failed_calls += 1

    def save_anyway(self):
        return None

    def stop(self):
        self.stop_calls += 1

    def finish_shutdown(self):
        self.shutdown_finished.emit()


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


class ReadyRuntime:
    def __init__(self, metadata):
        self.metadata = metadata

    def close(self):
        return None


class RecordingStore:
    def __init__(self, root: Path):
        self.saved = []
        self.root = root

    def save(self, model, metadata):
        self.saved.append(metadata)
        path = self.root / "model.pkl"
        model.save_model(path)
        return object()


class FrameRuntime:
    def __init__(self):
        self.closed = False
        self.metadata = None

    def extract(self, _frame):
        return FrameObservation(np.ones(2), True, False)

    def estimate(self, _observation, *, timestamp):
        return GazeEstimate(True, True, False, 1.0, 10.0, 20.0, 10.0, 20.0)

    def close(self):
        self.closed = True


class FakeVideoCapture:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        return True, np.zeros((12, 16, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeTimerSignal:
    def connect(self, callback):
        self.callback = callback


class FakeTimer:
    def __init__(self, _parent):
        self.timeout = FakeTimerSignal()
        self._interval = -1

    def start(self, interval=0):
        self._interval = interval

    def stop(self):
        return None

    def interval(self):
        return self._interval


def test_camera_worker_caps_frame_timer_instead_of_running_unbounded(monkeypatch):
    capture = FakeVideoCapture()
    monkeypatch.setattr(worker_module.cv2, "VideoCapture", lambda _index: capture)
    monkeypatch.setattr(worker_module, "QTimer", FakeTimer)
    worker = CameraWorker(0, FrameRuntime(), None)

    worker.start()

    assert worker._timer.interval() == 33
    worker.stop()
    assert capture.released


def test_camera_runtime_is_constructed_in_worker_start_not_ui_constructor(monkeypatch):
    capture = FakeVideoCapture()
    runtime = FrameRuntime()
    calls = []
    monkeypatch.setattr(worker_module.cv2, "VideoCapture", lambda _index: capture)
    monkeypatch.setattr(worker_module, "QTimer", FakeTimer)

    worker = CameraWorker(
        0,
        None,
        None,
        runtime_factory=lambda *args: calls.append(args) or runtime,
        runtime_args=("face.task", 1920, 1080),
    )

    assert calls == []
    worker.start()
    assert calls == [("face.task", 1920, 1080)]
    worker.stop()
    assert runtime.closed


def test_slow_runtime_initialization_does_not_block_the_ui_thread(qtbot, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(worker_module.cv2, "VideoCapture", lambda _index: FakeVideoCapture())

    def slow_runtime(*_args):
        time.sleep(0.25)
        return FrameRuntime()

    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda index: CalibrationEnvironment(1920, 1080, 1.0, index, "gaze-grid-v1"),
        tmp_path / "face_landmarker.task",
        runtime_factory=slow_runtime,
        publisher_factory=FakePublisher,
    )
    states = []
    controller.camera_state_changed.connect(lambda ready, message: states.append((ready, message)))

    started_at = time.monotonic()
    controller.start_camera(0)
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.15
    qtbot.waitUntil(lambda: any(ready for ready, _message in states), timeout=1000)
    with qtbot.waitSignal(controller.shutdown_finished, timeout=2000):
        controller.stop()


def test_camera_worker_emits_preview_at_ten_fps_while_preserving_gaze_packets(monkeypatch):
    worker = CameraWorker(0, FrameRuntime(), None)
    worker._capture = FakeVideoCapture()
    timestamps = iter((0.00, 0.03, 0.06, 0.11))
    monkeypatch.setattr(worker_module.time, "monotonic", lambda: next(timestamps))
    packets = []
    previews = []
    worker.packet_ready.connect(packets.append)
    worker.preview_ready.connect(previews.append)

    for _index in range(4):
        worker._read_frame()

    assert len(packets) == 4
    assert len(previews) == 2


def test_primary_controls_are_clickable_even_before_camera_is_ready(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    assert not window.validation_checkbox.isChecked()
    assert window.calibrate_button.isEnabled()
    assert window.stream_button.isEnabled()
    assert window.stop_stream_button.isEnabled()
    controller.camera_state_changed.emit(True, "摄像头 0 已连接")
    assert window.calibrate_button.isEnabled()
    assert window.camera_status_label.text() == "摄像头 0 已连接"
    controller.stream_state_changed.emit(True, "眼动数据正在输出")
    assert window.stream_button.isEnabled()
    assert window.stop_stream_button.isEnabled()
    assert window.stop_stream_button.text() == "暂停输出"


def test_pause_output_has_visible_feedback_even_while_preview_continues(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)

    controller.stream_state_changed.emit(False, "眼动输出已暂停（摄像头预览继续）")

    assert "预览继续" in window.result_label.text()
    assert window.stream_button.text() == "恢复输出"


def test_pause_output_responds_on_mouse_press(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.show()
    controller.stream_state_changed.emit(True, "眼动数据正在输出")

    window.stop_stream_button.pressed.emit()

    assert controller.stop_stream_calls == 1
    assert "正在暂停" in window.result_label.text()


def test_resume_output_responds_to_real_mouse_click(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.show()
    controller.camera_state_changed.emit(True, "摄像头 0 已连接")
    controller.stream_state_changed.emit(False, "眼动输出已暂停")

    qtbot.mouseClick(window.stream_button, Qt.MouseButton.LeftButton)

    assert controller.start_stream_calls == 1
    assert "正在恢复" in window.result_label.text()


def test_reconnect_responds_before_camera_initialization(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.show()
    controller.camera_state_changed.emit(True, "摄像头 0 已连接")

    window.connect_button.pressed.emit()

    assert "正在重新连接" in window.camera_status_label.text()
    assert not window.connect_button.isEnabled()
    qtbot.waitUntil(lambda: controller.camera_calls == [0], timeout=300)


def test_window_close_hides_immediately_and_finishes_after_async_shutdown(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.show()

    window.close()

    assert controller.stop_calls == 1
    assert not window.isVisible()
    assert window._closing

    controller.finish_shutdown()

    assert window._shutdown_complete


def test_prediction_summary_rejects_outlier_and_requires_eight_samples():
    samples = [(500.0 + index, 300.0 - index) for index in range(10)]
    samples.append((5000.0, -4000.0))

    prediction = stable_median_prediction(samples, min_samples=8, max_samples=20)

    assert prediction == pytest.approx((504.5, 295.5), abs=1.0)
    with pytest.raises(RuntimeError, match="at least 8"):
        stable_median_prediction(samples[:7], min_samples=8, max_samples=20)


def test_prediction_stage_extends_a_short_window_for_low_camera_fps(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda camera_index: CalibrationEnvironment(
            1920, 1080, 1.0, camera_index, "gaze-grid-v1"
        ),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    controller._calibration_active = True
    controller._start_prediction_stage("bias", ("target_7",))
    outcomes = []
    controller.calibration_finished.connect(
        lambda passed, message, failed: outcomes.append((passed, message, failed))
    )

    for timestamp in (0.0, 0.25, 0.35, 0.45, 0.55, 0.70):
        controller._process_prediction_stage(
            CapturePacket(
                timestamp,
                FrameObservation(np.ones(2), True, False),
                GazeEstimate(True, True, False, 1.0, 10.0, 20.0, 10.0, 20.0),
                10.0,
            )
        )

    assert outcomes == []
    assert controller._prediction_stage == "bias"
    controller.stop()


def test_loaded_calibration_starts_output_when_camera_becomes_ready(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda index: CalibrationEnvironment(1920, 1080, 1.0, index, "gaze-grid-v3-reference"),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    metadata = CalibrationMetadata(
        "cal-loaded",
        "2026-08-11T12:00:00+08:00",
        CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v3-reference"),
        (0.0, 0.0),
        (2.0, 2.0),
    )
    controller.worker = object()
    controller.runtime = ReadyRuntime(metadata)
    states = []
    controller.stream_state_changed.connect(lambda running, message: states.append((running, message)))

    controller._on_camera_state(True, "摄像头 0 已连接")

    assert controller._streaming
    assert states[-1] == (True, "眼动数据正在输出")
    controller.worker = None
    controller.stop()


def test_calibration_scene_uses_shared_layout_centers(qtbot, tmp_path: Path):
    window = CaptureWindow(FakeCaptureController(), AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.resize(1280, 720)
    window.show_calibration_point("target_3")
    assert window.highlight_center() == pytest.approx(
        build_layout(1280, 720).targets[3].center
    )


def test_calibration_button_opens_fullscreen_mode_menu_with_estimates(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window._begin_calibration()

    assert window.mode_dialog is not None
    assert window.mode_dialog.isFullScreen()
    assert "25–35 秒" in window.mode_dialog.fast_button.text()
    assert "50–60 秒" in window.mode_dialog.precise_button.text()
    assert window.mode_dialog.grid_rows_spin.value() == 3
    assert window.mode_dialog.grid_columns_spin.value() == 3
    assert controller.calibration_calls == []


def test_mode_menu_starts_selected_calibration_with_validation_choice(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.validation_checkbox.setChecked(True)
    window._begin_calibration()
    window.mode_dialog.grid_rows_spin.setValue(4)
    window.mode_dialog.grid_columns_spin.setValue(5)

    qtbot.mouseClick(window.mode_dialog.precise_button, Qt.MouseButton.LeftButton)

    assert controller.calibration_calls == [(CalibrationMode.PRECISE, True, 4, 5)]
    assert window._calibration_mode
    assert not window._controls.isVisible()


def test_precise_calibration_uses_selected_initial_grid(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda index: CalibrationEnvironment(1920, 1080, 1.0, index, "gaze-grid-v1"),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    controller.environment = CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v1")
    controller._camera_ready = True

    controller.start_calibration(
        CalibrationMode.PRECISE,
        False,
        grid_rows=4,
        grid_columns=5,
    )

    assert controller._session is not None
    grid_ids = [point.target_id for point in controller._session.points if point.target_id.startswith("grid_")]
    assert len(grid_ids) == 20
    assert grid_ids[0] == "grid_0_0"
    assert grid_ids[-1] == "grid_3_0"
    controller.stop()


def test_escape_from_mode_menu_returns_without_starting(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.show()
    window._begin_calibration()

    qtbot.keyClick(window.mode_dialog, Qt.Key.Key_Escape)

    assert controller.calibration_calls == []
    assert not window._calibration_mode
    assert window._controls.isVisible()


def test_failed_region_retry_reenters_fullscreen_calibration_scene(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.show()

    controller.calibration_point_changed.emit("target_4")

    assert window.isFullScreen()
    assert not window._controls.isVisible()
    assert window.highlight_center() == pytest.approx(build_layout(window.width(), window.height()).targets[4].center)


def test_failed_validation_exposes_retry_and_save_actions(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    controller.calibration_finished.emit(False, "命中 4/6", ("target_1", "target_4"))
    assert window.retry_failed_button.isVisibleTo(window)
    assert not window.save_anyway_button.isVisibleTo(window)


def test_retry_failed_regions_responds_to_a_real_mouse_click(qtbot, tmp_path: Path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    window.show()
    controller.calibration_finished.emit(False, "target_7 有效注视样本不足", ("target_7",))

    qtbot.mouseClick(window.retry_failed_button, Qt.MouseButton.LeftButton)

    assert controller.retry_failed_calls == 1
    assert "正在重校" in window.result_label.text()


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


def test_saved_calibration_automatically_resumes_output(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda index: CalibrationEnvironment(1920, 1080, 1.0, index, "gaze-grid-v3-reference"),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    metadata = CalibrationMetadata(
        "cal-saved",
        "2026-08-11T12:00:00+08:00",
        CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v3-reference"),
        (0.0, 0.0),
        (2.0, 2.0),
    )
    controller.worker = object()
    controller.runtime = ReadyRuntime(metadata)
    controller._calibration_active = True
    states = []
    controller.stream_state_changed.connect(lambda running, message: states.append((running, message)))

    controller._on_model_saved(object())

    assert controller._streaming
    assert states[-1] == (True, "眼动数据正在输出")
    controller.worker = None
    controller.stop()


def test_runtime_initialization_failure_is_reported_without_escaping(qtbot, tmp_path: Path):
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

    qtbot.waitUntil(lambda: bool(states), timeout=1000)
    assert states == [(False, "眼动运行时初始化失败：模型无法加载")]
    if not controller._shutdown_emitted:
        with qtbot.waitSignal(controller.shutdown_finished, timeout=2000):
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


def test_stop_camera_never_waits_on_the_ui_thread(monkeypatch, tmp_path: Path):
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
        def request_stop(self):
            events.append("worker_request_stop")

    worker = FakeWorker()

    class FakeThread(QObject):
        finished = pyqtSignal()

        def __init__(self):
            super().__init__()

        def isRunning(self):
            return True

        def quit(self):
            events.append("thread_quit")

        def wait(self, timeout):
            events.append(("thread_wait", timeout))
            raise AssertionError("UI thread must never wait for the camera thread")

        def requestInterruption(self):
            events.append("thread_interrupt")

        def terminate(self):
            events.append("thread_terminate")

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
    thread = FakeThread()
    controller.thread = thread

    controller.stop_camera()

    assert events[0][0] == "worker_stop"
    assert not any(isinstance(event, tuple) and event[0] == "thread_wait" for event in events)
    assert controller.worker is worker
    assert controller.thread is thread


def test_camera_packets_are_coalesced_before_the_ui_processes_them(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda camera_index: CalibrationEnvironment(
            1920, 1080, 1.0, camera_index, "gaze-grid-v1"
        ),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    packets = [
        CapturePacket(
            float(index),
            FrameObservation(np.ones(2), True, False),
            GazeEstimate(True, True, False, 1.0, 10.0, 20.0, 10.0, 20.0),
            30.0,
        )
        for index in range(100)
    ]
    processed = []
    controller._on_packet = processed.append

    for packet in packets:
        controller._queue_latest_packet(packet)

    assert processed == []
    controller._drain_latest_packet()
    assert processed == [packets[-1]]
    controller.stop()


def test_sustained_leave_then_return_starts_five_point_reentry_calibration(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda index: CalibrationEnvironment(1920, 1080, 1.0, index, "gaze-grid-v1"),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    metadata = CalibrationMetadata(
        "cal-ready",
        "2026-08-11T12:00:00+08:00",
        CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v1"),
        (0.0, 0.0),
        (2.0, 2.0),
    )
    controller.environment = metadata.environment
    controller.runtime = ReadyRuntime(metadata)
    controller.worker = object()
    controller._camera_ready = True
    controller._streaming = True
    points = []
    controller.calibration_point_changed.connect(points.append)

    def packet(timestamp, face_detected):
        return CapturePacket(
            timestamp,
            FrameObservation(None, face_detected, False),
            GazeEstimate(False, face_detected, False, 0.0),
            30.0,
        )

    controller._on_packet(packet(0.0, False))
    controller._on_packet(packet(1.5, False))
    assert controller._reentry_pending
    assert not controller._streaming

    controller._on_packet(packet(1.6, True))

    assert controller._calibration_active
    assert controller._prediction_stage == "reentry"
    assert points[-1] == "reentry_top_left"
    controller.worker = None
    controller.stop()


def test_five_point_reentry_correction_is_saved_and_resumes_output(tmp_path: Path):
    controller = CaptureController(
        AppPaths.for_root(tmp_path),
        lambda index: CalibrationEnvironment(1920, 1080, 1.0, index, "gaze-grid-v1"),
        tmp_path / "face_landmarker.task",
        publisher_factory=FakePublisher,
    )
    environment = CalibrationEnvironment(1920, 1080, 1.0, 0, "gaze-grid-v1")
    metadata = CalibrationMetadata(
        "cal-before-reentry",
        "2026-08-11T12:00:00+08:00",
        environment,
        (0.0, 0.0),
        (2.0, 2.0),
    )
    runtime = ReadyRuntime(metadata)
    controller.environment = environment
    controller.runtime = runtime
    controller.worker = object()
    controller._camera_ready = True
    controller._calibration_active = True
    controller._reentry_pending = True
    controller._provisional_metadata = metadata
    layout = build_layout(1920, 1080)
    controller._reentry_predictions = {
        name: (point[0] - 24.0, point[1] + 16.0)
        for name, point in reentry_calibration_points(layout)
    }
    configured = []
    saved = []
    controller._configure_requested.connect(configured.append)
    controller._save_requested.connect(saved.append)

    controller._finish_reentry_correction()

    assert len(configured) == 1
    assert configured[0].calibration_id.startswith("cal-reentry-")
    assert configured[0].screen_affine
    controller._on_model_configured(configured[0])
    assert saved == configured
    runtime.metadata = configured[0]
    controller._on_model_saved(object())
    assert controller._streaming
    assert not controller._reentry_pending
    controller.worker = None
    controller.stop()
