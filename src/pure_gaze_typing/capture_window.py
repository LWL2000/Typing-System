from __future__ import annotations

from datetime import datetime, timezone
import logging
import time
import uuid

import numpy as np
from PyQt6.QtCore import QMetaObject, QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCloseEvent, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .calibration import (
    CalibrationEnvironment,
    CalibrationMetadata,
    CalibrationPoint,
    CalibrationSession,
    CalibrationStore,
    score_validation,
)
from .capture_worker import CameraWorker, CapturePacket
from .eyetrax_runtime import EyeTraxRuntime
from .layout import build_layout, calibration_points, hit_test
from .paths import AppPaths
from .protocol import GazeSample, Heartbeat, UdpPublisher


LOGGER = logging.getLogger("pure_gaze_typing.capture")


class CaptureController(QObject):
    camera_state_changed = pyqtSignal(bool, str)
    calibration_point_changed = pyqtSignal(str)
    calibration_finished = pyqtSignal(bool, str, object)
    stream_state_changed = pyqtSignal(bool, str)
    preview_ready = pyqtSignal(object)
    _train_requested = pyqtSignal(object, object, object)

    def __init__(
        self,
        paths: AppPaths,
        environment_factory,
        face_model_path,
        *,
        runtime_factory=EyeTraxRuntime,
        publisher_factory=UdpPublisher,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.environment_factory = environment_factory
        self.face_model_path = face_model_path
        self.runtime_factory = runtime_factory
        self.publisher = publisher_factory()
        self.store = CalibrationStore(paths.calibration_dir)
        self.thread: QThread | None = None
        self.worker: CameraWorker | None = None
        self.runtime: EyeTraxRuntime | None = None
        self.environment: CalibrationEnvironment | None = None
        self._camera_ready = False
        self._streaming = False
        self._session: CalibrationSession | None = None
        self._validate_after_training = False
        self._training_sent = False
        self._base_training: tuple[np.ndarray, np.ndarray] | None = None
        self._validation_index: int | None = None
        self._validation_started_at: float | None = None
        self._validation_hits: dict[str, int] = {}
        self._failed_targets: tuple[str, ...] = ()
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(1000)
        self._heartbeat.timeout.connect(self._publish_heartbeat)
        self._heartbeat.start()

    def start_camera(self, camera_index: int) -> None:
        LOGGER.info("camera_start_requested index=%s", camera_index)
        self.stop_camera()
        self.environment = self.environment_factory(int(camera_index))
        LOGGER.info("camera_environment_ready environment=%s", self.environment)
        try:
            runtime = self.runtime_factory(
                self.face_model_path,
                self.environment.screen_width,
                self.environment.screen_height,
            )
        except Exception as error:
            LOGGER.exception("camera_runtime_initialization_failed")
            self.camera_state_changed.emit(False, f"眼动运行时初始化失败：{error}")
            return
        LOGGER.info("camera_runtime_ready runtime=%s", type(runtime).__name__)
        stored = self.store.load(self.environment)
        if stored is not None:
            runtime.load_calibration(stored)
            LOGGER.info("camera_calibration_loaded id=%s", stored.metadata.calibration_id)
        self.runtime = runtime
        self.thread = QThread(self)
        self.worker = CameraWorker(camera_index, runtime, self.store)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.camera_opened.connect(self._on_camera_state)
        self.worker.preview_ready.connect(self.preview_ready)
        self.worker.packet_ready.connect(self._on_packet)
        self.worker.model_saved.connect(self._on_model_saved)
        self.worker.failed.connect(self._on_worker_failure)
        self.worker.stopped.connect(self.worker.deleteLater)
        self.worker.stopped.connect(self.thread.quit)
        self._train_requested.connect(self.worker.train_and_save)
        self.thread.start()
        LOGGER.info("camera_thread_started")

    def start_calibration(self, validate: bool) -> None:
        if not self._camera_ready or self.environment is None:
            self.calibration_finished.emit(False, "请先连接摄像头", ())
            return
        layout = build_layout(self.environment.screen_width, self.environment.screen_height)
        points = tuple(
            CalibrationPoint(name, x, y, 1.0 if name == "center" else 0.8)
            for name, (x, y) in calibration_points(layout)
        )
        self._start_session(points, validate)

    def _start_session(self, points: tuple[CalibrationPoint, ...], validate: bool) -> None:
        self._session = CalibrationSession(points)
        self._validate_after_training = bool(validate)
        self._training_sent = False
        self._validation_index = None
        self._validation_hits.clear()
        if self._session.current_point_id:
            self.calibration_point_changed.emit(self._session.current_point_id)

    def retry_failed_regions(self) -> None:
        if not self._failed_targets or self.environment is None:
            return
        layout = build_layout(self.environment.screen_width, self.environment.screen_height)
        point_map = dict(calibration_points(layout))
        points = tuple(
            CalibrationPoint(target_id, *point_map[target_id], 0.8)
            for target_id in self._failed_targets
        )
        self._start_session(points, True)

    def save_anyway(self) -> None:
        self._validation_index = None
        self.calibration_finished.emit(True, "校准已保存（已跳过未命中区域）", ())

    def start_streaming(self) -> None:
        if self.worker is None or self.runtime is None or self.runtime.metadata is None:
            self.stream_state_changed.emit(False, "请先完成校准")
            return
        self._streaming = True
        self.stream_state_changed.emit(True, "眼动数据正在输出")

    def stop_streaming(self) -> None:
        self._streaming = False
        self.stream_state_changed.emit(False, "眼动数据输出已停止")

    def stop_camera(self) -> None:
        if self.worker is not None and self.thread is not None:
            if self.thread.isRunning():
                LOGGER.info("camera_worker_stop_requested")
                QMetaObject.invokeMethod(
                    self.worker,
                    "stop",
                    Qt.ConnectionType.BlockingQueuedConnection,
                )
            self.thread.quit()
            if not self.thread.wait(5000):
                LOGGER.warning("camera_thread_stop_wait_extended")
                self.thread.wait()
            LOGGER.info("camera_thread_stopped")
        if self.runtime is not None:
            self.runtime.close()
            LOGGER.info("camera_runtime_closed")
        self.worker = None
        self.thread = None
        self.runtime = None
        self._camera_ready = False

    def stop(self) -> None:
        self._heartbeat.stop()
        self.stop_streaming()
        self.stop_camera()
        self.publisher.close()

    def _on_camera_state(self, ready: bool, message: str) -> None:
        self._camera_ready = ready
        self.camera_state_changed.emit(ready, message)

    def _on_worker_failure(self, message: str) -> None:
        self.camera_state_changed.emit(False, message)

    def _on_packet(self, packet: CapturePacket) -> None:
        if self._session is not None and not self._session.complete:
            before = self._session.current_point_id
            self._session.add_frame(
                packet.timestamp,
                packet.observation.features,
                blink=packet.observation.blink,
                face_detected=packet.observation.face_detected,
            )
            after = self._session.current_point_id
            if self._session.blocked_reason:
                self.calibration_finished.emit(False, "当前位置有效画面不足，请调整坐姿后重试", ())
            elif after != before and after is not None:
                self.calibration_point_changed.emit(after)
            if self._session.complete and not self._training_sent:
                self._train_completed_session()
        elif self._validation_index is not None:
            self._process_validation(packet)
        if self._streaming:
            self._publish_estimate(packet)

    def _train_completed_session(self) -> None:
        assert self._session is not None and self.environment is not None
        features, labels = self._session.training_data()
        if self._base_training is not None:
            features = np.vstack((self._base_training[0], features))
            labels = np.vstack((self._base_training[1], labels))
        self._base_training = (features.copy(), labels.copy())
        lower = tuple(map(float, np.percentile(features, 2, axis=0)))
        upper = tuple(map(float, np.percentile(features, 98, axis=0)))
        metadata = CalibrationMetadata(
            f"cal-{uuid.uuid4().hex[:12]}",
            datetime.now(timezone.utc).isoformat(),
            self.environment,
            lower,
            upper,
        )
        self._training_sent = True
        self._session = None
        self._train_requested.emit(features, labels, metadata)

    def _on_model_saved(self, _stored) -> None:
        if self._validate_after_training:
            self._validation_index = 0
            self._validation_started_at = None
            self._validation_hits = {f"target_{index}": 0 for index in range(6)}
            self.calibration_point_changed.emit("target_0")
        else:
            self.calibration_finished.emit(True, "快速校准已保存", ())

    def _process_validation(self, packet: CapturePacket) -> None:
        assert self.environment is not None and self._validation_index is not None
        now = packet.timestamp
        if self._validation_started_at is None:
            self._validation_started_at = now
        target_id = f"target_{self._validation_index}"
        estimate = packet.estimate
        if estimate.valid and estimate.screen_x is not None and estimate.screen_y is not None:
            layout = build_layout(self.environment.screen_width, self.environment.screen_height)
            if hit_test(layout, estimate.screen_x, estimate.screen_y, include_back=False) == target_id:
                self._validation_hits[target_id] += 1
        if now - self._validation_started_at < 0.8:
            return
        self._validation_index += 1
        self._validation_started_at = None
        if self._validation_index < 6:
            self.calibration_point_changed.emit(f"target_{self._validation_index}")
            return
        result = score_validation(self._validation_hits)
        self._validation_index = None
        self._failed_targets = result.failed_target_ids
        message = f"命中 {result.hit_count}/{result.total_count}"
        self.calibration_finished.emit(result.passed, message, result.failed_target_ids)

    def _publish_estimate(self, packet: CapturePacket) -> None:
        assert self.runtime is not None and self.runtime.metadata is not None
        estimate = packet.estimate
        metadata = self.runtime.metadata
        sample = GazeSample(
            timestamp=time.time(),
            valid=estimate.valid,
            face_detected=estimate.face_detected,
            blink=estimate.blink,
            quality=estimate.quality,
            fps=packet.fps,
            calibration_id=metadata.calibration_id,
            layout_version=metadata.environment.layout_version,
            screen_x=estimate.screen_x if estimate.valid else None,
            screen_y=estimate.screen_y if estimate.valid else None,
            raw_x=estimate.raw_x if estimate.valid else None,
            raw_y=estimate.raw_y if estimate.valid else None,
        )
        self.publisher.send(sample)

    def _publish_heartbeat(self) -> None:
        metadata = self.runtime.metadata if self.runtime is not None else None
        heartbeat = Heartbeat(
            timestamp=time.time(),
            camera_ok=self._camera_ready,
            calibration_ready=metadata is not None,
            calibration_id="" if metadata is None else metadata.calibration_id,
            layout_version="gaze-grid-v1" if metadata is None else metadata.environment.layout_version,
            fps=0.0,
            error=None if self._camera_ready else "camera_not_ready",
        )
        self.publisher.send(heartbeat)


class CaptureWindow(QMainWindow):
    def __init__(self, controller: CaptureController, paths: AppPaths) -> None:
        super().__init__()
        self.controller = controller
        self.paths = paths
        self._calibration_mode = False
        self._highlight_id: str | None = None
        self.setWindowTitle("眼动采集校准")
        self.setMinimumSize(920, 640)
        self.setStyleSheet(
            "QMainWindow{background:#f4f6f8;color:#15202b;}"
            "QPushButton{min-height:38px;padding:0 16px;border:1px solid #9aa7b2;background:#ffffff;}"
            "QPushButton:disabled{color:#8b949e;background:#edf0f2;}"
            "QLabel{font-size:14px;}"
        )
        self._controls = QWidget(self)
        self.setCentralWidget(self._controls)
        root = QVBoxLayout(self._controls)
        title = QLabel("眼动采集与快速校准")
        title.setStyleSheet("font-size:24px;font-weight:600;")
        root.addWidget(title)

        connection = QHBoxLayout()
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"摄像头 {index}" for index in range(5)])
        self.connect_button = QPushButton("连接摄像头")
        self.camera_status_label = QLabel("摄像头未连接")
        connection.addWidget(self.camera_combo)
        connection.addWidget(self.connect_button)
        connection.addWidget(self.camera_status_label, 1)
        root.addLayout(connection)

        self.preview_label = QLabel("等待摄像头画面")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(360)
        self.preview_label.setStyleSheet("background:#20252a;color:#dbe2e8;border:1px solid #65727d;")
        root.addWidget(self.preview_label, 1)

        options = QHBoxLayout()
        self.validation_checkbox = QCheckBox("校准后进行精度验证")
        self.calibrate_button = QPushButton("快速校准")
        self.calibrate_button.setEnabled(False)
        self.stream_button = QPushButton("开始输出")
        self.stop_stream_button = QPushButton("停止输出")
        self.retry_failed_button = QPushButton("重校异常区域")
        self.save_anyway_button = QPushButton("仍然保存")
        self.retry_failed_button.hide()
        self.save_anyway_button.hide()
        for widget in (
            self.validation_checkbox,
            self.calibrate_button,
            self.stream_button,
            self.stop_stream_button,
            self.retry_failed_button,
            self.save_anyway_button,
        ):
            options.addWidget(widget)
        root.addLayout(options)
        self.result_label = QLabel("请选择摄像头并完成快速校准")
        root.addWidget(self.result_label)

        self.connect_button.clicked.connect(
            lambda: self.controller.start_camera(self.camera_combo.currentIndex())
        )
        self.calibrate_button.clicked.connect(self._begin_calibration)
        self.stream_button.clicked.connect(self.controller.start_streaming)
        self.stop_stream_button.clicked.connect(self.controller.stop_streaming)
        self.retry_failed_button.clicked.connect(self.controller.retry_failed_regions)
        self.save_anyway_button.clicked.connect(self.controller.save_anyway)
        controller.camera_state_changed.connect(self._on_camera_state)
        controller.calibration_point_changed.connect(self.show_calibration_point)
        controller.calibration_finished.connect(self._on_calibration_finished)
        controller.stream_state_changed.connect(self._on_stream_state)
        controller.preview_ready.connect(self._on_preview)

    def _begin_calibration(self) -> None:
        self._calibration_mode = True
        self._controls.hide()
        self.showFullScreen()
        self.controller.start_calibration(self.validation_checkbox.isChecked())

    def show_calibration_point(self, point_id: str) -> None:
        self._calibration_mode = True
        self._highlight_id = point_id
        self.update()

    def highlight_center(self) -> tuple[float, float] | None:
        if self._highlight_id is None:
            return None
        layout = build_layout(max(1, self.width()), max(1, self.height()))
        point_map = dict(calibration_points(layout))
        return point_map.get(self._highlight_id)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._calibration_mode:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f8fa"))
        layout = build_layout(self.width(), self.height())
        painter.setPen(QPen(QColor("#a6b0b8"), 2))
        for rect in layout.targets:
            painter.drawRect(round(rect.left), round(rect.top), round(rect.width), round(rect.height))
        back = layout.back_target
        painter.drawRect(round(back.left), round(back.top), round(back.width), round(back.height))
        center = self.highlight_center()
        if center is not None:
            painter.setBrush(QColor("#2f7d62"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(round(center[0] - 18), round(center[1] - 18), 36, 36)

    def _on_camera_state(self, ready: bool, message: str) -> None:
        self.camera_status_label.setText(message)
        self.calibrate_button.setEnabled(ready)

    def _on_calibration_finished(self, passed: bool, message: str, failed_targets) -> None:
        self.result_label.setText(message)
        self._calibration_mode = False
        self._controls.show()
        if self.isFullScreen():
            self.showNormal()
        show_choices = not passed and bool(failed_targets)
        self.retry_failed_button.setVisible(show_choices)
        self.save_anyway_button.setVisible(show_choices)
        self.update()

    def _on_stream_state(self, running: bool, message: str) -> None:
        self.result_label.setText(message)
        self.stream_button.setEnabled(not running)
        self.stop_stream_button.setEnabled(running)

    def _on_preview(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        self.controller.stop()
        event.accept()
