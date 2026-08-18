from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
import threading
import time
import uuid

import numpy as np
from PyQt6.QtCore import QMetaObject, QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCloseEvent, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .calibration import (
    CalibrationEnvironment,
    CalibrationMode,
    CalibrationMetadata,
    CalibrationPoint,
    CalibrationProfile,
    CalibrationSession,
    CalibrationStore,
    REFERENCE_PIPELINE_VERSION,
    apply_screen_affine,
    calibration_profile,
    fit_screen_affine,
    score_validation,
    stable_median_prediction,
)
from .capture_worker import CameraWorker, CapturePacket
from .eyetrax_runtime import EyeTraxRuntime
from .layout import (
    LAYOUT_VERSION,
    build_layout,
    calibration_points,
    hit_test,
    reentry_calibration_points,
    uniform_grid_calibration_points,
    validation_points,
)
from .paths import AppPaths
from .presence import PresenceEvent, SeatReturnDetector
from .protocol import GazeSample, Heartbeat, UdpPublisher


LOGGER = logging.getLogger("pure_gaze_typing.capture")


class CaptureController(QObject):
    MIN_PREDICTION_SAMPLES = 8
    MAX_PREDICTION_CAPTURE_SECONDS = 1.60

    camera_state_changed = pyqtSignal(bool, str)
    calibration_point_changed = pyqtSignal(str)
    calibration_finished = pyqtSignal(bool, str, object)
    stream_state_changed = pyqtSignal(bool, str)
    preview_ready = pyqtSignal(object)
    shutdown_finished = pyqtSignal()
    _train_requested = pyqtSignal(object, object, object)
    _configure_requested = pyqtSignal(object)
    _save_requested = pyqtSignal(object)
    _load_requested = pyqtSignal(object)

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
        self._profile: CalibrationProfile = calibration_profile(CalibrationMode.FAST)
        self._calibration_active = False
        self._retrying = False
        self._validate_after_training = False
        self._training_sent = False
        self._base_training: tuple[np.ndarray, np.ndarray] | None = None
        self._provisional_metadata: CalibrationMetadata | None = None
        self._prediction_stage: str | None = None
        self._prediction_target_ids: tuple[str, ...] = ()
        self._prediction_index: int | None = None
        self._prediction_started_at: float | None = None
        self._prediction_samples: list[tuple[float, float]] = []
        self._prediction_frame_counts: dict[str, int] = {}
        self._bias_predictions: dict[str, tuple[float, float]] = {}
        self._after_configure: str | None = None
        self._validation_index: int | None = None
        self._validation_started_at: float | None = None
        self._validation_hits: dict[str, int] = {}
        self._validation_errors: dict[str, float] = {}
        self._failed_targets: tuple[str, ...] = ()
        self._grid_rows = 3
        self._grid_columns = 3
        self._presence = SeatReturnDetector(absence_seconds=1.5)
        self._reentry_pending = False
        self._reentry_predictions: dict[str, tuple[float, float]] = {}
        self._saving_reason = "full"
        self._pending_camera_index: int | None = None
        self._camera_stop_requested = False
        self._shutdown_requested = False
        self._shutdown_emitted = False
        self._publisher_closed = False
        self._packet_lock = threading.Lock()
        self._pending_packet: CapturePacket | None = None
        self._packet_timer = QTimer(self)
        self._packet_timer.setInterval(25)
        self._packet_timer.timeout.connect(self._drain_latest_packet)
        self._packet_timer.start()
        self._stop_watchdog = QTimer(self)
        self._stop_watchdog.setSingleShot(True)
        self._stop_watchdog.setInterval(3000)
        self._stop_watchdog.timeout.connect(self._force_camera_thread_stop)
        self._heartbeat = QTimer(self)
        self._heartbeat.setInterval(1000)
        self._heartbeat.timeout.connect(self._publish_heartbeat)
        self._heartbeat.start()

    def start_camera(self, camera_index: int) -> None:
        camera_index = int(camera_index)
        LOGGER.info("camera_start_requested index=%s", camera_index)
        if self._shutdown_requested:
            LOGGER.warning("camera_start_rejected shutdown_in_progress=True")
            return
        if self.thread is not None:
            self._pending_camera_index = camera_index
            self.camera_state_changed.emit(False, "正在重新连接摄像头…")
            self._request_camera_stop()
            return
        self._launch_camera(camera_index)

    def _launch_camera(self, camera_index: int) -> None:
        self.environment = self.environment_factory(int(camera_index))
        LOGGER.info("camera_environment_ready environment=%s", self.environment)
        stored = self.store.load(self.environment)
        self.runtime = None
        self.thread = QThread(self)
        self.worker = CameraWorker(
            camera_index,
            None,
            self.store,
            runtime_factory=self.runtime_factory,
            runtime_args=(
                self.face_model_path,
                self.environment.screen_width,
                self.environment.screen_height,
            ),
            stored_calibration=stored,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.runtime_ready.connect(self._on_runtime_ready)
        self.worker.camera_opened.connect(self._on_camera_state)
        self.worker.preview_ready.connect(self.preview_ready)
        self.worker.packet_ready.connect(
            self._queue_latest_packet,
            Qt.ConnectionType.DirectConnection,
        )
        self.worker.model_trained.connect(self._on_model_trained)
        self.worker.model_configured.connect(self._on_model_configured)
        self.worker.model_saved.connect(self._on_model_saved)
        self.worker.failed.connect(self._on_worker_failure)
        self.worker.stopped.connect(self.worker.deleteLater)
        self.worker.stopped.connect(
            self.thread.quit,
            Qt.ConnectionType.DirectConnection,
        )
        self.thread.finished.connect(self._on_camera_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self._train_requested.connect(self.worker.train_model)
        self._configure_requested.connect(self.worker.configure_metadata)
        self._save_requested.connect(self.worker.save_current)
        self._load_requested.connect(self.worker.load_calibration)
        self.thread.start()
        LOGGER.info("camera_thread_started")

    def start_calibration(
        self,
        mode: CalibrationMode | str | bool = CalibrationMode.FAST,
        validate: bool = False,
        grid_rows: int = 3,
        grid_columns: int = 3,
    ) -> None:
        if isinstance(mode, bool):
            validate = mode
            mode = CalibrationMode.FAST
        if not self._camera_ready or self.environment is None:
            self.calibration_finished.emit(False, "请先连接摄像头", ())
            return
        self._profile = calibration_profile(mode)
        self._grid_rows = int(grid_rows)
        self._grid_columns = int(grid_columns)
        if not 2 <= self._grid_rows <= 5 or not 2 <= self._grid_columns <= 5:
            self.calibration_finished.emit(False, "校准网格行列数必须在 2 到 5 之间", ())
            return
        self._validate_after_training = bool(validate)
        self._saving_reason = "full"
        self._calibration_active = True
        self._retrying = False
        self._streaming = False
        self._base_training = None
        self._provisional_metadata = None
        self._validation_hits = {f"validation_{index}": 0 for index in range(6)}
        self._validation_errors = {}
        self._failed_targets = ()
        layout = build_layout(self.environment.screen_width, self.environment.screen_height)
        duration = self._profile.settle_seconds + self._profile.capture_seconds
        center = (layout.screen_width / 2.0, layout.screen_height / 2.0)
        point_specs: list[tuple[str, tuple[float, float], float]] = [
            ("pose_center", center, self._profile.pose_hold_seconds),
        ]
        point_specs.extend(
            (name, point, duration)
            for name, point in uniform_grid_calibration_points(
                layout,
                rows=self._grid_rows,
                columns=self._grid_columns,
            )
        )
        target_points = list(calibration_points(layout))
        for pass_index in range(self._profile.target_passes):
            ordered = target_points
            if pass_index == 1 and self._profile.reverse_second_pass:
                ordered = [target_points[0], *reversed(target_points[1:])]
            point_specs.extend((name, point, duration) for name, point in ordered)
        points = tuple(
            CalibrationPoint(name, point[0], point[1], point_duration)
            for name, point, point_duration in point_specs
        )
        self._start_training_session(points)

    def _start_training_session(self, points: tuple[CalibrationPoint, ...]) -> None:
        self._session = CalibrationSession(
            points,
            settle_seconds=self._profile.settle_seconds,
            min_valid_frames=8,
            max_point_seconds=max(2.0, self._profile.settle_seconds + self._profile.capture_seconds * 2.5),
        )
        self._training_sent = False
        self._prediction_stage = None
        self._prediction_index = None
        self._validation_index = None
        if self._session.current_point_id:
            self.calibration_point_changed.emit(self._session.current_point_id)

    def retry_failed_regions(self) -> None:
        if not self._failed_targets or self.environment is None:
            return
        self._calibration_active = True
        self._retrying = True
        layout = build_layout(self.environment.screen_width, self.environment.screen_height)
        point_map = dict((*calibration_points(layout), *validation_points(layout)))
        duration = self._profile.settle_seconds + self._profile.capture_seconds
        points = tuple(
            CalibrationPoint(target_id, *point_map[target_id], duration)
            for target_id in self._failed_targets
        )
        self._start_training_session(points)

    def save_anyway(self) -> None:
        self.calibration_finished.emit(False, "精度验证开启时必须达到至少 5/6", self._failed_targets)

    def cancel_calibration(self) -> None:
        if not self._calibration_active:
            return
        self._calibration_active = False
        self._session = None
        self._prediction_stage = None
        self._prediction_index = None
        self._after_configure = None
        self._retrying = False
        self._provisional_metadata = None
        self._reentry_pending = False
        self._presence.reset()
        if self.environment is not None:
            stored = self.store.load(self.environment)
            if stored is not None and self.worker is not None:
                self._load_requested.emit(stored)
            elif self.runtime is not None:
                self.runtime.metadata = None
        self.calibration_finished.emit(False, "校准已取消", ())

    def start_streaming(self) -> None:
        LOGGER.info("stream_start_requested")
        if self.worker is None or self.runtime is None or self.runtime.metadata is None:
            LOGGER.warning("stream_start_rejected calibration_ready=%s", bool(self.runtime and self.runtime.metadata))
            self.stream_state_changed.emit(False, "请先完成校准")
            return
        self._streaming = True
        LOGGER.info("stream_started calibration_id=%s", self.runtime.metadata.calibration_id)
        self.stream_state_changed.emit(True, "眼动数据正在输出")

    def stop_streaming(self) -> None:
        self._streaming = False
        LOGGER.info("stream_stopped")
        self.stream_state_changed.emit(False, "眼动输出已暂停（摄像头预览继续）")

    def _queue_latest_packet(self, packet: CapturePacket) -> None:
        with self._packet_lock:
            self._pending_packet = packet

    def _drain_latest_packet(self) -> None:
        with self._packet_lock:
            packet = self._pending_packet
            self._pending_packet = None
        if packet is not None:
            self._on_packet(packet)

    def stop_camera(self) -> None:
        self._pending_camera_index = None
        self._request_camera_stop()

    def _request_camera_stop(self) -> None:
        if self.worker is None or self.thread is None:
            if self._shutdown_requested:
                self._finish_shutdown()
            return
        if self._camera_stop_requested:
            return
        self._camera_stop_requested = True
        self._camera_ready = False
        self._streaming = False
        self._reentry_pending = False
        self._presence.reset()
        with self._packet_lock:
            self._pending_packet = None
        LOGGER.info("camera_worker_stop_requested async=True")
        try:
            QMetaObject.invokeMethod(
                self.worker,
                "stop",
                Qt.ConnectionType.QueuedConnection,
            )
            self.worker.request_stop()
        except RuntimeError:
            LOGGER.info("camera_worker_already_deleted")
        self.thread.requestInterruption()
        self._stop_watchdog.start()

    def _force_camera_thread_stop(self) -> None:
        thread = self.thread
        if thread is None or not thread.isRunning():
            return
        LOGGER.error("camera_thread_force_terminate async=True")
        if self.worker is not None:
            self.worker.request_stop()
        thread.requestInterruption()
        thread.quit()
        thread.terminate()

    def _on_camera_thread_finished(self) -> None:
        self._stop_watchdog.stop()
        LOGGER.info("camera_thread_stopped")
        self.worker = None
        self.thread = None
        self.runtime = None
        self._camera_stop_requested = False
        with self._packet_lock:
            self._pending_packet = None
        if self._shutdown_requested:
            self._finish_shutdown()
            return
        pending_camera_index = self._pending_camera_index
        self._pending_camera_index = None
        if pending_camera_index is not None:
            QTimer.singleShot(0, lambda: self.start_camera(pending_camera_index))

    def stop(self) -> None:
        if self._shutdown_requested:
            return
        LOGGER.info("capture_shutdown_requested")
        self._shutdown_requested = True
        self._pending_camera_index = None
        self._heartbeat.stop()
        self._packet_timer.stop()
        self.stop_streaming()
        self._request_camera_stop()
        if self.thread is None:
            self._finish_shutdown()

    def _finish_shutdown(self) -> None:
        if self._shutdown_emitted:
            return
        if not self._publisher_closed:
            self.publisher.close()
            self._publisher_closed = True
        self._shutdown_emitted = True
        LOGGER.info("capture_shutdown_finished")
        self.shutdown_finished.emit()

    def _on_camera_state(self, ready: bool, message: str) -> None:
        self._camera_ready = ready
        LOGGER.info("camera_state_changed ready=%s message=%s", ready, message)
        self.camera_state_changed.emit(ready, message)
        if ready and self.runtime is not None and self.runtime.metadata is not None:
            LOGGER.info("stream_auto_start source=loaded_calibration")
            self.start_streaming()

    def _on_runtime_ready(self, runtime: EyeTraxRuntime) -> None:
        self.runtime = runtime
        LOGGER.info("camera_runtime_ready runtime=%s", type(runtime).__name__)
        if runtime.metadata is not None:
            LOGGER.info("camera_calibration_loaded id=%s", runtime.metadata.calibration_id)

    def _on_worker_failure(self, message: str) -> None:
        self._camera_ready = False
        self._streaming = False
        if self._calibration_active:
            self._calibration_active = False
            self._session = None
            self._prediction_stage = None
            self.calibration_finished.emit(False, message, ())
        self.stream_state_changed.emit(False, "眼动输出已停止")
        self.camera_state_changed.emit(False, message)

    def _on_packet(self, packet: CapturePacket) -> None:
        self._process_presence(packet)
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
        elif self._prediction_stage is not None:
            self._process_prediction_stage(packet)
        if self._streaming:
            self._publish_estimate(packet)

    def _process_presence(self, packet: CapturePacket) -> None:
        if self._calibration_active:
            return
        event = self._presence.update(
            packet.timestamp,
            face_detected=packet.observation.face_detected,
            active=self._streaming or self._reentry_pending,
        )
        if event is PresenceEvent.LEFT:
            self._reentry_pending = True
            self._streaming = False
            LOGGER.info("seat_left_reentry_required")
            self.stream_state_changed.emit(False, "检测到被试离座，已暂停输出")
        elif event is PresenceEvent.RETURNED:
            LOGGER.info("seat_returned_reentry_start")
            self._start_reentry_calibration()

    def _start_reentry_calibration(self) -> None:
        if self.environment is None or self.runtime is None or self.runtime.metadata is None:
            self._reentry_pending = False
            self.calibration_finished.emit(False, "无可用校准，请重新完成初始校准", ())
            return
        self._profile = calibration_profile(CalibrationMode.FAST)
        self._calibration_active = True
        self._streaming = False
        self._provisional_metadata = self.runtime.metadata
        self._reentry_predictions = {}
        layout = build_layout(self.environment.screen_width, self.environment.screen_height)
        target_ids = tuple(name for name, _point in reentry_calibration_points(layout))
        self._start_prediction_stage("reentry", target_ids)

    def _train_completed_session(self) -> None:
        assert self._session is not None and self.environment is not None
        features, labels = self._session.prepared_training_data(
            max_samples_per_point=self._profile.max_samples_per_point,
            min_samples=8,
        )
        if self._base_training is not None:
            features = np.vstack((self._base_training[0], features))
            labels = np.vstack((self._base_training[1], labels))
        self._base_training = (features.copy(), labels.copy())
        lower = tuple(map(float, np.percentile(features, 2, axis=0)))
        upper = tuple(map(float, np.percentile(features, 98, axis=0)))
        previous = self._provisional_metadata
        metadata = CalibrationMetadata(
            f"cal-{uuid.uuid4().hex[:12]}" if previous is None else previous.calibration_id,
            datetime.now(timezone.utc).isoformat() if previous is None else previous.created_at,
            self.environment,
            lower,
            upper,
            calibration_mode=self._profile.mode.value,
            pipeline_version=REFERENCE_PIPELINE_VERSION,
            screen_affine=() if previous is None else previous.screen_affine,
        )
        self._training_sent = True
        self._session = None
        self._train_requested.emit(features, labels, metadata)

    def _on_model_trained(self, metadata: CalibrationMetadata) -> None:
        if not self._calibration_active:
            return
        self._provisional_metadata = metadata
        if self.environment is None:
            return
        if self._retrying:
            target_ids = self._failed_targets
        else:
            target_ids = tuple(name for name, _point in calibration_points(
                build_layout(self.environment.screen_width, self.environment.screen_height)
            ))
        self._bias_predictions = {}
        self._start_prediction_stage("bias", target_ids)

    def _start_prediction_stage(self, stage: str, target_ids: tuple[str, ...]) -> None:
        if not target_ids:
            raise ValueError("prediction stage requires at least one target")
        self._prediction_stage = stage
        self._prediction_target_ids = tuple(target_ids)
        self._prediction_index = 0
        self._prediction_started_at = None
        self._prediction_samples = []
        self._prediction_frame_counts = {
            "total": 0,
            "accepted": 0,
            "no_face": 0,
            "blink": 0,
            "no_raw_prediction": 0,
        }
        self.calibration_point_changed.emit(self._prediction_target_ids[0])

    def _process_prediction_stage(self, packet: CapturePacket) -> None:
        assert self._prediction_stage is not None and self._prediction_index is not None
        now = packet.timestamp
        if self._prediction_started_at is None:
            self._prediction_started_at = now
        elapsed = now - self._prediction_started_at
        settle = self._profile.validation_settle_seconds
        capture = self._profile.validation_capture_seconds
        estimate = packet.estimate
        if elapsed >= settle:
            self._prediction_frame_counts["total"] += 1
            if not packet.observation.face_detected:
                self._prediction_frame_counts["no_face"] += 1
            elif packet.observation.blink:
                self._prediction_frame_counts["blink"] += 1
            elif (
                estimate.raw_x is None
                or estimate.raw_y is None
                or not np.isfinite(estimate.raw_x)
                or not np.isfinite(estimate.raw_y)
            ):
                self._prediction_frame_counts["no_raw_prediction"] += 1
            else:
                self._prediction_samples.append((estimate.raw_x, estimate.raw_y))
                self._prediction_frame_counts["accepted"] += 1
        if elapsed < settle + capture:
            return
        if (
            len(self._prediction_samples) < self.MIN_PREDICTION_SAMPLES
            and elapsed < settle + self.MAX_PREDICTION_CAPTURE_SECONDS
        ):
            return
        if len(self._prediction_samples) < self.MIN_PREDICTION_SAMPLES:
            target_id = self._prediction_target_ids[self._prediction_index]
            if self._prediction_stage == "reentry":
                self._prediction_started_at = None
                self._prediction_samples = []
                self.calibration_point_changed.emit(target_id)
                return
            LOGGER.warning(
                "prediction_samples_insufficient target=%s samples=%s fps=%.1f counts=%s",
                target_id,
                len(self._prediction_samples),
                packet.fps,
                self._prediction_frame_counts,
            )
            self._prediction_stage = None
            self._calibration_active = False
            self.calibration_finished.emit(False, f"{target_id} 有效注视样本不足，请重试", (target_id,))
            return

        target_id = self._prediction_target_ids[self._prediction_index]
        median_prediction = stable_median_prediction(
            self._prediction_samples,
            min_samples=self.MIN_PREDICTION_SAMPLES,
            max_samples=20,
        )
        if self._prediction_stage == "bias":
            self._bias_predictions[target_id] = median_prediction
        elif self._prediction_stage == "reentry":
            self._reentry_predictions[target_id] = median_prediction
        else:
            self._record_validation_result(target_id, median_prediction)

        self._prediction_index += 1
        self._prediction_started_at = None
        self._prediction_samples = []
        self._prediction_frame_counts = {
            "total": 0,
            "accepted": 0,
            "no_face": 0,
            "blink": 0,
            "no_raw_prediction": 0,
        }
        if self._prediction_index < len(self._prediction_target_ids):
            self.calibration_point_changed.emit(self._prediction_target_ids[self._prediction_index])
            return

        completed_stage = self._prediction_stage
        self._prediction_stage = None
        self._prediction_index = None
        if completed_stage == "bias":
            self._finish_bias_correction()
        elif completed_stage == "reentry":
            self._finish_reentry_correction()
        else:
            self._finish_validation()

    def _point_map(self) -> dict[str, tuple[float, float]]:
        assert self.environment is not None
        layout = build_layout(self.environment.screen_width, self.environment.screen_height)
        return dict((
            *calibration_points(layout),
            *validation_points(layout),
            *reentry_calibration_points(layout),
        ))

    def _finish_reentry_correction(self) -> None:
        assert self._provisional_metadata is not None
        point_map = self._point_map()
        names = tuple(self._reentry_predictions)
        predictions = np.asarray([self._reentry_predictions[name] for name in names], dtype=float)
        targets = np.asarray([point_map[name] for name in names], dtype=float)
        try:
            matrix = fit_screen_affine(predictions, targets)
        except (ValueError, RuntimeError):
            LOGGER.exception("reentry_calibration_fit_failed")
            self._reentry_predictions = {}
            target_ids = tuple(name for name, _point in reentry_calibration_points(
                build_layout(self.environment.screen_width, self.environment.screen_height)
            ))
            self._start_prediction_stage("reentry", target_ids)
            return
        coefficients = tuple(tuple(map(float, row)) for row in matrix)
        self._provisional_metadata = replace(
            self._provisional_metadata,
            calibration_id=f"cal-reentry-{uuid.uuid4().hex[:10]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            screen_affine=coefficients,
            validation_hits=None,
            validation_total=None,
            validation_median_error_px=None,
            validation_max_error_px=None,
        )
        self._after_configure = "reentry_save"
        self._saving_reason = "reentry"
        self._configure_requested.emit(self._provisional_metadata)

    def _finish_bias_correction(self) -> None:
        assert self._provisional_metadata is not None
        point_map = self._point_map()
        names = tuple(self._bias_predictions)
        predictions = np.asarray([self._bias_predictions[name] for name in names], dtype=float)
        targets = np.asarray([point_map[name] for name in names], dtype=float)
        if self._retrying and self._provisional_metadata.screen_affine:
            matrix = np.asarray(self._provisional_metadata.screen_affine, dtype=float).copy()
            corrected = np.column_stack((predictions, np.ones(len(predictions)))) @ matrix
            matrix[2] += np.median(targets - corrected, axis=0)
        else:
            matrix = fit_screen_affine(predictions, targets)
        coefficients = tuple(tuple(map(float, row)) for row in matrix)
        self._provisional_metadata = replace(
            self._provisional_metadata,
            screen_affine=coefficients,
        )
        self._after_configure = "validate" if self._validate_after_training else "save"
        self._configure_requested.emit(self._provisional_metadata)

    def _on_model_configured(self, metadata: CalibrationMetadata) -> None:
        if not self._calibration_active:
            return
        self._provisional_metadata = metadata
        action = self._after_configure
        self._after_configure = None
        if action == "validate":
            target_ids = self._failed_targets if self._retrying else tuple(
                f"validation_{index}" for index in range(6)
            )
            self._start_prediction_stage("validation", target_ids)
        elif action == "save":
            self._save_requested.emit(metadata)
        elif action == "reentry_save":
            self._save_requested.emit(metadata)

    def _record_validation_result(
        self,
        target_id: str,
        raw_prediction: tuple[float, float],
    ) -> None:
        assert self.environment is not None and self._provisional_metadata is not None
        corrected = apply_screen_affine(
            self._provisional_metadata.screen_affine,
            raw_prediction[0],
            raw_prediction[1],
        )
        layout = build_layout(self.environment.screen_width, self.environment.screen_height)
        expected = dict(validation_points(layout))[target_id]
        self._validation_errors[target_id] = float(np.hypot(
            corrected[0] - expected[0],
            corrected[1] - expected[1],
        ))
        self._validation_hits[target_id] = int(
            hit_test(layout, corrected[0], corrected[1], target_count=6)
            == target_id.replace("validation_", "target_")
        )

    def _finish_validation(self) -> None:
        result = score_validation(self._validation_hits, min_hits_per_target=1)
        self._failed_targets = result.failed_target_ids
        errors = tuple(self._validation_errors.values())
        median_error = None if not errors else float(np.median(errors))
        max_error = None if not errors else float(np.max(errors))
        detail = "" if median_error is None else (
            f"，中位误差 {median_error:.0f}px，最大误差 {max_error:.0f}px"
        )
        message = f"命中 {result.hit_count}/{result.total_count}{detail}"
        if result.passed:
            assert self._provisional_metadata is not None
            self._provisional_metadata = replace(
                self._provisional_metadata,
                validation_hits=result.hit_count,
                validation_total=result.total_count,
                validation_median_error_px=median_error,
                validation_max_error_px=max_error,
            )
            self._save_requested.emit(self._provisional_metadata)
            return
        self._calibration_active = False
        self.calibration_finished.emit(False, message, result.failed_target_ids)

    def _on_model_saved(self, _stored) -> None:
        if not self._calibration_active:
            return
        self._calibration_active = False
        self._retrying = False
        if self._saving_reason == "reentry":
            message = "五点快速校正已保存，眼动输出已恢复"
            self._reentry_pending = False
            self._presence.reset()
            self._saving_reason = "full"
            self.calibration_finished.emit(True, message, ())
            LOGGER.info("stream_auto_start source=reentry_calibration")
            self.start_streaming()
            return
        mode_name = "精确校准" if self._profile.mode is CalibrationMode.PRECISE else "快速校准"
        if self._validate_after_training and self._provisional_metadata is not None:
            message = (
                f"{mode_name}已保存，命中 "
                f"{self._provisional_metadata.validation_hits}/{self._provisional_metadata.validation_total}"
            )
        else:
            message = f"{mode_name}已保存"
        self.calibration_finished.emit(True, message, ())
        LOGGER.info("stream_auto_start source=saved_calibration")
        self.start_streaming()

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
        metadata = (
            self.runtime.metadata
            if self.runtime is not None and not self._calibration_active
            else None
        )
        heartbeat = Heartbeat(
            timestamp=time.time(),
            camera_ok=self._camera_ready,
            calibration_ready=metadata is not None,
            calibration_id="" if metadata is None else metadata.calibration_id,
            layout_version=LAYOUT_VERSION if metadata is None else metadata.environment.layout_version,
            fps=0.0,
            error=None if self._camera_ready else "camera_not_ready",
            streaming=self._streaming,
        )
        self.publisher.send(heartbeat)


class CalibrationModeDialog(QDialog):
    mode_selected = pyqtSignal(object, int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("选择校准方式")
        self.setModal(False)
        self.setStyleSheet(
            "QDialog{background:#f4f6f5;color:#17221e;}"
            "QLabel{color:#17221e;}"
            "QPushButton{min-height:76px;min-width:280px;padding:12px 24px;"
            "border:1px solid #91a29a;border-radius:4px;background:#ffffff;"
            "font-size:18px;font-weight:600;text-align:left;}"
            "QPushButton:hover{border-color:#2f7d62;background:#edf5f1;}"
            "QPushButton#cancelButton{min-height:42px;min-width:120px;font-size:15px;"
            "font-weight:400;text-align:center;}"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(64, 56, 64, 48)
        root.setSpacing(22)
        root.addStretch(2)
        title = QLabel("选择校准方式")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:30px;font-weight:600;")
        root.addWidget(title)
        subtitle = QLabel("保持自然坐姿，按照屏幕上的绿色圆点注视")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("font-size:17px;color:#516159;")
        root.addWidget(subtitle)

        choices = QHBoxLayout()
        choices.setSpacing(24)
        choices.addStretch()
        self.fast_button = QPushButton("快速校准（推荐）\n约 25–35 秒")
        self.precise_button = QPushButton("精确校准\n约 50–60 秒")
        choices.addWidget(self.fast_button)
        choices.addWidget(self.precise_button)
        choices.addStretch()
        root.addLayout(choices)

        grid_row = QHBoxLayout()
        grid_row.addStretch()
        grid_label = QLabel("精确校准初始网格")
        grid_label.setStyleSheet("font-size:16px;font-weight:600;")
        self.grid_rows_spin = QSpinBox()
        self.grid_columns_spin = QSpinBox()
        for spin in (self.grid_rows_spin, self.grid_columns_spin):
            spin.setRange(2, 5)
            spin.setValue(3)
            spin.setFixedWidth(78)
            spin.setStyleSheet(
                "QSpinBox{min-height:36px;padding:0 8px;background:#ffffff;"
                "border:1px solid #91a29a;font-size:16px;}"
            )
        multiply = QLabel("×")
        multiply.setAlignment(Qt.AlignmentFlag.AlignCenter)
        multiply.setStyleSheet("font-size:20px;")
        grid_row.addWidget(grid_label)
        grid_row.addSpacing(14)
        grid_row.addWidget(self.grid_rows_spin)
        grid_row.addWidget(multiply)
        grid_row.addWidget(self.grid_columns_spin)
        grid_row.addStretch()
        root.addLayout(grid_row)
        self.grid_estimate_label = QLabel()
        self.grid_estimate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.grid_estimate_label.setStyleSheet("font-size:14px;color:#5a6962;")
        root.addWidget(self.grid_estimate_label)

        cancel_row = QHBoxLayout()
        cancel_row.addStretch()
        self.cancel_button = QPushButton("返回")
        self.cancel_button.setObjectName("cancelButton")
        cancel_row.addWidget(self.cancel_button)
        cancel_row.addStretch()
        root.addLayout(cancel_row)
        root.addStretch(3)

        self.fast_button.clicked.connect(lambda: self._choose(CalibrationMode.FAST))
        self.precise_button.clicked.connect(lambda: self._choose(CalibrationMode.PRECISE))
        self.cancel_button.clicked.connect(self.reject)
        self.grid_rows_spin.valueChanged.connect(self._update_grid_estimate)
        self.grid_columns_spin.valueChanged.connect(self._update_grid_estimate)
        self._update_grid_estimate()

    def _update_grid_estimate(self) -> None:
        point_count = self.grid_rows_spin.value() * self.grid_columns_spin.value()
        seconds = max(40, round(55 + (point_count - 9) * 1.55))
        self.grid_estimate_label.setText(f"当前网格预计约 {seconds} 秒")

    def _choose(self, mode: CalibrationMode) -> None:
        rows = self.grid_rows_spin.value() if mode is CalibrationMode.PRECISE else 3
        columns = self.grid_columns_spin.value() if mode is CalibrationMode.PRECISE else 3
        self.setModal(False)
        self.accept()
        self.mode_selected.emit(mode, rows, columns)


class CaptureWindow(QMainWindow):
    def __init__(self, controller: CaptureController, paths: AppPaths) -> None:
        super().__init__()
        self.controller = controller
        self.paths = paths
        self._calibration_mode = False
        self._highlight_id: str | None = None
        self.mode_dialog: CalibrationModeDialog | None = None
        self._last_preview_at = 0.0
        self._closing = False
        self._shutdown_complete = False
        self._grid_rows = 3
        self._grid_columns = 3
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
        self.validation_checkbox = QCheckBox("校准后进行精度验证（推荐）")
        self.calibrate_button = QPushButton("重新校准…")
        self.stream_button = QPushButton("恢复输出")
        self.stop_stream_button = QPushButton("暂停输出")
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
        self.result_label.setStyleSheet(
            "padding:10px 12px;background:#ffffff;border:1px solid #aab5bd;font-weight:600;"
        )
        root.addWidget(self.result_label)

        self.connect_button.pressed.connect(self._request_camera_connection)
        self.calibrate_button.clicked.connect(self._begin_calibration)
        self.stream_button.pressed.connect(self._request_stream_start)
        self.stop_stream_button.pressed.connect(self._request_stream_stop)
        self.retry_failed_button.clicked.connect(self._request_retry_failed_regions)
        self.save_anyway_button.clicked.connect(self.controller.save_anyway)
        controller.camera_state_changed.connect(self._on_camera_state)
        controller.calibration_point_changed.connect(self.show_calibration_point)
        controller.calibration_finished.connect(self._on_calibration_finished)
        controller.stream_state_changed.connect(self._on_stream_state)
        controller.preview_ready.connect(self._on_preview)
        controller.shutdown_finished.connect(self._finish_close)

    def _request_camera_connection(self) -> None:
        reconnecting = self.connect_button.text() == "重新连接"
        action = "重新连接" if reconnecting else "连接"
        LOGGER.info("ui_action camera_%s", "reconnect" if reconnecting else "connect")
        self.camera_status_label.setText(f"正在{action}摄像头…")
        self.result_label.setText(f"正在{action}摄像头，请稍候")
        self.connect_button.setEnabled(False)
        camera_index = self.camera_combo.currentIndex()
        QTimer.singleShot(50, lambda: self.controller.start_camera(camera_index))

    def _request_stream_start(self) -> None:
        LOGGER.info("ui_action stream_resume")
        self.result_label.setText("正在恢复眼动输出…")
        self.controller.start_streaming()

    def _request_stream_stop(self) -> None:
        LOGGER.info("ui_action stream_pause")
        self.result_label.setText("正在暂停眼动输出…")
        self.controller.stop_streaming()

    def _begin_calibration(self) -> None:
        LOGGER.info("ui_action calibration_open")
        self.result_label.setText("请选择校准方式")
        self.mode_dialog = CalibrationModeDialog(self)
        self.mode_dialog.mode_selected.connect(self._start_selected_calibration)
        self.mode_dialog.rejected.connect(self._cancel_mode_choice)
        self.mode_dialog.showFullScreen()
        self.mode_dialog.raise_()
        self.mode_dialog.activateWindow()

    def _request_retry_failed_regions(self) -> None:
        LOGGER.info("ui_action calibration_retry_failed_regions")
        self.result_label.setText("正在重校异常区域…")
        self.controller.retry_failed_regions()

    def _start_selected_calibration(
        self,
        mode: CalibrationMode,
        grid_rows: int,
        grid_columns: int,
    ) -> None:
        self._release_mode_dialog()
        self._calibration_mode = True
        self._grid_rows = int(grid_rows)
        self._grid_columns = int(grid_columns)
        self._controls.hide()
        self.showFullScreen()
        self.controller.start_calibration(
            mode,
            self.validation_checkbox.isChecked(),
            grid_rows,
            grid_columns,
        )

    def _cancel_mode_choice(self) -> None:
        self._release_mode_dialog()
        self._calibration_mode = False
        self._controls.show()

    def _release_mode_dialog(self) -> None:
        dialog = self.mode_dialog
        if dialog is None:
            return
        dialog.setModal(False)
        dialog.hide()
        dialog.deleteLater()
        self.mode_dialog = None

    def show_calibration_point(self, point_id: str) -> None:
        self._calibration_mode = True
        self._highlight_id = point_id
        self._controls.hide()
        if not self.isFullScreen():
            self.showFullScreen()
        self.update()

    def highlight_center(self) -> tuple[float, float] | None:
        if self._highlight_id is None:
            return None
        layout = build_layout(max(1, self.width()), max(1, self.height()))
        point_map = dict(calibration_points(layout))
        point_map.update(validation_points(layout))
        point_map.update(uniform_grid_calibration_points(
            layout,
            rows=self._grid_rows,
            columns=self._grid_columns,
        ))
        point_map.update(reentry_calibration_points(layout))
        point_map["pose_center"] = (layout.screen_width / 2.0, layout.screen_height / 2.0)
        return point_map.get(self._highlight_id)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if not self._calibration_mode:
            return
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#f7f8fa"))
        layout = build_layout(self.width(), self.height())
        painter.setPen(QPen(QColor("#a6b0b8"), 2))
        for rect in layout.grid_cells:
            painter.drawRect(round(rect.left), round(rect.top), round(rect.width), round(rect.height))
        center = self.highlight_center()
        if center is not None:
            painter.setBrush(QColor("#2f7d62"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(round(center[0] - 18), round(center[1] - 18), 36, 36)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#2f7d62"), 4))
            painter.drawEllipse(round(center[0] - 34), round(center[1] - 34), 68, 68)
        painter.setPen(QColor("#35443d"))
        font = painter.font()
        font.setPointSize(18)
        font.setBold(True)
        painter.setFont(font)
        if self._highlight_id == "pose_center":
            prompt = "保持自然坐姿并注视中心"
        elif self._highlight_id and self._highlight_id.startswith("reentry_"):
            prompt = "已检测到重新就座，请完成五点快速校正"
        else:
            prompt = "请注视绿色圆点"
        painter.drawText(
            0,
            28,
            self.width(),
            44,
            Qt.AlignmentFlag.AlignCenter,
            prompt,
        )

    def _on_camera_state(self, ready: bool, message: str) -> None:
        self.camera_status_label.setText(message)
        self.connect_button.setEnabled(True)
        self.connect_button.setText("重新连接" if ready else "连接摄像头")

    def _on_calibration_finished(self, passed: bool, message: str, failed_targets) -> None:
        self._release_mode_dialog()
        self.result_label.setText(message)
        self._calibration_mode = False
        self._controls.show()
        if self.isFullScreen():
            self.showNormal()
        show_choices = not passed and bool(failed_targets)
        self.retry_failed_button.setVisible(show_choices)
        self.save_anyway_button.hide()
        self.update()

    def _on_stream_state(self, running: bool, message: str) -> None:
        self.result_label.setText(message)
        self.stream_button.setEnabled(True)
        self.stop_stream_button.setEnabled(True)
        self.stream_button.setText("已在自动输出" if running else "恢复输出")

    def _on_preview(self, image: QImage) -> None:
        now = time.monotonic()
        if now - self._last_preview_at < 1.0 / 15.0:
            return
        self._last_preview_at = now
        pixmap = QPixmap.fromImage(image)
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and self._calibration_mode:
            self.controller.cancel_calibration()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._shutdown_complete:
            event.accept()
            return
        event.ignore()
        if self._closing:
            return
        LOGGER.info("ui_action window_close")
        self._closing = True
        if self.mode_dialog is not None:
            self.mode_dialog.close()
        self.hide()
        self.controller.stop()

    def _finish_close(self) -> None:
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        LOGGER.info("ui_window_close_finished")
        QTimer.singleShot(0, self.close)
