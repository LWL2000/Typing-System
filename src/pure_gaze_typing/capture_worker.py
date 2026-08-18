from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import threading
import time
from typing import Callable

import cv2
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from .calibration import CalibrationMetadata, CalibrationStore, StoredCalibration
from .eyetrax_runtime import EyeTraxRuntime, FrameObservation, GazeEstimate


LOGGER = logging.getLogger("pure_gaze_typing.capture")


@dataclass(frozen=True)
class CapturePacket:
    timestamp: float
    observation: FrameObservation
    estimate: GazeEstimate
    fps: float


class CameraWorker(QObject):
    FRAME_INTERVAL_MS = 33
    PREVIEW_INTERVAL_SECONDS = 0.10

    camera_opened = pyqtSignal(bool, str)
    preview_ready = pyqtSignal(object)
    packet_ready = pyqtSignal(object)
    model_trained = pyqtSignal(object)
    model_configured = pyqtSignal(object)
    model_saved = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()
    runtime_ready = pyqtSignal(object)

    def __init__(
        self,
        camera_index: int,
        runtime: EyeTraxRuntime | None,
        store: CalibrationStore | None,
        *,
        runtime_factory: Callable[..., EyeTraxRuntime] | None = None,
        runtime_args: tuple[object, ...] = (),
        stored_calibration: StoredCalibration | None = None,
    ) -> None:
        super().__init__()
        self.camera_index = int(camera_index)
        self.runtime = runtime
        self.store = store
        self.runtime_factory = runtime_factory
        self.runtime_args = tuple(runtime_args)
        self.stored_calibration = stored_calibration
        self._capture = None
        self._timer: QTimer | None = None
        self._previous_frame_at: float | None = None
        self._last_preview_at: float | None = None
        self._first_frame_logged = False
        self._stop_requested = threading.Event()
        self._stopped = False

    @pyqtSlot()
    def start(self) -> None:
        LOGGER.info("camera_worker_start index=%s", self.camera_index)
        if self._stop_requested.is_set():
            self._finish_stop()
            return
        if self.runtime is None:
            if self.runtime_factory is None:
                self.failed.emit("眼动运行时初始化器缺失")
                self._finish_stop()
                return
            try:
                self.runtime = self.runtime_factory(*self.runtime_args)
                if self.stored_calibration is not None:
                    self.runtime.load_calibration(self.stored_calibration)
            except Exception as error:
                LOGGER.exception("camera_runtime_initialization_failed")
                self.failed.emit(f"眼动运行时初始化失败：{error}")
                self._finish_stop()
                return
        self.runtime_ready.emit(self.runtime)
        if self._stop_requested.is_set():
            self._finish_stop()
            return
        self._capture = cv2.VideoCapture(self.camera_index)
        LOGGER.info("camera_capture_created")
        if not self._capture.isOpened():
            self.camera_opened.emit(False, f"无法打开摄像头 {self.camera_index}")
            self._capture.release()
            self._capture = None
            self._finish_stop()
            return
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_frame)
        self._timer.start(self.FRAME_INTERVAL_MS)
        LOGGER.info("camera_timer_started interval_ms=%s", self.FRAME_INTERVAL_MS)
        self.camera_opened.emit(True, f"摄像头 {self.camera_index} 已连接")

    @pyqtSlot()
    def _read_frame(self) -> None:
        if self._stop_requested.is_set():
            self.stop()
            return
        if self._capture is None or self.runtime is None:
            return
        ok, frame = self._capture.read()
        if not ok:
            self.failed.emit("摄像头画面读取失败")
            return
        if not self._first_frame_logged:
            LOGGER.info("camera_first_frame_read shape=%s", frame.shape)
        now = time.monotonic()
        fps = 0.0 if self._previous_frame_at is None else 1.0 / max(now - self._previous_frame_at, 1e-6)
        self._previous_frame_at = now
        try:
            observation = self.runtime.extract(frame)
            estimate = self.runtime.estimate(observation, timestamp=now)
        except Exception as error:
            self.failed.emit(f"眼动处理失败：{error}")
            return
        if not self._first_frame_logged:
            LOGGER.info("camera_first_frame_processed face_detected=%s", observation.face_detected)
        self.packet_ready.emit(CapturePacket(now, observation, estimate, fps))
        if (
            self._last_preview_at is None
            or now - self._last_preview_at >= self.PREVIEW_INTERVAL_SECONDS
        ):
            self._last_preview_at = now
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(
                rgb.data,
                width,
                height,
                channels * width,
                QImage.Format.Format_RGB888,
            ).copy()
            self.preview_ready.emit(image)
            if not self._first_frame_logged:
                LOGGER.info("camera_first_preview_emitted size=%sx%s", width, height)
                self._first_frame_logged = True

    @pyqtSlot(object, object, object)
    def train_model(self, features, labels, metadata: CalibrationMetadata) -> None:
        if self.runtime is None:
            self.failed.emit("眼动运行时未就绪")
            return
        if self._timer is not None:
            self._timer.stop()
        try:
            self.runtime.train(features, labels)
            threshold = self.runtime.feature_range_threshold(features)
            metadata = replace(metadata, feature_range_threshold=threshold)
            self.runtime.set_metadata(metadata)
            self.model_trained.emit(metadata)
        except Exception as error:
            self.failed.emit(f"校准模型训练失败：{error}")
        finally:
            if self._timer is not None:
                self._timer.start(self.FRAME_INTERVAL_MS)

    @pyqtSlot(object)
    def configure_metadata(self, metadata: CalibrationMetadata) -> None:
        if self.runtime is None:
            self.failed.emit("眼动运行时未就绪")
            return
        try:
            self.runtime.set_metadata(metadata)
            self.model_configured.emit(metadata)
        except Exception as error:
            self.failed.emit(f"校准偏差修正失败：{error}")

    @pyqtSlot(object)
    def save_current(self, metadata: CalibrationMetadata) -> None:
        if self.runtime is None or self.store is None:
            self.failed.emit("眼动运行时或校准存储未就绪")
            return
        if self._timer is not None:
            self._timer.stop()
        try:
            self.runtime.set_metadata(metadata)
            stored = self.store.save(self.runtime, metadata)
            self.model_saved.emit(stored)
        except Exception as error:
            self.failed.emit(f"校准模型保存失败：{error}")
        finally:
            if self._timer is not None:
                self._timer.start(self.FRAME_INTERVAL_MS)

    @pyqtSlot(object)
    def load_calibration(self, stored: StoredCalibration) -> None:
        if self.runtime is None:
            self.failed.emit("眼动运行时未就绪")
            return
        try:
            self.runtime.load_calibration(stored)
        except Exception as error:
            self.failed.emit(f"校准模型加载失败：{error}")

    @pyqtSlot()
    def stop(self) -> None:
        self._stop_requested.set()
        self._finish_stop()

    def request_stop(self) -> None:
        """Thread-safe stop flag; resource cleanup remains on the worker thread."""
        self._stop_requested.set()

    def _finish_stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        LOGGER.info("camera_worker_stopping")
        if self._timer is not None:
            self._timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self.runtime is not None:
            try:
                self.runtime.close()
            except Exception:
                LOGGER.exception("camera_runtime_close_failed")
        LOGGER.info("camera_worker_stopped")
        self.stopped.emit()
