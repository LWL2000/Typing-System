from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage

from .calibration import CalibrationMetadata, CalibrationStore, StoredCalibration
from .eyetrax_runtime import EyeTraxRuntime, FrameObservation, GazeEstimate


@dataclass(frozen=True)
class CapturePacket:
    timestamp: float
    observation: FrameObservation
    estimate: GazeEstimate
    fps: float


class CameraWorker(QObject):
    camera_opened = pyqtSignal(bool, str)
    preview_ready = pyqtSignal(object)
    packet_ready = pyqtSignal(object)
    model_saved = pyqtSignal(object)
    failed = pyqtSignal(str)
    stopped = pyqtSignal()

    def __init__(
        self,
        camera_index: int,
        runtime: EyeTraxRuntime,
        store: CalibrationStore,
    ) -> None:
        super().__init__()
        self.camera_index = int(camera_index)
        self.runtime = runtime
        self.store = store
        self._capture = None
        self._timer: QTimer | None = None
        self._previous_frame_at: float | None = None

    @pyqtSlot()
    def start(self) -> None:
        self._capture = cv2.VideoCapture(self.camera_index)
        if not self._capture.isOpened():
            self.camera_opened.emit(False, f"无法打开摄像头 {self.camera_index}")
            self._capture.release()
            self._capture = None
            return
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_frame)
        self._timer.start(0)
        self.camera_opened.emit(True, f"摄像头 {self.camera_index} 已连接")

    @pyqtSlot()
    def _read_frame(self) -> None:
        if self._capture is None:
            return
        ok, frame = self._capture.read()
        if not ok:
            self.failed.emit("摄像头画面读取失败")
            return
        now = time.monotonic()
        fps = 0.0 if self._previous_frame_at is None else 1.0 / max(now - self._previous_frame_at, 1e-6)
        self._previous_frame_at = now
        try:
            observation = self.runtime.extract(frame)
            estimate = self.runtime.estimate(observation)
        except Exception as error:
            self.failed.emit(f"眼动处理失败：{error}")
            return
        self.packet_ready.emit(CapturePacket(now, observation, estimate, fps))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        self.preview_ready.emit(image)

    @pyqtSlot(object, object, object)
    def train_and_save(self, features, labels, metadata: CalibrationMetadata) -> None:
        if self._timer is not None:
            self._timer.stop()
        try:
            self.runtime.train(features, labels)
            self.runtime.set_metadata(metadata)
            stored = self.store.save(self.runtime, metadata)
            self.model_saved.emit(stored)
        except Exception as error:
            self.failed.emit(f"校准模型保存失败：{error}")
        finally:
            if self._timer is not None:
                self._timer.start(0)

    @pyqtSlot(object)
    def load_calibration(self, stored: StoredCalibration) -> None:
        try:
            self.runtime.load_calibration(stored)
        except Exception as error:
            self.failed.emit(f"校准模型加载失败：{error}")

    @pyqtSlot()
    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self.runtime.close()
        self.stopped.emit()
