from pathlib import Path

import pytest
from PyQt6.QtCore import QObject, pyqtSignal

from pure_gaze_typing.capture_window import CaptureWindow
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
