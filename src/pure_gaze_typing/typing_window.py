from __future__ import annotations

import time

from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCloseEvent, QFont, QPainter, QPen, QShowEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .layout import build_layout
from .paths import AppPaths
from .settings import TypingSettings, save_settings
from .typing_controller import ConnectionStatus, ControllerUpdate, TypingController
from .typing_engine import PageKind


class StartupWindow(QMainWindow):
    start_requested = pyqtSignal(object)

    def __init__(
        self,
        controller: TypingController,
        settings: TypingSettings,
        paths: AppPaths,
    ) -> None:
        super().__init__()
        self.controller = controller
        self.paths = paths
        self.setWindowTitle("纯眼动打字器")
        self.setMinimumSize(620, 400)
        self.setStyleSheet(
            "QMainWindow{background:#f4f6f8;color:#17212b;}"
            "QPushButton{min-height:42px;border:1px solid #8d9aa5;background:white;padding:0 18px;}"
            "QPushButton:disabled{color:#89939b;background:#e9edef;}"
            "QLabel,QCheckBox,QDoubleSpinBox{font-size:15px;}"
        )
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        title = QLabel("纯眼动打字器")
        title.setStyleSheet("font-size:28px;font-weight:600;")
        root.addWidget(title)
        self.status_label = QLabel("等待眼动采集程序")
        self.status_label.setStyleSheet("padding:12px;background:#ffffff;border:1px solid #bbc4cb;")
        root.addWidget(self.status_label)
        form = QFormLayout()
        self.gaze_checkbox = QCheckBox("显示实时眼动位置")
        self.gaze_checkbox.setChecked(settings.show_gaze_point)
        self.dwell_spin = QDoubleSpinBox()
        self.dwell_spin.setRange(0.5, 3.0)
        self.dwell_spin.setSingleStep(0.1)
        self.dwell_spin.setDecimals(1)
        self.dwell_spin.setSuffix(" 秒")
        self.dwell_spin.setValue(settings.dwell_seconds)
        form.addRow("眼动位置", self.gaze_checkbox)
        form.addRow("凝视停留时间", self.dwell_spin)
        root.addLayout(form)
        root.addStretch(1)
        actions = QHBoxLayout()
        self.start_button = QPushButton("开始实验")
        self.start_button.setEnabled(False)
        self.exit_button = QPushButton("退出")
        actions.addStretch(1)
        actions.addWidget(self.start_button)
        actions.addWidget(self.exit_button)
        root.addLayout(actions)
        self.start_button.clicked.connect(self._start)
        self.exit_button.clicked.connect(self.close)
        controller.status_changed.connect(self._on_status)
        self._connection_timer = QTimer(self)
        self._connection_timer.setInterval(33)
        self._connection_timer.timeout.connect(
            lambda: controller.tick(time.monotonic())
        )
        self._on_status(controller.status if hasattr(controller, "status") else ConnectionStatus(False, False, "等待眼动采集程序"))

    def _on_status(self, status: ConnectionStatus) -> None:
        self.status_label.setText(status.message)
        self.start_button.setEnabled(status.online and status.calibration_compatible)

    def _start(self) -> None:
        self._connection_timer.stop()
        settings = TypingSettings(
            self.gaze_checkbox.isChecked(),
            self.dwell_spin.value(),
            False,
        )
        save_settings(self.paths.settings_file, settings)
        self.controller.update_settings(settings)
        self.start_requested.emit(settings)

    def showEvent(self, event: QShowEvent) -> None:
        self._connection_timer.start()
        super().showEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._connection_timer.stop()
        super().closeEvent(event)


class KeyboardCanvas(QWidget):
    def __init__(self, settings: TypingSettings) -> None:
        super().__init__()
        self.show_gaze_point = settings.show_gaze_point
        self.target_labels: tuple[str, ...] = ("",) * 6
        self.update_state: ControllerUpdate | None = None
        self.setMinimumSize(800, 600)

    def set_controller_update(self, update: ControllerUpdate) -> None:
        self.update_state = update
        self.target_labels = update.target_labels
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f7f8fa"))
        layout = build_layout(max(1, self.width()), max(1, self.height()))
        update = self.update_state
        labels = self.target_labels
        current_target = None if update is None else update.dwell_target_id
        progress = 0.0 if update is None else update.dwell_progress

        painter.setPen(QColor("#17212b"))
        painter.setFont(QFont("Microsoft YaHei", 16))
        top_text = "" if update is None else update.current_text
        painter.drawText(
            round(layout.top_bar.left + layout.back_target.width + 14),
            round(layout.top_bar.top),
            round(layout.top_bar.width - layout.back_target.width - 28),
            round(layout.top_bar.height),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            top_text,
        )
        show_back = update is not None and update.page_kind is not PageKind.MAIN and not update.preparing
        if show_back:
            back = layout.back_target
            painter.setPen(QPen(QColor("#41505d"), 2))
            painter.setBrush(QColor("#ffffff"))
            painter.drawRect(round(back.left), round(back.top), round(back.width), round(back.height))
            painter.drawText(
                round(back.left), round(back.top), round(back.width), round(back.height),
                Qt.AlignmentFlag.AlignCenter, "← 返回",
            )
            if current_target == "back":
                painter.fillRect(
                    round(back.left), round(back.bottom - 7), round(back.width * progress), 7,
                    QColor("#2f7d62"),
                )

        painter.setFont(QFont("Microsoft YaHei", 24, QFont.Weight.DemiBold))
        for index, rect in enumerate(layout.targets):
            painter.setPen(QPen(QColor("#7f8b95"), 2))
            painter.setBrush(QColor("#ffffff" if labels[index] else "#edf0f2"))
            painter.drawRect(round(rect.left), round(rect.top), round(rect.width), round(rect.height))
            painter.setPen(QColor("#17212b"))
            painter.drawText(
                round(rect.left), round(rect.top), round(rect.width), round(rect.height),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                labels[index],
            )
            if current_target == f"target_{index}":
                painter.fillRect(
                    round(rect.left), round(rect.bottom - 9), round(rect.width * progress), 9,
                    QColor("#2f7d62"),
                )

        if update is not None and update.preparing:
            painter.setBrush(QColor("#1f6f5a"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self.width() // 2 - 18, self.height() // 2 - 18, 36, 36)
            painter.setPen(QColor("#17212b"))
            painter.setFont(QFont("Microsoft YaHei", 18))
            painter.drawText(0, self.height() // 2 + 42, self.width(), 40, Qt.AlignmentFlag.AlignCenter, "请注视中心")

        if update is not None and update.message:
            painter.setPen(QColor("#9c3f2f"))
            painter.setFont(QFont("Microsoft YaHei", 15))
            painter.drawText(0, self.height() - 48, self.width(), 32, Qt.AlignmentFlag.AlignCenter, update.message)
        if update is not None and update.blink_count:
            painter.setPen(QColor("#925f17"))
            painter.drawText(self.width() - 170, 24, 150, 30, Qt.AlignmentFlag.AlignRight, f"眨眼 {update.blink_count}/3")
        if (
            update is not None
            and self.show_gaze_point
            and update.gaze_point is not None
            and not update.preparing
        ):
            x, y = update.gaze_point
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(QColor("#d53f3f"))
            painter.drawEllipse(round(x - 8), round(y - 8), 16, 16)


class TypingWindow(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, controller: TypingController, settings: TypingSettings) -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("纯眼动打字器")
        self.canvas = KeyboardCanvas(settings)
        self.setCentralWidget(self.canvas)
        self.canvas.set_controller_update(controller.current_update())
        controller.update_ready.connect(self.canvas.set_controller_update)
        controller.session_finished.connect(self._on_session_finished)
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(lambda: controller.tick(time.monotonic()))
        self._timer.start()

    def _on_session_finished(self, _text: str) -> None:
        QTimer.singleShot(450, self.close)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._timer.stop()
        self.controller.end_session()
        self.closed.emit()
        event.accept()
