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
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .layout import MAIN_GRID_CELL_INDICES, SUBMENU_GRID_CELL_INDICES, build_layout
from .paths import AppPaths
from .settings import TypingSettings, save_settings
from .typing_controller import ConnectionStatus, ControllerUpdate, TypingController


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
            "QLabel,QCheckBox,QDoubleSpinBox,QSpinBox{font-size:15px;}"
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
        self.adaptive_checkbox = QCheckBox("实时自适应校正（推荐）")
        self.adaptive_checkbox.setChecked(settings.adaptive_correction_enabled)
        self.dwell_spin = QDoubleSpinBox()
        self.dwell_spin.setRange(0.5, 3.0)
        self.dwell_spin.setSingleStep(0.1)
        self.dwell_spin.setDecimals(1)
        self.dwell_spin.setSuffix(" 秒")
        self.dwell_spin.setValue(settings.dwell_seconds)
        self.blink_count_spin = QSpinBox()
        self.blink_count_spin.setRange(2, 5)
        self.blink_count_spin.setSuffix(" 次")
        self.blink_count_spin.setValue(settings.blink_return_count)
        form.addRow("眼动位置", self.gaze_checkbox)
        form.addRow("自适应", self.adaptive_checkbox)
        form.addRow("凝视停留时间", self.dwell_spin)
        form.addRow("连续眨眼返回次数", self.blink_count_spin)
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
        self.start_button.setEnabled(
            status.online and status.calibration_compatible and status.gaze_ready
        )

    def _start(self) -> None:
        self._connection_timer.stop()
        settings = TypingSettings(
            show_gaze_point=self.gaze_checkbox.isChecked(),
            dwell_seconds=self.dwell_spin.value(),
            validate_calibration=False,
            adaptive_correction_enabled=self.adaptive_checkbox.isChecked(),
            blink_return_count=self.blink_count_spin.value(),
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
        self.blink_return_count = settings.blink_return_count
        self.target_labels: tuple[str, ...] = ("",) * 8
        self.update_state: ControllerUpdate | None = None
        self.setMinimumSize(800, 600)
        self.history_view = QPlainTextEdit(self)
        self.history_view.setReadOnly(True)
        self.history_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.history_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.history_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.history_view.setFont(QFont("Microsoft YaHei", 15))
        self.history_view.document().setDocumentMargin(0)
        self.history_view.setStyleSheet(
            "QPlainTextEdit{background:#111416;color:#f4f7f6;border:1px solid #515b58;"
            "padding:4px 10px;}"
            "QScrollBar:vertical{width:10px;background:#111416;}"
            "QScrollBar::handle:vertical{background:#53cda8;min-height:20px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )
        self._sync_history_geometry()

    def set_controller_update(self, update: ControllerUpdate) -> None:
        self.update_state = update
        self.target_labels = update.target_labels
        if self.history_view.toPlainText() != update.current_text:
            scroll = self.history_view.verticalScrollBar()
            was_at_bottom = scroll.value() >= scroll.maximum() - 1
            self.history_view.setPlainText(update.current_text)
            if was_at_bottom:
                scroll.setValue(scroll.maximum())
        self.update()

    def resizeEvent(self, event) -> None:
        self._sync_history_geometry()
        super().resizeEvent(event)

    def _sync_history_geometry(self) -> None:
        layout = build_layout(max(1, self.width()), max(1, self.height()))
        line_height = self.history_view.fontMetrics().lineSpacing()
        top = 36
        height = min(
            line_height * 3 + 16,
            max(line_height + 16, round(layout.top_bar.height) - top - 8),
        )
        self.history_view.setGeometry(
            10,
            top,
            max(1, round(layout.top_bar.width) - 20),
            height,
        )

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#090b0c"))
        layout = build_layout(max(1, self.width()), max(1, self.height()))
        update = self.update_state
        labels = self.target_labels
        current_target = None if update is None else update.dwell_target_id
        progress = 0.0 if update is None else update.dwell_progress

        painter.setPen(QColor("#f3f6f5"))
        painter.setFont(QFont("Microsoft YaHei", 13, QFont.Weight.DemiBold))
        painter.drawText(18, 4, 250, 28, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "纯眼动打字器")
        if update is not None and update.blink_count:
            painter.setPen(QColor("#e5ad52"))
            painter.drawText(
                self.width() - 190,
                4,
                172,
                28,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"眨眼 {update.blink_count}/{update.blink_required}",
            )
        elif update is not None:
            painter.setPen(QColor("#55d2ac" if update.status.online else "#e5ad52"))
            painter.drawText(
                self.width() - 300,
                4,
                282,
                28,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                "眼动已连接  ·  自适应校正开启"
                if update.status.online
                else update.status.message,
            )

        cell_indices = MAIN_GRID_CELL_INDICES if len(labels) == 8 else SUBMENU_GRID_CELL_INDICES
        labels_by_cell = {cell_index: labels[position] for position, cell_index in enumerate(cell_indices)}
        positions_by_cell = {cell_index: position for position, cell_index in enumerate(cell_indices)}
        font_size = 25 if len(labels) == 8 else 30
        painter.setFont(QFont("Microsoft YaHei", font_size, QFont.Weight.DemiBold))
        for cell_index, rect in enumerate(layout.grid_cells):
            position = positions_by_cell.get(cell_index)
            label = labels_by_cell.get(cell_index, "")
            active = position is not None and current_target == f"target_{position}"
            inset = 5
            draw_left = round(rect.left) + inset
            draw_top = round(rect.top) + inset
            draw_width = max(1, round(rect.width) - inset * 2)
            draw_height = max(1, round(rect.height) - inset * 2)
            if active:
                fill = QColor("#334640")
                border = QColor("#55d2ac")
            elif label:
                fill = QColor("#303437")
                border = QColor("#4c5355")
            else:
                fill = QColor("#171a1c")
                border = QColor("#303638")
            painter.setPen(QPen(border, 4 if active else 1))
            painter.setBrush(fill)
            painter.drawRect(draw_left, draw_top, draw_width, draw_height)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(
                draw_left,
                draw_top,
                draw_width,
                draw_height,
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                label,
            )
            if active:
                painter.fillRect(
                    draw_left + 10,
                    draw_top + draw_height - 14,
                    round((draw_width - 20) * progress),
                    9,
                    QColor("#55d2ac"),
                )

        if update is not None and update.preparing:
            painter.fillRect(
                round(layout.keyboard_bounds.left),
                round(layout.keyboard_bounds.top),
                round(layout.keyboard_bounds.width),
                round(layout.keyboard_bounds.height),
                QColor(9, 11, 12, 225),
            )
            painter.setBrush(QColor("#55d2ac"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(self.width() // 2 - 18, self.height() // 2 - 18, 36, 36)
            painter.setPen(QColor("#f3f3f3"))
            painter.setFont(QFont("Microsoft YaHei", 18))
            painter.drawText(0, self.height() // 2 + 42, self.width(), 40, Qt.AlignmentFlag.AlignCenter, "请注视中心")

        if update is not None and update.message:
            painter.setPen(QColor("#e5ad52"))
            painter.setFont(QFont("Microsoft YaHei", 13))
            painter.drawText(280, 4, max(1, self.width() - 580), 28, Qt.AlignmentFlag.AlignCenter, update.message)
        if (
            update is not None
            and self.show_gaze_point
            and update.gaze_point is not None
            and not update.preparing
        ):
            x, y = update.gaze_point
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(QColor("#e23b3b"))
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
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(lambda: controller.tick(time.monotonic()))
        self._timer.start()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._timer.stop()
        self.controller.end_session()
        self.closed.emit()
        event.accept()
