from __future__ import annotations

from dataclasses import asdict, dataclass
import time

from PyQt6.QtCore import QObject, pyqtSignal

from .adaptive_gaze import AdaptiveDecision, AdaptiveGazeSession
from .eyetrax_runtime import CenterDriftCorrector
from .gaze_selection import DwellSelector, TripleBlinkDetector
from .layout import LAYOUT_VERSION, LayoutSpec, build_layout, hit_test
from .paths import AppPaths
from .protocol import GazeSample, Heartbeat, ProtocolMessage, UdpReceiver
from .session_log import SessionRecorder
from .settings import TypingSettings
from .typing_engine import PageKind, TypingEngine


@dataclass(frozen=True)
class ConnectionStatus:
    online: bool
    calibration_compatible: bool
    message: str
    gaze_ready: bool = False


@dataclass(frozen=True)
class ControllerUpdate:
    status: ConnectionStatus
    target_labels: tuple[str, ...]
    page_kind: PageKind
    current_text: str
    gaze_point: tuple[float, float] | None
    dwell_target_id: str | None
    dwell_progress: float
    blink_count: int
    message: str
    preparing: bool


class TypingController(QObject):
    status_changed = pyqtSignal(object)
    update_ready = pyqtSignal(object)
    session_finished = pyqtSignal(str)

    def __init__(
        self,
        paths: AppPaths,
        settings: TypingSettings,
        screen_width: int,
        screen_height: int,
        *,
        receiver: UdpReceiver | None = None,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.settings = settings
        self.layout: LayoutSpec = build_layout(screen_width, screen_height)
        self.receiver = receiver or UdpReceiver()
        self.engine = TypingEngine()
        self.dwell = self._new_dwell(settings)
        self.blinks = TripleBlinkDetector()
        self.drift = CenterDriftCorrector(screen_width, screen_height)
        self.adaptive = self._new_adaptive(settings)
        self.recorder: SessionRecorder | None = None
        self._status = ConnectionStatus(False, False, "等待眼动采集程序")
        self._expected_calibration_id = ""
        self._last_message_at: float | None = None
        self._last_valid_gaze_at: float | None = None
        self._last_gaze_point: tuple[float, float] | None = None
        self._last_dwell_target: str | None = None
        self._last_dwell_progress = 0.0
        self._blink_count = 0
        self._message = ""
        self._preparing = False
        self._prepare_started_at: float | None = None
        self.last_triggered_target: str | None = None

    @property
    def status(self) -> ConnectionStatus:
        return self._status

    def update_settings(self, settings: TypingSettings) -> None:
        was_enabled = self.settings.adaptive_correction_enabled
        self.settings = settings
        self.dwell = self._new_dwell(settings)
        self.adaptive.set_enabled(settings.adaptive_correction_enabled)
        if was_enabled and not settings.adaptive_correction_enabled:
            self._record_event("adaptive_disabled", {"reason": "user_setting"})

    def start_session(self, *, skip_prepare: bool = False) -> None:
        if self.recorder is not None:
            self.end_session()
        self.engine = TypingEngine()
        self.dwell = self._new_dwell(self.settings)
        self.blinks.reset()
        self.drift = CenterDriftCorrector(self.layout.screen_width, self.layout.screen_height)
        self.adaptive = self._new_adaptive(self.settings)
        self.recorder = SessionRecorder.start(
            self.paths,
            self.settings,
            {
                "calibration_id": self._expected_calibration_id,
                "layout_version": self.layout.version,
                "screen_width": self.layout.screen_width,
                "screen_height": self.layout.screen_height,
            },
            error_callback=lambda message: self._set_message(message),
        )
        self._record_event(
            "adaptive_reset",
            {"reason": "new_session", "matrix_version": 0},
        )
        if not self.settings.adaptive_correction_enabled:
            self._record_event("adaptive_disabled", {"reason": "user_setting"})
        self._preparing = not skip_prepare
        self._prepare_started_at = None
        self._message = "请注视屏幕中心" if self._preparing else ""
        self._emit_update()

    def end_session(self) -> None:
        self.adaptive.clear_window()
        if self.recorder is not None:
            self.recorder.finish(self.engine.full_text())
            self.recorder = None

    def close(self) -> None:
        self.end_session()
        self.receiver.close()

    def tick(self, now: float | None = None) -> ControllerUpdate:
        current = time.monotonic() if now is None else float(now)
        for message in self.receiver.poll():
            self.process_message(message, current)
        if self._last_message_at is not None and current - self._last_message_at > 2.0:
            self._disconnect("眼动连接中断")
        return self.current_update()

    def process_message(self, message: ProtocolMessage, now: float) -> ControllerUpdate:
        current = float(now)
        self._last_message_at = current
        if isinstance(message, Heartbeat):
            previous_calibration_id = self._expected_calibration_id
            compatible = message.calibration_ready and message.layout_version == self.layout.version
            calibration_changed = bool(previous_calibration_id) and (
                message.calibration_id != previous_calibration_id
                or message.layout_version != self.layout.version
                or not message.calibration_ready
            )
            if calibration_changed:
                decision = self.adaptive.reset("calibration_changed")
                self._record_event("adaptive_reset", asdict(decision))
                self.dwell.reset()
            if message.calibration_id != previous_calibration_id:
                self._last_valid_gaze_at = None
            self._expected_calibration_id = message.calibration_id if compatible else ""
            if not message.camera_ok:
                self._disconnect("摄像头未就绪")
            elif not compatible:
                self._set_status(False, False, "校准不可用或界面尺寸不匹配", False)
            else:
                gaze_ready = (
                    self._last_valid_gaze_at is not None
                    and current - self._last_valid_gaze_at <= 2.0
                )
                status_message = "眼动数据已就绪" if gaze_ready else "采集端已连接，等待有效眼动数据"
                self._set_status(True, True, status_message, gaze_ready)
            return self.current_update()

        compatible = (
            bool(self._expected_calibration_id)
            and message.calibration_id == self._expected_calibration_id
            and message.layout_version == self.layout.version
        )
        if not compatible:
            if self._expected_calibration_id:
                decision = self.adaptive.reset("calibration_changed")
                self._record_event("adaptive_reset", asdict(decision))
                self._expected_calibration_id = ""
            self._last_valid_gaze_at = None
            self._set_status(False, False, "眼动校准标识不匹配", False)
            self.dwell.reset()
            return self.current_update()

        if message.valid and message.screen_x is not None and message.screen_y is not None:
            self._last_valid_gaze_at = current
            self._set_status(True, True, "眼动数据已就绪", True)

        blink_update = self.blinks.update(
            current,
            face_detected=message.face_detected,
            blink=message.blink,
        )
        self._blink_count = blink_update.count
        if blink_update.triple_blink and self.engine.page_kind is not PageKind.MAIN:
            self.engine.return_to_main()
            self.dwell.reset()
            self.adaptive.clear_window()
            self.last_triggered_target = "triple_blink_return"
            self._set_message("已返回主菜单")
            self._record_event("triple_blink_return", {})
            self._emit_update()
            return self.current_update()

        if not self._status.online:
            self._emit_update()
            return self.current_update()

        base_point: tuple[float, float] | None = None
        if message.valid and message.screen_x is not None and message.screen_y is not None:
            base_point = self.drift.apply(message.screen_x, message.screen_y)

        if self._preparing:
            self._last_gaze_point = base_point
            self._process_prepare(message, current, base_point)
            self._record_gaze(message, None, 0.0)
            self._emit_update()
            return self.current_update()

        corrected = self._adaptive_point(message, current, base_point)
        if corrected is not None:
            self._last_gaze_point = corrected

        geometry_target = None
        if corrected is not None:
            geometry_target = hit_test(
                self.layout,
                corrected[0],
                corrected[1],
                target_count=len(self.engine.targets()),
            )
        logical_target = self._logical_target(geometry_target)
        dwell_update = self.dwell.update(
            current,
            geometry_target,
            valid=message.valid,
            blink=message.blink,
        )
        self._last_dwell_target = dwell_update.target_id
        self._last_dwell_progress = dwell_update.progress
        self._record_gaze(message, geometry_target, dwell_update.progress)
        if dwell_update.triggered_target_id is not None:
            triggered_logical = self._logical_target(dwell_update.triggered_target_id)
            if triggered_logical is not None:
                self._consider_adaptive_anchor(dwell_update.triggered_target_id)
                self._activate(triggered_logical)
        elif logical_target is None and not message.blink:
            self._message = ""
        self._emit_update()
        return self.current_update()

    def current_update(self) -> ControllerUpdate:
        targets = self.engine.targets()
        labels = ["" for _ in range(len(targets))]
        for target in targets:
            labels[target.position] = target.label
        return ControllerUpdate(
            self._status,
            tuple(labels),
            self.engine.page_kind,
            self.engine.full_text(),
            self._last_gaze_point,
            self._last_dwell_target,
            self._last_dwell_progress,
            self._blink_count,
            self._message,
            self._preparing,
        )

    def _new_dwell(self, settings: TypingSettings) -> DwellSelector:
        return DwellSelector(settings.dwell_seconds)

    def _new_adaptive(self, settings: TypingSettings) -> AdaptiveGazeSession:
        return AdaptiveGazeSession(
            screen_size=(self.layout.screen_width, self.layout.screen_height),
            enabled=settings.adaptive_correction_enabled,
        )

    def _adaptive_point(
        self,
        message: GazeSample,
        now: float,
        base_point: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if not self.adaptive.enabled:
            return base_point
        try:
            if base_point is None:
                self.adaptive.observe(
                    now,
                    valid=False,
                    blink=message.blink,
                    quality=message.quality,
                )
                return None
            self.adaptive.observe(
                now,
                valid=message.valid,
                blink=message.blink,
                x=base_point[0],
                y=base_point[1],
                quality=message.quality,
            )
            return self.adaptive.apply(*base_point)
        except Exception as error:
            self.adaptive.set_enabled(False)
            self._record_event(
                "adaptive_disabled",
                {"reason": "runtime_error", "error": str(error)},
            )
            self._message = "自适应校正异常，已使用基础眼动"
            return base_point

    def _consider_adaptive_anchor(self, geometry_target: str) -> None:
        position = int(geometry_target.removeprefix("target_"))
        rect = self.layout.targets_for(len(self.engine.targets()))[position]
        try:
            decision = self.adaptive.consider_anchor(
                geometry_target,
                (rect.left, rect.top, rect.width, rect.height),
            )
        except Exception as error:
            self.adaptive.set_enabled(False)
            self._record_event(
                "adaptive_disabled",
                {"reason": "runtime_error", "error": str(error)},
            )
            self._message = "自适应校正异常，已使用基础眼动"
            return
        self._record_adaptive_decision(decision)

    def _record_adaptive_decision(self, decision: AdaptiveDecision) -> None:
        if decision.rollback_performed:
            event = "adaptive_rollback"
            self._message = "自适应校正已回滚，已暂停继续学习"
        elif decision.accepted:
            event = "adaptive_update_accepted"
        else:
            event = "adaptive_update_rejected"
        self._record_event(event, asdict(decision))

    def _process_prepare(
        self,
        message: GazeSample,
        now: float,
        corrected: tuple[float, float] | None,
    ) -> None:
        if corrected is None or not message.valid or message.blink:
            return
        if self._prepare_started_at is None:
            self._prepare_started_at = now
        elapsed = now - self._prepare_started_at
        if elapsed < 0.25:
            return
        self.drift.collect(message.screen_x, message.screen_y)
        if elapsed >= 1.25:
            self.drift.finish((self.layout.screen_width / 2.0, self.layout.screen_height / 2.0))
            self._preparing = False
            self._message = ""

    def _logical_target(self, geometry_target: str | None) -> str | None:
        if geometry_target is None or not geometry_target.startswith("target_"):
            return None
        position = int(geometry_target.removeprefix("target_"))
        target = next((item for item in self.engine.targets() if item.position == position), None)
        return None if target is None else target.target_id

    def _activate(self, logical_target: str) -> None:
        self.adaptive.clear_window()
        self.last_triggered_target = logical_target
        self._last_dwell_target = None
        self._last_dwell_progress = 0.0
        effect = self.engine.activate(logical_target)
        self._record_event("selection", {"target_id": logical_target, "action": effect.action})
        if effect.sent_text is not None:
            if self.recorder is None:
                self._set_message("当前没有实验记录，无法发送")
                return
            self._record_event("line_sent", {"text": self.engine.current_line})
            self.engine.confirm_send()
            self._set_message("已换行，可继续输入")

    def _record_gaze(self, sample: GazeSample, target_id: str | None, progress: float) -> None:
        if self.recorder is not None:
            self.recorder.record_gaze(sample, target_id, progress)

    def _record_event(self, name: str, payload: dict[str, object]) -> None:
        if self.recorder is not None:
            self.recorder.record_event(name, payload)

    def _disconnect(self, message: str) -> None:
        self.dwell.reset()
        self.blinks.reset()
        self.adaptive.clear_window()
        self._last_valid_gaze_at = None
        self._last_dwell_target = None
        self._last_dwell_progress = 0.0
        self._set_status(False, self._status.calibration_compatible, message, False)

    def _set_status(self, online: bool, compatible: bool, message: str, gaze_ready: bool = False) -> None:
        status = ConnectionStatus(online, compatible, message, gaze_ready)
        if status != self._status:
            self._status = status
            self.status_changed.emit(status)
            self._emit_update()

    def _set_message(self, message: str) -> None:
        self._message = message
        self._emit_update()

    def _emit_update(self) -> None:
        self.update_ready.emit(self.current_update())
