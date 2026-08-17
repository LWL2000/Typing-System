# Pure Gaze Typing Adaptive Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add default-on, session-only, guarded Huber-RLS gaze correction to the pure gaze typing application without changing or overwriting the base EyeTrax calibration.

**Architecture:** A new adaptive_gaze.py module owns stable-window quality gates, affine inspection, RLS updates, rollback, and snapshots. TypingController feeds pre-session-affine coordinates into that module and supplies a real key rectangle only after dwell activation; settings and the startup UI expose a backward-compatible default-on switch.

**Tech Stack:** Python 3.10, NumPy, PyQt6, pytest, PyInstaller, UDP JSON.

## Global Constraints

- Modify only D:\CodeX\脑电软件设计\cap32_gaze_typing\.worktrees\implementation and its release output.
- Do not modify D:\claude\新协议2_实时自适应眼动融合版.
- Do not persist the session affine into Ridge models or calibration metadata.
- Preserve keyboard geometry, dwell behavior, triple-blink return, send-newline behavior, and the one-way capture UDP protocol.
- With adaptive correction disabled, coordinates must follow the existing center-drift-only path.
- Use D:\python3.10.11\python.exe for tests and builds.
- Implement every behavior test-first and commit each independently reviewable task.

---

### Task 1: Guarded Session Affine Core

**Files:**
- Create: src/pure_gaze_typing/adaptive_gaze.py
- Create: tests/test_adaptive_gaze.py

**Interfaces:**
- Consumes: timestamped pre-session-affine gaze coordinates and target rectangle (left, top, width, height).
- Produces: AdaptiveObservation, StableWindowStats, AffineConstraints, RlsConfig, AdaptiveDecision, AdaptiveSnapshot, AdaptiveGazeSession.
- AdaptiveGazeSession.observe(timestamp, *, valid, blink, x=None, y=None, quality=0.0) -> None records valid and invalid samples.
- AdaptiveGazeSession.apply(x, y) -> tuple[float, float] applies the latest reliable session affine.
- AdaptiveGazeSession.consider_anchor(target_id, target_rect) -> AdaptiveDecision gates and optionally commits one RLS update.
- AdaptiveGazeSession.reset(reason) -> AdaptiveDecision restores identity and clears the sample window.
- AdaptiveGazeSession.set_enabled(enabled) -> None switches between adaptive and identity output.

- [ ] **Step 1: Write identity, disabled, and stable-window failure tests**

~~~python
def test_disabled_session_stays_identity():
    session = AdaptiveGazeSession((1920, 1080), enabled=False)
    for index in range(12):
        session.observe(
            index * 0.05,
            valid=True,
            blink=False,
            x=300.0,
            y=220.0,
            quality=0.9,
        )
    assert session.apply(320.0, 240.0) == pytest.approx((320.0, 240.0))
    decision = session.consider_anchor("target_0", (200.0, 150.0, 300.0, 240.0))
    assert not decision.accepted
    assert decision.reason == "disabled"


def test_window_requires_eight_valid_samples():
    session = AdaptiveGazeSession((1920, 1080))
    for index in range(7):
        session.observe(
            index * 0.05,
            valid=True,
            blink=False,
            x=300.0,
            y=220.0,
            quality=0.9,
        )
    assert (
        session.consider_anchor("target_0", (200.0, 150.0, 300.0, 240.0)).reason
        == "insufficient_samples"
    )
~~~

- [ ] **Step 2: Run the new test file and confirm collection fails**

Run: D:\python3.10.11\python.exe -m pytest tests/test_adaptive_gaze.py -q

Expected: FAIL because pure_gaze_typing.adaptive_gaze does not exist.

- [ ] **Step 3: Implement observations, pruning, stable-window statistics, and identity apply**

~~~python
@dataclass(frozen=True)
class AdaptiveObservation:
    timestamp: float
    valid: bool
    blink: bool
    x: float | None
    y: float | None
    quality: float


class AdaptiveGazeSession:
    def observe(self, timestamp, *, valid, blink, x=None, y=None, quality=0.0):
        self._observations.append(AdaptiveObservation(...))
        self._prune(float(timestamp))

    def apply(self, x, y):
        if not self.enabled:
            return float(x), float(y)
        corrected = apply_affine([[float(x), float(y)]], self._matrix, self.origin)[0]
        return float(corrected[0]), float(corrected[1])
~~~

Defaults: 0.55-second window, 8 valid samples, maximum invalid ratio 0.25, maximum radial MAD 38 px.

- [ ] **Step 4: Add tests for invalid ratio, blink exclusion, high MAD, non-finite coordinates, and time-window pruning**

The expected rejection reasons are invalid_ratio, high_mad, invalid_coordinate, and insufficient_samples. Invalid or blink samples remain in the deque so invalid ratio is measurable.

- [ ] **Step 5: Run stable-window tests**

Run: D:\python3.10.11\python.exe -m pytest tests/test_adaptive_gaze.py -q

Expected: all window and identity tests PASS.

- [ ] **Step 6: Write failing trusted-core and accepted-update tests**

~~~python
def test_anchor_requires_median_inside_inner_core():
    session = stable_session_at((205.0, 160.0))
    decision = session.consider_anchor("target_0", (200.0, 150.0, 300.0, 240.0))
    assert not decision.accepted
    assert decision.reason == "outside_target_core"


def test_accepted_update_reduces_residual():
    session = stable_session_at((300.0, 220.0))
    target = np.asarray((350.0, 270.0))
    before = np.linalg.norm(np.asarray(session.apply(300.0, 220.0)) - target)
    decision = session.consider_anchor("target_0", (200.0, 150.0, 300.0, 240.0))
    after = np.linalg.norm(np.asarray(session.apply(300.0, 220.0)) - target)
    assert decision.accepted
    assert decision.matrix_version == 1
    assert after < before
~~~

- [ ] **Step 7: Implement Huber-RLS and affine constraints**

~~~python
@dataclass(frozen=True)
class RlsConfig:
    forgetting_factor: float = 0.995
    huber_delta_px: float = 90.0
    max_step_px: float = 12.0
    initial_covariance: float = 60.0


@dataclass(frozen=True)
class AffineConstraints:
    min_scale: float = 0.80
    max_scale: float = 1.25
    max_rotation_deg: float = 10.0
    max_shear: float = 0.12
    max_translation_ratio: float = 0.18
    max_condition: float = 20.0
~~~

consider_anchor must check the inner 65 percent target core, residual no greater than 18 percent of screen diagonal, Huber weighting, 12 px clipped correction, candidate matrix constraints, and non-worsening residual.

- [ ] **Step 8: Add unsafe rejection, rollback, reset, and history tests**

~~~python
def test_three_unsafe_rejections_roll_back_and_suspend():
    session = session_with_one_accepted_update()
    for _index in range(3):
        refill_window(session, (300.0, 220.0))
        decision = session.consider_anchor(
            "target_far",
            (1700.0, 850.0, 180.0, 160.0),
        )
        assert not decision.accepted
    snapshot = session.snapshot()
    assert snapshot.rollback_count == 1
    assert snapshot.suspended


def test_reset_restores_identity_and_learning():
    session = session_with_one_accepted_update()
    decision = session.reset("new_session")
    assert decision.reason == "new_session"
    assert session.apply(300.0, 220.0) == pytest.approx((300.0, 220.0))
    assert not session.snapshot().suspended
~~~

Only residual_too_large, invalid_affine, and residual_worsened count as unsafe. Retain at most eight reliable matrices. Three consecutive unsafe rejections roll back one matrix and suspend updates.

- [ ] **Step 9: Run tests and commit**

Run: D:\python3.10.11\python.exe -m pytest tests/test_adaptive_gaze.py -q

Expected: all adaptive core tests PASS.

Commit:

~~~text
git add src/pure_gaze_typing/adaptive_gaze.py tests/test_adaptive_gaze.py
git commit -m "feat: add guarded adaptive gaze session"
~~~

---

### Task 2: Backward-Compatible Setting and Startup Toggle

**Files:**
- Modify: src/pure_gaze_typing/settings.py
- Modify: src/pure_gaze_typing/typing_window.py
- Modify: tests/test_settings.py
- Modify: tests/test_typing_window.py

**Interfaces:**
- Consumes: old three-field settings JSON and startup controls.
- Produces: TypingSettings.adaptive_correction_enabled: bool = True and StartupWindow.adaptive_checkbox.

- [ ] **Step 1: Write failing settings migration tests**

~~~python
def test_old_settings_enable_adaptive_by_default(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "show_gaze_point": False,
        "dwell_seconds": 1.2,
        "validate_calibration": True,
    }), encoding="utf-8")
    assert load_settings(path).adaptive_correction_enabled is True


def test_adaptive_setting_requires_boolean():
    with pytest.raises(ValueError):
        TypingSettings(adaptive_correction_enabled="yes")
~~~

- [ ] **Step 2: Run focused settings tests and confirm failure**

Run: D:\python3.10.11\python.exe -m pytest tests/test_settings.py -q

Expected: FAIL because the fourth field is missing and old JSON is rejected.

- [ ] **Step 3: Add the field and explicit old-schema migration**

~~~python
@dataclass(frozen=True)
class TypingSettings:
    show_gaze_point: bool = True
    dwell_seconds: float = 1.0
    validate_calibration: bool = False
    adaptive_correction_enabled: bool = True


old_fields = {"show_gaze_point", "dwell_seconds", "validate_calibration"}
new_fields = old_fields | {"adaptive_correction_enabled"}
if set(payload) == old_fields:
    payload["adaptive_correction_enabled"] = True
elif set(payload) != new_fields:
    raise ValueError("settings fields do not match the current schema")
~~~

- [ ] **Step 4: Write the failing startup checkbox test**

~~~python
def test_startup_exposes_default_on_adaptive_checkbox(qtbot, tmp_path):
    window = StartupWindow(
        FakeTypingController(),
        TypingSettings(),
        AppPaths.for_root(tmp_path),
    )
    qtbot.addWidget(window)
    assert window.adaptive_checkbox.text() == "实时自适应校正（推荐）"
    assert window.adaptive_checkbox.isChecked()
~~~

- [ ] **Step 5: Add the checkbox and include it in StartupWindow._start**

~~~python
self.adaptive_checkbox = QCheckBox("实时自适应校正（推荐）")
self.adaptive_checkbox.setChecked(settings.adaptive_correction_enabled)
form.addRow("自适应", self.adaptive_checkbox)

settings = TypingSettings(
    self.gaze_checkbox.isChecked(),
    self.dwell_spin.value(),
    self.validation_checkbox.isChecked(),
    self.adaptive_checkbox.isChecked(),
)
~~~

- [ ] **Step 6: Run focused tests and commit**

Run: D:\python3.10.11\python.exe -m pytest tests/test_settings.py tests/test_typing_window.py -q

Expected: all focused tests PASS.

Commit:

~~~text
git add src/pure_gaze_typing/settings.py src/pure_gaze_typing/typing_window.py tests/test_settings.py tests/test_typing_window.py
git commit -m "feat: expose adaptive gaze setting"
~~~

---

### Task 3: Typing Controller Integration and Event Logging

**Files:**
- Modify: src/pure_gaze_typing/typing_controller.py
- Modify: tests/test_typing_window.py
- Modify: tests/test_session_log.py

**Interfaces:**
- Consumes: AdaptiveGazeSession and adaptive_correction_enabled.
- Produces: corrected display/selection coordinates and adaptive event rows in existing events.csv.

- [ ] **Step 1: Write disabled-path and accepted-anchor failure tests**

~~~python
def test_disabled_adaptation_preserves_center_only_coordinates(tmp_path):
    controller = ready_controller(
        tmp_path,
        TypingSettings(adaptive_correction_enabled=False),
    )
    controller.start_session(skip_prepare=True)
    update = controller.process_message(valid_sample(1.0, 300.0, 220.0), 1.0)
    assert update.gaze_point == pytest.approx(controller.drift.apply(300.0, 220.0))
    assert controller.adaptive.snapshot().matrix_version == 0


def test_core_dwell_updates_affine_and_records_event(tmp_path):
    controller = ready_recording_controller(tmp_path)
    for index in range(24):
        timestamp = 1.0 + index * 0.05
        controller.process_message(valid_sample(timestamp, 250.0, 225.0), timestamp)
    assert controller.adaptive.snapshot().matrix_version == 1
    assert "adaptive_update_accepted" in read_event_names(
        controller.recorder.events_path
    )
~~~

- [ ] **Step 2: Run controller tests and confirm failure**

Run: D:\python3.10.11\python.exe -m pytest tests/test_typing_window.py -q

Expected: FAIL because TypingController.adaptive does not exist.

- [ ] **Step 3: Instantiate, feed, and apply the session adapter**

~~~python
self.adaptive = AdaptiveGazeSession(
    (self.layout.screen_width, self.layout.screen_height),
    enabled=settings.adaptive_correction_enabled,
)

base_point = self.drift.apply(message.screen_x, message.screen_y)
self.adaptive.observe(
    current,
    valid=message.valid,
    blink=message.blink,
    x=base_point[0],
    y=base_point[1],
    quality=message.quality,
)
corrected = self.adaptive.apply(*base_point)
~~~

During invalid or blink frames call observe with x=None and y=None. Do not collect during the startup center preparation stage.

- [ ] **Step 4: Evaluate one anchor before page activation**

Resolve the triggered target position against layout.targets_for(len(engine.targets())). Pass that rectangle to consider_anchor before _activate changes the page. Record one decision event and clear the window after every successful action.

~~~python
rect = self.layout.targets_for(len(self.engine.targets()))[position]
decision = self.adaptive.consider_anchor(
    dwell_update.triggered_target_id,
    (rect.left, rect.top, rect.width, rect.height),
)
self._record_adaptive_decision(decision)
self._activate(triggered_logical)
self.adaptive.clear_window()
~~~

- [ ] **Step 5: Add a single decision-to-event mapping**

~~~python
def _record_adaptive_decision(self, decision):
    if decision.rollback_performed:
        event = "adaptive_rollback"
    elif decision.accepted:
        event = "adaptive_update_accepted"
    else:
        event = "adaptive_update_rejected"
    self._record_event(event, asdict(decision))
~~~

Do not log per-frame rejections. Log only an anchor decision, reset, rollback, or user disable.

- [ ] **Step 6: Write calibration reset, submenu rectangle, gaze visibility, and lifecycle tests**

~~~python
def test_calibration_change_resets_adaptive_matrix(tmp_path):
    controller = controller_with_accepted_update(tmp_path)
    controller.process_message(
        Heartbeat(2.0, True, True, "cal-2", LAYOUT_VERSION, 30.0),
        2.0,
    )
    assert controller.adaptive.snapshot().matrix_version == 0
    assert controller.adaptive.apply(300.0, 220.0) == pytest.approx((300.0, 220.0))


def test_submenu_anchor_uses_submenu_rectangle(tmp_path):
    controller = ready_recording_controller(tmp_path)
    controller.engine.activate("main_group_0")
    feed_stable_dwell(controller, controller.layout.submenu_targets[0].center)
    assert controller.last_triggered_target == "letter_A"
    assert controller.adaptive.last_target_id == "target_0"
~~~

Also assert visible and hidden gaze configurations produce the same matrix version and selected key.

- [ ] **Step 7: Make lifecycle and setting changes deterministic**

- update_settings calls adaptive.set_enabled.
- start_session creates a fresh adaptive session.
- a calibration ID change calls reset("calibration_changed").
- triple-blink page return and every successful selection clear the stable window.
- reset events are written only while a recorder exists.
- disabling adaptation clears the window and applies identity session correction immediately.

- [ ] **Step 8: Run controller/log tests and commit**

Run: D:\python3.10.11\python.exe -m pytest tests/test_typing_window.py tests/test_session_log.py tests/test_adaptive_gaze.py -q

Expected: all focused tests PASS.

Commit:

~~~text
git add src/pure_gaze_typing/typing_controller.py tests/test_typing_window.py tests/test_session_log.py
git commit -m "feat: adapt gaze from trusted typing selections"
~~~

---

### Task 4: Documentation and Full Regression

**Files:**
- Modify: README.md
- Test: all tests under tests/

**Interfaces:**
- Consumes: completed runtime behavior.
- Produces: user-facing explanation of the switch and session-only boundary.

- [ ] **Step 1: Document the default-on behavior**

Add this meaning to README.md:

~~~text
“实时自适应校正（推荐）”默认开启。它只在本次打字会话中使用稳定、成功触发且位于按键核心区的注视样本进行小步修正；异常更新会拒绝或回滚。该功能不会覆盖眼动采集程序保存的基础校准模型。关闭后使用原有中心点修正路径。
~~~

- [ ] **Step 2: Run the complete suite**

Run: D:\python3.10.11\python.exe -m pytest -q

Expected: all tests PASS without collection errors.

- [ ] **Step 3: Run syntax and patch checks**

Run:

~~~text
D:\python3.10.11\python.exe -m compileall -q src tests
git diff --check
~~~

Expected: both commands exit 0.

- [ ] **Step 4: Commit documentation**

~~~text
git add README.md
git commit -m "docs: explain adaptive gaze correction"
~~~

---

### Task 5: Package, Release, and Real Application Verification

**Files:**
- Build: packaging/build.ps1
- Replace: D:\CodeX\脑电软件设计\cap32_gaze_typing\release\纯眼动打字系统
- Preserve: D:\CodeX\脑电软件设计\cap32_gaze_typing\release\纯眼动打字系统数据

**Interfaces:**
- Consumes: tested source tree.
- Produces: locally executable applications and pushed Git history.

- [ ] **Step 1: Build both applications**

Run: powershell -ExecutionPolicy Bypass -File packaging\build.ps1

Expected: PyInstaller succeeds, release validation passes, and both self-tests return 0.

- [ ] **Step 2: Replace only known release artifacts**

Confirm no old release process is running. Copy the built 眼动采集校准 and 纯眼动打字器 directory contents plus 使用说明.md into the visible release directory. Do not delete or overwrite the sibling 纯眼动打字系统数据 directory.

- [ ] **Step 3: Verify hashes and self-tests**

Compare SHA-256 for both source-built and visible-release executables. Run both with --self-test and require exit code 0.

- [ ] **Step 4: Run a real startup interaction check**

Launch visible 纯眼动打字器.exe, verify the startup page contains a checked 实时自适应校正（推荐） option, close the window, and confirm process exit. With capture and camera available, confirm live UDP makes the start button enabled and the interface remains responsive.

- [ ] **Step 5: Final verification and push**

Run:

~~~text
D:\python3.10.11\python.exe -m pytest -q
git status --short
git push origin feature/pure-gaze-typing
~~~

Expected: full suite passes, worktree is clean, and remote contains every adaptive correction commit.

