# Pure Gaze Typing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and package an independent Windows two-program gaze typing system with fast EyeTrax calibration, optional gaze-point display, configurable dwell selection, gaze/blink navigation, and local experiment records.

**Architecture:** A shared `pure_gaze_typing` package provides immutable settings, normalized screen geometry, a versioned UDP protocol, calibration storage, selection state machines, typing state, and session logging. `眼动采集校准.exe` owns the camera and EyeTrax estimator and publishes gaze samples; `纯眼动打字器.exe` receives those samples and renders a static PyQt6 keyboard. Both executables are built from this repository and have no runtime dependency on either reference project.

**Tech Stack:** Python 3.10, PyQt6 6.x, OpenCV 4.x, EyeTrax 0.4.0, MediaPipe 0.10.x, NumPy 1.26.x, scikit-learn 1.x, pytest, pytest-qt, PyInstaller 6.x, PowerShell.

## Global Constraints

- Keep every new source, test, build script, and document under `D:\CodeX\脑电软件设计\cap32_gaze_typing`.
- Treat `D:\claude\新协议版` and `D:\CodeX\脑电软件设计\cap32_brain_typing` as read-only references; do not import from them at runtime.
- Write user data only under `%APPDATA%\PureGazeTyping`.
- Use UDP loopback `127.0.0.1:9101`; no internet or remote text transmission.
- Use one shared layout implementation for calibration and typing.
- Default dwell time is `1.0` second; valid range is `0.5-3.0` seconds with `0.1` second increments.
- First-run gaze-point display defaults to enabled and then remembers the last setting.
- Default calibration is one fast round of about `8-10` seconds; six-region precision validation is optional and defaults to disabled.
- Invalid gaze, camera loss, or a connection timeout must never trigger a target.
- Use test-driven development and commit after every task.

---

## Planned File Structure

```text
cap32_gaze_typing/
├── .gitignore
├── pyproject.toml
├── README.md
├── resources/
│   └── .gitkeep
├── src/pure_gaze_typing/
│   ├── __init__.py
│   ├── paths.py
│   ├── app_logging.py
│   ├── settings.py
│   ├── layout.py
│   ├── protocol.py
│   ├── calibration.py
│   ├── eyetrax_runtime.py
│   ├── gaze_selection.py
│   ├── typing_engine.py
│   ├── session_log.py
│   ├── capture_worker.py
│   ├── capture_window.py
│   ├── capture_app.py
│   ├── typing_controller.py
│   ├── typing_window.py
│   └── typing_app.py
├── tests/
│   ├── conftest.py
│   ├── test_settings.py
│   ├── test_layout.py
│   ├── test_protocol.py
│   ├── test_calibration.py
│   ├── test_eyetrax_runtime.py
│   ├── test_gaze_selection.py
│   ├── test_typing_engine.py
│   ├── test_session_log.py
│   ├── test_capture_window.py
│   └── test_typing_window.py
└── packaging/
    ├── build.ps1
    ├── pure_gaze_capture.spec
    ├── pure_gaze_typing.spec
    ├── capture_launcher.py
    ├── typing_launcher.py
    └── smoke_test.py
```

`paths.py` owns all filesystem locations. `layout.py` is the sole geometry authority. `protocol.py` contains transport DTOs but no UI. `calibration.py`, `gaze_selection.py`, and `typing_engine.py` are pure state machines. Hardware and Qt code remain at the edges so core behavior can be tested without a camera or display.

---

### Task 1: Project Foundation, Paths, and Settings

**Files:**
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/pure_gaze_typing/__init__.py`
- Create: `src/pure_gaze_typing/paths.py`
- Create: `src/pure_gaze_typing/app_logging.py`
- Create: `src/pure_gaze_typing/settings.py`
- Create: `tests/conftest.py`
- Create: `tests/test_settings.py`
- Create: `resources/.gitkeep`

**Interfaces:**
- Produces: `AppPaths.for_root(root: Path) -> AppPaths`
- Produces: `AppPaths.default() -> AppPaths`
- Produces: `TypingSettings(show_gaze_point: bool = True, dwell_seconds: float = 1.0, validate_calibration: bool = False)`
- Produces: `load_settings(path: Path) -> TypingSettings`
- Produces: `save_settings(path: Path, settings: TypingSettings) -> None`
- Produces: `configure_logging(paths: AppPaths, app_name: str) -> Path`

- [ ] **Step 1: Add package metadata and test dependencies**

Create `pyproject.toml` with Python `>=3.10,<3.11`, runtime dependencies `numpy>=1.24,<2`, `scikit-learn>=1.3,<2`, `opencv-python>=4.8,<5`, `PyQt6>=6.5,<7`, `eyetrax==0.4.0`, `mediapipe>=0.10,<0.11`, and optional test dependencies `pytest>=8,<10`, `pytest-qt>=4.4,<5`, and `PyInstaller>=6,<7`. Configure setuptools for `src/` and pytest for `tests/`. Ignore Python caches plus repository-local `build/`, `dist/`, `release/`, and the staged `resources/face_landmarker.task`.

- [ ] **Step 2: Write failing settings tests**

```python
def test_default_settings_match_product_defaults(tmp_path):
    settings = load_settings(tmp_path / "missing.json")
    assert settings == TypingSettings(True, 1.0, False)


def test_settings_round_trip_and_validation(tmp_path):
    path = tmp_path / "settings.json"
    save_settings(path, TypingSettings(False, 1.4, True))
    assert load_settings(path) == TypingSettings(False, 1.4, True)
    with pytest.raises(ValueError, match="0.5"):
        TypingSettings(True, 0.4, False)


def test_logging_creates_utf8_file_under_app_data(tmp_path):
    path = configure_logging(AppPaths.for_root(tmp_path), "typing")
    logging.getLogger("pure_gaze_typing").info("眼动已连接")
    assert path.parent == tmp_path / "logs"
    assert "眼动已连接" in path.read_text(encoding="utf-8")
```

- [ ] **Step 3: Run the focused tests and confirm failure**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_settings.py -v`  
Expected: collection fails because `pure_gaze_typing.settings` does not exist.

- [ ] **Step 4: Implement paths and settings**

Use frozen dataclasses. `AppPaths.default()` must resolve `%APPDATA%\PureGazeTyping`; `for_root()` must derive `calibration`, `logs`, `sessions`, and `settings.json`. Validate finite dwell values in `TypingSettings.__post_init__`, round to one decimal place, and save JSON atomically through a sibling `.tmp` file before `Path.replace()`.

```python
@dataclass(frozen=True)
class TypingSettings:
    show_gaze_point: bool = True
    dwell_seconds: float = 1.0
    validate_calibration: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.dwell_seconds) or not 0.5 <= self.dwell_seconds <= 3.0:
            raise ValueError("dwell_seconds must be between 0.5 and 3.0")
        object.__setattr__(self, "dwell_seconds", round(self.dwell_seconds, 1))
```

Malformed settings files must be renamed with a `.invalid.json` suffix and replaced with defaults; missing files return defaults without creating a file.

`configure_logging()` must create one timestamped UTF-8 log per process launch, attach both file and stderr handlers to the `pure_gaze_typing` logger, and avoid duplicate handlers when called twice in a self-test.

- [ ] **Step 5: Run tests**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_settings.py -v`  
Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add .gitignore pyproject.toml resources/.gitkeep src/pure_gaze_typing tests/conftest.py tests/test_settings.py
git commit -m "feat: add gaze typing project foundation"
```

---

### Task 2: Shared Calibration and Keyboard Geometry

**Files:**
- Create: `src/pure_gaze_typing/layout.py`
- Create: `tests/test_layout.py`

**Interfaces:**
- Produces: `PixelRect(left: float, top: float, width: float, height: float)` with `center`, `contains(x, y)`, `intersects(other)`, and `scaled(width, height)` helpers
- Produces: `LayoutSpec(version: str, screen_width: int, screen_height: int, top_bar: PixelRect, back_target: PixelRect, targets: tuple[PixelRect, ...])`
- Produces: `build_layout(width: int, height: int) -> LayoutSpec`
- Produces: `calibration_points(layout: LayoutSpec) -> tuple[tuple[str, tuple[float, float]], ...]`
- Produces: `hit_test(layout: LayoutSpec, x: float, y: float, include_back: bool) -> str | None`

- [ ] **Step 1: Write failing geometry tests**

```python
def test_layout_has_six_stable_targets_and_back_region():
    layout = build_layout(1920, 1080)
    assert layout.version == "gaze-grid-v1"
    assert len(layout.targets) == 6
    assert [rect.center for rect in layout.targets] == [
        (326.4, 356.4), (960.0, 356.4), (1593.6, 356.4),
        (326.4, 788.4), (960.0, 788.4), (1593.6, 788.4),
    ]
    assert all(not a.intersects(b) for i, a in enumerate(layout.targets) for b in layout.targets[i + 1:])


def test_calibration_and_hit_testing_use_the_same_centers():
    layout = build_layout(1280, 720)
    points = dict(calibration_points(layout))
    for index, rect in enumerate(layout.targets):
        x, y = points[f"target_{index}"]
        assert hit_test(layout, x, y, include_back=False) == f"target_{index}"
    assert hit_test(layout, *points["back"], include_back=True) == "back"
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_layout.py -v`  
Expected: import failure for `pure_gaze_typing.layout`.

- [ ] **Step 3: Implement normalized geometry once**

Use a top-left origin. Define target rectangles by normalized edges `(0.04, 0.20, 0.26, 0.26)`, `(0.37, 0.20, 0.26, 0.26)`, `(0.70, 0.20, 0.26, 0.26)` and the same columns at top `0.60`; their centers are therefore `0.17/0.50/0.83` by `0.33/0.73`. Define the top bar as `(0.02, 0.02, 0.96, 0.10)` and the fixed back target as `(0.02, 0.02, 0.14, 0.10)`.

`calibration_points()` must return center, back, then six target centers in stable order. `hit_test()` must return `None` in gaps rather than selecting the nearest target.

- [ ] **Step 4: Run layout tests**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_layout.py -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/pure_gaze_typing/layout.py tests/test_layout.py
git commit -m "feat: define shared gaze keyboard geometry"
```

---

### Task 3: Versioned UDP Protocol and Connection State

**Files:**
- Create: `src/pure_gaze_typing/protocol.py`
- Create: `tests/test_protocol.py`

**Interfaces:**
- Produces: `GazeSample(timestamp: float, valid: bool, face_detected: bool, blink: bool, quality: float, fps: float, calibration_id: str, layout_version: str, screen_x: float | None = None, screen_y: float | None = None, raw_x: float | None = None, raw_y: float | None = None)`
- Produces: `Heartbeat(timestamp: float, camera_ok: bool, calibration_ready: bool, calibration_id: str, layout_version: str, fps: float, error: str | None = None)`
- Produces: `encode_message(message: GazeSample | Heartbeat) -> bytes`
- Produces: `decode_message(payload: bytes) -> GazeSample | Heartbeat`
- Produces: `UdpPublisher(host: str = "127.0.0.1", port: int = 9101)`
- Produces: `UdpReceiver(host: str = "127.0.0.1", port: int = 9101)` with `poll()`, `last_received_at`, and `is_online(now: float, timeout: float = 2.0)`

- [ ] **Step 1: Write failing protocol tests**

```python
def test_gaze_sample_round_trip_rejects_nonfinite_coordinates():
    sample = GazeSample(10.0, True, True, False, 0.9, 29.8, "cal-1", "gaze-grid-v1", 640.0, 360.0)
    assert decode_message(encode_message(sample)) == sample
    with pytest.raises(ValueError, match="finite"):
        GazeSample(10.0, True, True, False, 0.9, 30.0, "cal-1", "gaze-grid-v1", math.nan, 1.0)


def test_receiver_online_state_expires_after_two_seconds():
    receiver = UdpReceiver.__new__(UdpReceiver)
    receiver.last_received_at = 5.0
    assert receiver.is_online(6.9)
    assert not receiver.is_online(7.01)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_protocol.py -v`  
Expected: import failure for `pure_gaze_typing.protocol`.

- [ ] **Step 3: Implement strict JSON messages**

Set `PROTOCOL_VERSION = 1`. Encode `message_type` as `"gaze"` or `"heartbeat"`; reject unknown fields, missing fields, unsupported protocol versions, non-finite numbers, coordinates on invalid samples, and datagrams over 64 KiB. Bind the receiver non-blocking and drain all available packets in `poll()` while discarding malformed packets through a supplied warning callback.

The publisher must explicitly bind no listening interface and must only accept loopback hosts. Both classes expose `close()` and context-manager methods.

- [ ] **Step 4: Run protocol tests**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_protocol.py -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/pure_gaze_typing/protocol.py tests/test_protocol.py
git commit -m "feat: add local gaze UDP protocol"
```

---

### Task 4: Fast Calibration State Machine and Model Store

**Files:**
- Create: `src/pure_gaze_typing/calibration.py`
- Create: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `LayoutSpec`, `calibration_points()` from Task 2
- Consumes: `AppPaths.calibration_dir` from Task 1
- Produces: `CalibrationEnvironment(screen_width: int, screen_height: int, scale_factor: float, camera_index: int, layout_version: str)`
- Produces: `CalibrationPoint(target_id: str, screen_x: float, screen_y: float, duration_seconds: float)`
- Produces: `CalibrationMetadata(calibration_id: str, created_at: str, environment: CalibrationEnvironment, feature_min: tuple[float, ...], feature_max: tuple[float, ...])`
- Produces: `CalibrationSession(points, center_seconds=1.0, target_seconds=0.8, min_valid_frames=12, max_point_seconds=3.0)`
- Produces: `ValidationResult(hit_count: int, total_count: int, failed_target_ids: tuple[str, ...])`
- Produces: `score_validation(hit_counts: Mapping[str, int], min_hits_per_target: int = 2) -> ValidationResult`
- Produces: `CalibrationStore.save(model, metadata)`, `load(environment)`, and `compatibility(environment)`

- [ ] **Step 1: Write failing calibration timing tests**

```python
def test_fast_session_advances_after_duration_and_minimum_frames():
    session = CalibrationSession((CalibrationPoint("target_0", 100.0, 200.0, 0.8),), min_valid_frames=12)
    for index in range(11):
        session.add_frame(index * 0.08, np.array([index], dtype=float), blink=False, face_detected=True)
    assert session.current_point_id == "target_0"
    session.add_frame(0.88, np.array([12.0]), blink=False, face_detected=True)
    assert session.complete


def test_validation_is_optional_and_reports_failed_regions():
    result = score_validation({"target_0": 4, "target_1": 0, "target_2": 3, "target_3": 5, "target_4": 2, "target_5": 4})
    assert result.hit_count == 5
    assert result.passed
    assert result.failed_target_ids == ("target_1",)
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_calibration.py -v`  
Expected: import failure for `pure_gaze_typing.calibration`.

- [ ] **Step 3: Implement deterministic calibration collection**

`CalibrationSession.add_frame()` must ignore frames with no face, blink frames, missing features, and non-finite feature arrays. A point advances only when both its configured duration has elapsed and `min_valid_frames` exist. If wall time reaches `max_point_seconds` without enough frames, expose `blocked_reason = "insufficient_valid_frames"` and stop the timer until `resume_current_point()`.

Build the training matrices by repeating each point's exact screen coordinate for every accepted feature vector. Compute feature bounds from the 2nd and 98th percentiles for runtime quality checks.

- [ ] **Step 4: Implement atomic model and metadata storage**

Save the EyeTrax model as `model.pkl` and metadata as `metadata.json` inside a new calibration-id directory, then atomically replace `current.json` with the selected id. `compatibility()` must compare exact width, height, camera index, layout version, and scale factor rounded to two decimals. A partial or corrupt directory must return an explicit incompatible result rather than crashing.

- [ ] **Step 5: Run calibration tests**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_calibration.py -v`  
Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/pure_gaze_typing/calibration.py tests/test_calibration.py
git commit -m "feat: add fast gaze calibration core"
```

---

### Task 5: EyeTrax Runtime Adapter and Gaze Quality

**Files:**
- Create: `src/pure_gaze_typing/eyetrax_runtime.py`
- Create: `tests/test_eyetrax_runtime.py`

**Interfaces:**
- Consumes: `CalibrationMetadata` and `CalibrationStore` from Task 4
- Produces: `FrameObservation(features: numpy.ndarray | None, face_detected: bool, blink: bool)`
- Produces: `GazeEstimate(valid: bool, face_detected: bool, blink: bool, quality: float, raw_x: float | None, raw_y: float | None, screen_x: float | None, screen_y: float | None)`
- Produces: `EyeTraxRuntime(face_model_path: Path, estimator_factory=GazeEstimator)` with `extract(frame)`, `train(samples, labels)`, `load_model(path)`, `estimate(observation)`, and `close()`
- Produces: `CenterDriftCorrector.collect(x, y)`, `finish(screen_center)`, and `apply(x, y)`

- [ ] **Step 1: Write failing adapter tests using a fake estimator**

```python
def test_blink_frame_is_invalid_and_never_predicted():
    estimator = FakeEstimator(features=np.array([1.0, 2.0]), blink=True, prediction=(50.0, 60.0))
    runtime = EyeTraxRuntime(Path("face.task"), estimator_factory=lambda **_: estimator)
    estimate = runtime.process_frame(np.zeros((4, 4, 3), dtype=np.uint8))
    assert not estimate.valid
    assert estimate.blink
    assert estimator.predict_calls == 0


def test_center_drift_is_median_based_and_capped():
    corrector = CenterDriftCorrector(1920, 1080, max_x_ratio=0.08, max_y_ratio=0.05)
    for point in [(1200.0, 700.0), (1204.0, 698.0), (5000.0, 5000.0)]:
        corrector.collect(*point)
    offset = corrector.finish((960.0, 540.0))
    assert offset == pytest.approx((-153.6, -54.0))
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_eyetrax_runtime.py -v`  
Expected: import failure for `pure_gaze_typing.eyetrax_runtime`.

- [ ] **Step 3: Implement the adapter around EyeTrax 0.4.0**

Instantiate `eyetrax.gaze.GazeEstimator(model_name="ridge", face_landmarker_model=str(face_model_path))`. Use `extract_features(frame)` for features and blink, `train(X, y)` for calibration, `predict(X)` for raw coordinates, and EyeTrax's Kalman-EMA smoother for screen coordinates. Inject estimator and smoother factories so tests never need MediaPipe or a camera.

Compute a feature-range score against calibration percentile bounds. Quality is `1.0` inside bounds, falls linearly to `0.0` at twice the bound distance, and invalidates samples below `0.25` or beyond twice the accepted range. Clamp valid screen coordinates to the current display.

Expose `process_frame(frame) -> GazeEstimate` as the composition of `extract(frame)` and `estimate(observation)` so the worker has one exception boundary per frame.

- [ ] **Step 4: Implement center drift correction**

Collect valid non-blink coordinates during the one-second prepare screen, use coordinate medians, and calculate `expected_center - measured_center`. Cap horizontal correction at `8%` of screen width and vertical correction at `5%` of screen height. Record both measured and applied offsets.

- [ ] **Step 5: Run adapter tests**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_eyetrax_runtime.py -v`  
Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/pure_gaze_typing/eyetrax_runtime.py tests/test_eyetrax_runtime.py
git commit -m "feat: add independent EyeTrax runtime adapter"
```

---

### Task 6: Dwell Selection and Triple-Blink Gesture

**Files:**
- Create: `src/pure_gaze_typing/gaze_selection.py`
- Create: `tests/test_gaze_selection.py`

**Interfaces:**
- Consumes: target ids from `layout.hit_test()`
- Produces: `DwellUpdate(target_id: str | None, progress: float, triggered_target_id: str | None, armed: bool)`
- Produces: `DwellSelector(dwell_seconds: float, leave_grace_seconds: float = 0.2, rearm_seconds: float = 0.3, target_dwell_seconds: Mapping[str, float] | None = None)`
- Produces: `BlinkUpdate(count: int, triple_blink: bool)`
- Produces: `TripleBlinkDetector(window_seconds: float = 2.5, min_closed_seconds: float = 0.05, max_closed_seconds: float = 0.6, cooldown_seconds: float = 1.0)`

- [ ] **Step 1: Write failing dwell and blink tests**

```python
def test_short_gaze_excursion_preserves_progress_but_long_excursion_resets():
    selector = DwellSelector(1.0)
    selector.update(0.0, "target_0", valid=True, blink=False)
    selector.update(0.6, "target_0", valid=True, blink=False)
    assert selector.update(0.7, None, valid=True, blink=False).progress == pytest.approx(0.6)
    assert selector.update(0.85, "target_0", valid=True, blink=False).progress == pytest.approx(0.6)
    assert selector.update(1.2, None, valid=True, blink=False).progress == 0.0


def test_three_complete_blinks_within_window_trigger_once():
    detector = TripleBlinkDetector()
    events = []
    for start in (0.0, 0.6, 1.2):
        detector.update(start, face_detected=True, blink=True)
        events.append(detector.update(start + 0.12, face_detected=True, blink=False))
    assert [event.count for event in events] == [1, 2, 3]
    assert events[-1].triple_blink
    assert not detector.update(1.4, face_detected=True, blink=False).triple_blink


def test_back_target_never_uses_less_than_1_2_seconds():
    selector = DwellSelector(0.5, target_dwell_seconds={"back": 1.2})
    selector.update(0.0, "back", valid=True, blink=False)
    assert selector.update(0.6, "back", valid=True, blink=False).triggered_target_id is None
    assert selector.update(1.2, "back", valid=True, blink=False).triggered_target_id == "back"
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_gaze_selection.py -v`  
Expected: import failure for `pure_gaze_typing.gaze_selection`.

- [ ] **Step 3: Implement dwell selection as timestamp-based state**

Accumulate valid elapsed time only while the same target is active. Blink frames freeze progress. Invalid/no-face frames start the same `0.2`-second grace timer and then reset. After a trigger, suppress that target until gaze has remained outside it for `0.3` seconds. `target_dwell_seconds` overrides the base duration for named targets; the controller passes `{"back": max(settings.dwell_seconds, 1.2)}`. `reset()` must clear all progress on connection loss or page change.

- [ ] **Step 4: Implement edge-based blink counting**

Start a candidate on the false-to-true blink edge and count it only on the true-to-false edge when closed duration is `0.05-0.6` seconds. Reset on face loss, window expiry, or a closure longer than `0.6` seconds. Emit exactly one triple event and enforce a one-second cooldown.

- [ ] **Step 5: Run selection tests**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_gaze_selection.py -v`  
Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add src/pure_gaze_typing/gaze_selection.py tests/test_gaze_selection.py
git commit -m "feat: add gaze dwell and blink gestures"
```

---

### Task 7: Typing State Machine and Session Records

**Files:**
- Create: `src/pure_gaze_typing/typing_engine.py`
- Create: `src/pure_gaze_typing/session_log.py`
- Create: `tests/test_typing_engine.py`
- Create: `tests/test_session_log.py`

**Interfaces:**
- Consumes: `TypingSettings`, calibration id, and layout version
- Produces: `PageKind.MAIN`, `PageKind.LETTERS`, `PageKind.FUNCTIONS`
- Produces: `TargetSpec(target_id: str, label: str, position: int, action: str)`
- Produces: `TypingEffect(action: str, text_changed: bool, page_changed: bool, sent_text: str | None, requires_clear_confirmation: bool)`
- Produces: `TypingEngine(current_line: str = "")`
- Produces: `TypingEngine.targets()`, `activate(target_id)`, `confirm_send()`, `return_to_main()`, `resume()`, `full_text()`
- Produces: `SessionRecorder.start(paths, settings, calibration_context)`, `record_gaze(sample, target_id, progress)`, `record_event(name, payload)`, and `finish(final_text)`

- [ ] **Step 1: Write failing typing-state tests**

```python
def test_group_selection_types_letter_and_fixed_back_returns_main():
    engine = TypingEngine()
    engine.activate("main_group_0")
    assert [target.label for target in engine.targets()][:5] == list("ABCDE")
    engine.activate("letter_A")
    assert engine.current_line == "A"
    engine.return_to_main()
    assert engine.page_kind is PageKind.MAIN


def test_clear_requires_two_consecutive_clear_activations():
    engine = TypingEngine(current_line="TEST")
    engine.activate("main_functions")
    first = engine.activate("function_clear")
    assert first.requires_clear_confirmation
    assert engine.current_line == "TEST"
    engine.activate("function_clear")
    assert engine.current_line == ""
```

- [ ] **Step 2: Write failing record tests**

```python
def test_session_recorder_creates_isolated_parseable_files(tmp_path):
    recorder = SessionRecorder.start(AppPaths.for_root(tmp_path), TypingSettings(), {"calibration_id": "cal-1"})
    recorder.record_event("selection", {"target_id": "letter_A"})
    recorder.finish("A")
    assert recorder.result_path.read_text(encoding="utf-8") == "A"
    rows = list(csv.DictReader(recorder.events_path.open(encoding="utf-8-sig")))
    assert rows[0]["event"] == "selection"
```

- [ ] **Step 3: Run the tests and confirm failure**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_typing_engine.py tests/test_session_log.py -v`  
Expected: import failures for the two new modules.

- [ ] **Step 4: Implement the six-target typing engine**

Main targets are five groups plus `功能`. The first four letter pages expose five letters plus `空格`; `U-Z` exposes six letters. The function page exposes `删除`, `空格`, `发送`, `清空`, `暂停`, and `返回`. Any activation other than `清空` cancels pending clear confirmation. Pause keeps the engine state but only allows a `继续` target at the same calibrated position as `暂停`.

`发送` returns the current complete text in `TypingEffect.sent_text` without mutating history. After successful persistence, the controller calls `confirm_send()` to append the current line to history and clear it. A failed persistence leaves input unchanged.

- [ ] **Step 5: Implement crash-tolerant session files**

Open gaze and event CSV files with `utf-8-sig`, write headers immediately, flush every row, and never overwrite an existing timestamped session directory. Use ISO timestamps plus a four-character random suffix. `finish()` atomically writes `result.txt` and `session.json`. A write failure must call an injected error callback and mark recording degraded without raising into the UI event loop.

- [ ] **Step 6: Run state and record tests**

Run: `D:\python3.10.11\python.exe -m pytest tests/test_typing_engine.py tests/test_session_log.py -v`  
Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/pure_gaze_typing/typing_engine.py src/pure_gaze_typing/session_log.py tests/test_typing_engine.py tests/test_session_log.py
git commit -m "feat: add gaze typing state and session records"
```

---

### Task 8: Acquisition, Calibration, and Streaming Program

**Files:**
- Create: `src/pure_gaze_typing/capture_worker.py`
- Create: `src/pure_gaze_typing/capture_window.py`
- Create: `src/pure_gaze_typing/capture_app.py`
- Create: `tests/test_capture_window.py`

**Interfaces:**
- Consumes: Tasks 1-5 protocol, layout, calibration, and EyeTrax APIs
- Produces: `CameraWorker(QObject)` signals `frame_ready(QImage)`, `observation_ready(FrameObservation)`, `failed(str)`, and `stopped()`
- Produces: `CaptureController.start_camera(index)`, `start_calibration(validate)`, `retry_current_point()`, `save_anyway()`, `start_streaming()`, and `stop()`
- Produces: `CaptureWindow(controller, paths)`
- Produces: `capture_app.main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing Qt state tests**

```python
def test_validation_is_off_by_default_and_calibration_button_tracks_camera(qtbot, tmp_path):
    controller = FakeCaptureController()
    window = CaptureWindow(controller, AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    assert not window.validation_checkbox.isChecked()
    assert not window.calibrate_button.isEnabled()
    controller.camera_state_changed.emit(True, "摄像头 0 已连接")
    assert window.calibrate_button.isEnabled()


def test_calibration_scene_uses_shared_layout_centers(qtbot, tmp_path):
    window = CaptureWindow(FakeCaptureController(), AppPaths.for_root(tmp_path))
    window.resize(1280, 720)
    window.show_calibration_point("target_3")
    assert window.highlight_center() == build_layout(1280, 720).targets[3].center
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `$env:QT_QPA_PLATFORM='offscreen'; D:\python3.10.11\python.exe -m pytest tests/test_capture_window.py -v`  
Expected: import failure for `pure_gaze_typing.capture_window`.

- [ ] **Step 3: Implement camera ownership in a worker thread**

Open `cv2.VideoCapture(camera_index)` only inside `CameraWorker`; read and process frames on its QThread. Never touch Qt widgets from the worker. Stop by setting an event, releasing the camera in `finally`, closing EyeTrax once, and emitting `stopped`. Convert only preview frames to `QImage`; pass numeric observations separately.

- [ ] **Step 4: Implement the capture controller**

Feed observations to `CalibrationSession` while calibrating. Train and save only after all required points pass. If validation was selected, run six target checks, show `5/6` result, and expose only “重校异常区域” and “仍然保存” after a failed validation.

After calibration, load the saved model and publish `GazeSample` for every processed frame plus one `Heartbeat` per second. On frame-read failure publish an invalid sample and heartbeat error. Continue retrying without reusing stale coordinates.

- [ ] **Step 5: Implement the PyQt6 window and CLI self-test**

The normal window includes camera selector, preview, calibration compatibility text, validation checkbox, `快速校准`, `开始输出`, `停止输出`, and `重新校准`. Calibration switches to full screen and draws the shared top bar, back region, six targets, and one highlighted point at a time.

`capture_app.main(["--self-test"])` must construct the app offscreen, validate packaged resources and protocol serialization, then exit `0` without opening a camera.

- [ ] **Step 6: Run acquisition tests and full unit suite**

Run: `$env:QT_QPA_PLATFORM='offscreen'; D:\python3.10.11\python.exe -m pytest tests/test_capture_window.py -v`  
Run: `D:\python3.10.11\python.exe -m pytest -q`  
Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/pure_gaze_typing/capture_worker.py src/pure_gaze_typing/capture_window.py src/pure_gaze_typing/capture_app.py tests/test_capture_window.py
git commit -m "feat: add gaze capture and calibration program"
```

---

### Task 9: Typing Controller, Startup Settings, and Full-Screen Keyboard

**Files:**
- Create: `src/pure_gaze_typing/typing_controller.py`
- Create: `src/pure_gaze_typing/typing_window.py`
- Create: `src/pure_gaze_typing/typing_app.py`
- Create: `tests/test_typing_window.py`

**Interfaces:**
- Consumes: all shared APIs from Tasks 1-7
- Produces: `TypingController.process_message(message, now) -> ControllerUpdate`
- Produces: `TypingController.tick(now) -> ControllerUpdate`
- Produces: `ConnectionStatus(online: bool, calibration_compatible: bool, message: str)`
- Produces: `StartupWindow(controller, settings, paths)`
- Produces: `TypingWindow(controller, settings, layout)`
- Produces: `typing_app.main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing controller and UI tests**

```python
def test_hidden_gaze_point_does_not_change_selection(qtbot, tmp_path):
    visible = make_controller(tmp_path / "visible", show_gaze_point=True)
    hidden = make_controller(tmp_path / "hidden", show_gaze_point=False)
    for timestamp in (0.0, 0.5, 1.0):
        sample = valid_sample(timestamp, x=326.4, y=356.4)
        visible.process_message(sample, timestamp)
        hidden.process_message(sample, timestamp)
    assert visible.engine.page_kind == hidden.engine.page_kind
    assert visible.last_triggered_target == hidden.last_triggered_target


def test_start_button_requires_online_compatible_calibration(qtbot, tmp_path):
    controller = FakeTypingController()
    window = StartupWindow(controller, TypingSettings(), AppPaths.for_root(tmp_path))
    qtbot.addWidget(window)
    assert not window.start_button.isEnabled()
    controller.status_changed.emit(ConnectionStatus(True, True, "眼动已连接"))
    assert window.start_button.isEnabled()
```

- [ ] **Step 2: Run the tests and confirm failure**

Run: `$env:QT_QPA_PLATFORM='offscreen'; D:\python3.10.11\python.exe -m pytest tests/test_typing_window.py -v`  
Expected: import failures for the typing controller/window modules.

- [ ] **Step 3: Implement the pure controller loop**

On every gaze sample, verify calibration id and layout version, apply center drift correction, hit-test current targets plus the fixed back region, update blink and dwell state, dispatch at most one `TypingEngine` action, and record both the gaze row and any resulting event. Triple blink resets dwell and returns to main only outside the main page.

On heartbeat/sample timeout over `2.0` seconds, clear dwell and blink state, pause selection, and report disconnected. Resume only after at least five consecutive compatible valid samples. Use a 30 Hz Qt timer for receiver polling and repainting; do not make target selection depend on paint frequency.

If UDP binding fails, keep the settings window open, show `端口 9101 无法使用` with the operating-system error, and leave the start button disabled.

- [ ] **Step 4: Implement startup and prepare screens**

The startup window shows acquisition status, calibration compatibility, a `显示眼动位置` checkbox, a `0.5-3.0` second spin box with `0.1` increments, and `开始实验`/`退出` commands. Save settings when starting. Disable start until connection and calibration are both valid. Both executable entry points call `configure_logging()` before creating controllers and log uncaught exceptions through a Qt-safe exception hook.

After start, show a one-second center target, collect drift samples, then open the keyboard. If too few valid center samples exist, offer retry or continue with zero drift; do not silently invent an offset.

- [ ] **Step 5: Implement the full-screen keyboard**

Render target widgets from `build_layout()` with fixed dimensions. Draw the live gaze dot only when `show_gaze_point` is true. Always draw dwell progress, selection feedback, current text, connection warning, blink count, and the fixed back control where applicable. The pause state dims other targets and exposes only `继续` at the calibrated pause position.

Closing full screen finishes the session and returns to startup. Sending must call `SessionRecorder.finish()` first; only then show success and clear committed input.

- [ ] **Step 6: Add typing CLI self-test and run all tests**

`typing_app.main(["--self-test"])` must create settings, protocol, layout, engine, dwell selector, logger in a temporary directory, and offscreen windows, then exit `0` without binding the production UDP port.

Run: `$env:QT_QPA_PLATFORM='offscreen'; D:\python3.10.11\python.exe -m pytest tests/test_typing_window.py -v`  
Run: `D:\python3.10.11\python.exe -m pytest -q`  
Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add src/pure_gaze_typing/typing_controller.py src/pure_gaze_typing/typing_window.py src/pure_gaze_typing/typing_app.py tests/test_typing_window.py
git commit -m "feat: add pure gaze typing application"
```

---

### Task 10: Windows Packaging, Offline Resources, and End-to-End Verification

**Files:**
- Create: `packaging/capture_launcher.py`
- Create: `packaging/typing_launcher.py`
- Create: `packaging/pure_gaze_capture.spec`
- Create: `packaging/pure_gaze_typing.spec`
- Create: `packaging/build.ps1`
- Create: `packaging/smoke_test.py`
- Create: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `capture_app.main()` and `typing_app.main()`
- Produces: `release/纯眼动打字系统/眼动采集校准/眼动采集校准.exe`
- Produces: `release/纯眼动打字系统/纯眼动打字器/纯眼动打字器.exe`

- [ ] **Step 1: Write packaging smoke checks**

`packaging/smoke_test.py` must support both commands:

```powershell
D:\python3.10.11\python.exe packaging/smoke_test.py --release release/纯眼动打字系统
D:\python3.10.11\python.exe packaging/smoke_test.py --exe release/纯眼动打字系统/纯眼动打字器/纯眼动打字器.exe
```

The release check must require both executables, both `_internal` directories, `使用说明.md`, and a packaged `face_landmarker.task` under the acquisition bundle. The executable check must run the selected executable with `--self-test`, enforce a 30-second timeout, and require exit code `0`.

- [ ] **Step 2: Run smoke checks against the absent release and confirm failure**

Run: `D:\python3.10.11\python.exe packaging/smoke_test.py --release release/纯眼动打字系统`  
Expected: failure listing the missing release paths.

- [ ] **Step 3: Implement launchers and PyInstaller specs**

Both launchers prepend the bundled application root to `sys.path` and call the matching `main()`. The capture spec must collect `cv2`, `numpy`, `sklearn`, `eyetrax`, `mediapipe`, and PyQt6 packages plus the FaceLandmarker task. The typing spec must collect PyQt6 and package modules but exclude OpenCV, MediaPipe, and EyeTrax to keep the typing bundle smaller.

Resolve resources through `sys._MEIPASS` when frozen and the repository `resources/` directory when running from source.

- [ ] **Step 4: Implement guarded PowerShell build**

`packaging/build.ps1` accepts `-PythonPath` defaulting to `D:\python3.10.11\python.exe` and `-FaceModelPath` defaulting to `C:\Users\pc\.cache\eyetrax\mediapipe\face_landmarker.task`. It must:

1. Resolve every build, dist, staging, and release target and verify each is below the new repository root before removal.
2. Copy the model into `resources/face_landmarker.task` only for the build and verify its size is nonzero.
3. Build both onedir bundles with separate PyInstaller work/dist paths.
4. Assemble `release\纯眼动打字系统` with the two program directories and `使用说明.md`.
5. Run release and executable smoke checks.
6. Remove only the staged model copy after a successful or failed build; never access either reference project.

- [ ] **Step 5: Write user documentation**

Document this exact order: start `眼动采集校准.exe`, select camera, perform quick calibration with optional validation, click `开始输出`, start `纯眼动打字器.exe`, choose gaze-point visibility and dwell time, then start typing. Include the fixed gaze return, three-blink shortcut, local data directory, connection recovery, and recalibration triggers.

- [ ] **Step 6: Run source verification**

Run: `D:\python3.10.11\python.exe -m pip install -e ".[test]"`  
Run: `$env:QT_QPA_PLATFORM='offscreen'; D:\python3.10.11\python.exe -m pytest -q`  
Run: `D:\python3.10.11\python.exe -m pure_gaze_typing.capture_app --self-test`  
Run: `D:\python3.10.11\python.exe -m pure_gaze_typing.typing_app --self-test`  
Expected: installation succeeds, all tests pass, and both self-tests exit `0`.

- [ ] **Step 7: Build and verify the release**

Run: `powershell -ExecutionPolicy Bypass -File packaging/build.ps1`  
Expected: PyInstaller builds both bundles and both smoke checks pass.

- [ ] **Step 8: Perform hardware acceptance**

Using the target display and camera, verify: fast calibration completes near 8-10 seconds; validation is off by default and works when enabled; visible and hidden gaze-point modes select the same targets; dwell modification changes activation time; the fixed return and three-blink return work; camera interruption pauses selection and recovery resumes; session files contain the expected gaze, blink, target, event, and final text fields.

- [ ] **Step 9: Confirm reference projects are untouched**

Run read-only status checks in both reference repositories and compare them with the statuses captured before implementation. The new project must contain no runtime path or import pointing at `D:\claude\新协议版` or `cap32_brain_typing`:

```powershell
rg -n "D:\\claude\\新协议版|cap32_brain_typing" src packaging pyproject.toml README.md
```

Expected: no matches.

- [ ] **Step 10: Commit**

```powershell
git add README.md pyproject.toml packaging
git commit -m "build: package pure gaze typing release"
```

---

## Final Verification Gate

Before reporting completion, run all of the following from `D:\CodeX\脑电软件设计\cap32_gaze_typing` and record the outputs in the handoff:

```powershell
git status --short
D:\python3.10.11\python.exe -m pytest -q
D:\python3.10.11\python.exe -m pure_gaze_typing.capture_app --self-test
D:\python3.10.11\python.exe -m pure_gaze_typing.typing_app --self-test
D:\python3.10.11\python.exe packaging/smoke_test.py --release release/纯眼动打字系统
```

Expected: clean worktree, all tests pass, both source self-tests exit `0`, and release layout verification passes. Hardware-dependent calibration accuracy and camera interruption recovery must be reported separately from automated verification rather than inferred from unit tests.
