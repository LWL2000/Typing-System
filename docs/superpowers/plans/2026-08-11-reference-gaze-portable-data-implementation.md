# Reference Gaze Calibration and Portable Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add portable release data storage and replace the capture application's lightweight calibration/output behavior with selectable fast and precise variants of the proven reference gaze pipeline.

**Architecture:** Keep camera ownership, model training, and UDP publication in the capture application. Move calibration mathematics into focused pure functions, let the Qt controller orchestrate provisional training/validation, and keep the typing application protocol unchanged. Packaged path selection derives a shared sibling data root from each executable and migrates legacy AppData once without overwriting files.

**Tech Stack:** Python 3.10, PyQt6, EyeTrax 0.4.0, NumPy, OpenCV, pytest, PyInstaller, loopback UDP.

## Global Constraints

- Do not modify `D:\claude\新协议2`.
- Keep the two existing executable applications and UDP protocol.
- Offer **快速校准（推荐）** and **精确校准** before calibration.
- Optional validation requires at least 5/6 hits before saving; failed validation retries only abnormal regions.
- Store packaged data at `release\纯眼动打字系统数据` and never delete it during package replacement.
- Preserve all pre-existing user changes in the worktree.

---

### Task 1: Portable Data Root and Legacy Migration

**Files:**
- Modify: `src/pure_gaze_typing/paths.py`
- Create: `tests/test_paths.py`

**Interfaces:**
- Produces: `AppPaths.default(executable: Path | None = None, frozen: bool | None = None) -> AppPaths`
- Produces: `AppPaths.migrate_legacy(legacy_root: Path | None = None) -> bool`

- [ ] **Step 1: Write failing path-resolution tests** covering source `%APPDATA%`, both packaged executable directories resolving to the same `release\纯眼动打字系统数据`, empty-root migration, and non-overwrite behavior.
- [ ] **Step 2: Run `python -m pytest tests/test_paths.py -q`** and confirm failures are caused by the missing frozen resolution and migration APIs.
- [ ] **Step 3: Implement frozen path derivation and recursive non-destructive copy** using `sys.executable`, `sys.frozen`, `Path`, and `shutil.copy2`.
- [ ] **Step 4: Run `python -m pytest tests/test_paths.py -q`** and confirm all path tests pass.
- [ ] **Step 5: Commit** with `feat: store packaged data beside release apps`.

### Task 2: Reference Calibration Mathematics

**Files:**
- Modify: `src/pure_gaze_typing/calibration.py`
- Modify: `tests/test_calibration.py`

**Interfaces:**
- Produces: `CalibrationMode` enum and `CalibrationProfile` dataclass.
- Produces: `calibration_profile(mode: CalibrationMode) -> CalibrationProfile`.
- Produces: `filter_stable_features(features, max_samples, min_samples) -> np.ndarray`.
- Produces: `balance_point_samples(groups) -> tuple[np.ndarray, np.ndarray]`.
- Produces: `fit_screen_affine(predictions, targets) -> np.ndarray` and `apply_screen_affine(coefficients, x, y) -> tuple[float, float]`.
- Produces: staged metadata fields for pipeline version, range threshold, affine coefficients, and validation result.

- [ ] **Step 1: Write failing tests** for fast/precise timing and pass order, MAD outlier rejection, point balancing, affine recovery, and metadata round-trip compatibility.
- [ ] **Step 2: Run `python -m pytest tests/test_calibration.py -q`** and confirm the new API tests fail for missing behavior.
- [ ] **Step 3: Implement the pure calibration APIs** using the formulas and defaults from the reference pipeline, with fast mode reducing passes and capture windows only.
- [ ] **Step 4: Run `python -m pytest tests/test_calibration.py -q`** and confirm the focused suite passes.
- [ ] **Step 5: Commit** with `feat: add reference gaze calibration profiles`.

### Task 3: Reference Runtime Guard and Smoothing

**Files:**
- Modify: `src/pure_gaze_typing/eyetrax_runtime.py`
- Create: `tests/test_eyetrax_runtime.py`

**Interfaces:**
- Consumes: affine and range metadata from Task 2.
- Produces: `EyeTraxRuntime.estimate(observation, timestamp: float | None = None) -> GazeEstimate` with affine correction, calibrated-range quality, hard rejection, and smoother reset.

- [ ] **Step 1: Write failing runtime tests** for `ema_alpha=0.35`, affine-before-smoothing, scaled feature-range quality, hard out-of-range rejection, and reset after 0.8 seconds invalid input.
- [ ] **Step 2: Run `python -m pytest tests/test_eyetrax_runtime.py -q`** and confirm each behavior fails against the current runtime.
- [ ] **Step 3: Implement runtime behavior** and persist/load the reference fields on the EyeTrax model while preserving UDP-facing `GazeEstimate` fields.
- [ ] **Step 4: Run `python -m pytest tests/test_eyetrax_runtime.py -q`** and confirm the focused suite passes.
- [ ] **Step 5: Commit** with `fix: match reference gaze output pipeline`.

### Task 4: Provisional Training and Quality Gate

**Files:**
- Modify: `src/pure_gaze_typing/capture_worker.py`
- Modify: `src/pure_gaze_typing/capture_window.py`
- Modify: `tests/test_capture_window.py`

**Interfaces:**
- Consumes: calibration profiles and runtime APIs from Tasks 2-3.
- Produces: provisional in-memory training that calls `CalibrationStore.save` only after optional validation passes.
- Produces: failed-region retry that retains successful region results and retrains with replacement samples.

- [ ] **Step 1: Write failing controller tests** proving validation-disabled models save immediately, validation-enabled models do not save before 5/6, and failed-region retry promotes only after a passing cumulative score.
- [ ] **Step 2: Run `python -m pytest tests/test_capture_window.py -q`** and confirm the current premature-save behavior fails the tests.
- [ ] **Step 3: Separate train from save in `CameraWorker`** and add explicit provisional-model promotion.
- [ ] **Step 4: Implement controller staging, affine-correction collection, quality validation, and failed-region recapture** while preserving camera preview and heartbeat behavior.
- [ ] **Step 5: Run `python -m pytest tests/test_capture_window.py -q`** and confirm the focused suite passes.
- [ ] **Step 6: Commit** with `fix: save calibration only after quality gate`.

### Task 5: Dual-Mode Calibration UI

**Files:**
- Modify: `src/pure_gaze_typing/capture_window.py`
- Modify: `tests/test_capture_window.py`

**Interfaces:**
- Consumes: `CaptureController.start_calibration(mode, validate)` from Task 4.
- Produces: a full-screen mode chooser with fast/precise labels, estimated durations, cancel behavior, progress text, and Escape cancellation.

- [ ] **Step 1: Write failing Qt tests** for opening the chooser, selecting each mode, displaying duration text, and Escape returning without starting calibration.
- [ ] **Step 2: Run `python -m pytest tests/test_capture_window.py -q`** and confirm failures reflect the absent chooser.
- [ ] **Step 3: Implement the chooser and calibration scene** with stable dimensions, calm neutral styling, target rings, and mode-specific progress.
- [ ] **Step 4: Run `python -m pytest tests/test_capture_window.py -q`** and confirm the UI tests pass.
- [ ] **Step 5: Commit** with `feat: offer fast and precise gaze calibration`.

### Task 6: Compatibility, Packaging, and End-to-End Verification

**Files:**
- Modify: `src/pure_gaze_typing/layout.py`
- Modify: `packaging/build.ps1`
- Modify: `README.md`
- Modify: tests affected by the calibration version bump.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: rebuilt `眼动采集校准.exe` and `纯眼动打字器.exe` sharing the portable data root.

- [ ] **Step 1: Add or update failing compatibility tests** proving old lightweight calibrations are retained on disk but not loaded as the new pipeline version.
- [ ] **Step 2: Run the compatibility tests** and confirm they fail before the version bump.
- [ ] **Step 3: Bump the layout/calibration pipeline version, update packaging safeguards, and document the two calibration modes and data path.**
- [ ] **Step 4: Run `python -m pytest -q`** and confirm the complete suite passes.
- [ ] **Step 5: Run `powershell -ExecutionPolicy Bypass -File packaging/build.ps1`**, then run both packaged `--self-test` commands and verify exit code 0.
- [ ] **Step 6: Copy the package to the visible release folder without touching `release\纯眼动打字系统数据`, migrate legacy data, and verify both executable-derived paths are identical.**
- [ ] **Step 7: Run a camera preview and loopback UDP smoke test**, inspect logs for model/face/camera errors, and stop all test processes.
- [ ] **Step 8: Review `git diff`, commit, push `feature/pure-gaze-typing`, and report exact executable/data locations and verification evidence.**

## Self-Review

- Spec coverage: portable storage, both calibration modes, reference filtering/affine/range/smoothing behavior, optional 5/6 gate, failed-region retry, UI, packaging, and hardware smoke tests are each assigned to a task.
- Placeholder scan: no deferred implementation placeholders are present.
- Type consistency: Tasks 2-5 share `CalibrationMode`, `CalibrationProfile`, runtime metadata, and `start_calibration(mode, validate)` consistently.
