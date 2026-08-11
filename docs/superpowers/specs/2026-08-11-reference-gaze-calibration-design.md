# Reference Gaze Calibration Design

## Goal

Replace the current lightweight calibration and gaze output behavior with the proven pipeline from `D:\claude\新协议2\src\eyetrax_gaze_stream.py`, without modifying the reference project. Preserve the existing two-application architecture and UDP protocol.

## Calibration Modes

The capture application presents a gentle full-screen choice before calibration:

- **快速校准（推荐）**: approximately 25-35 seconds. Collect a 3x3 full-screen grid once, collect the center, back button, and six typing-region centers once, then perform affine bias correction.
- **精确校准**: approximately 50-60 seconds. Collect the same 3x3 grid, then collect the center, back button, and six typing-region centers in forward and reverse passes with the reference project's longer sample windows, followed by affine bias correction.

Both modes reject blinks and missing faces, remove unstable feature rows with median/MAD filtering, balance the sample count across points, train the EyeTrax ridge model, attach a feature-range guard, and apply a screen affine correction before smoothing.

## Runtime Gaze Output

Match the reference runtime behavior:

- Use EyeTrax `KalmanEMASmoother` with `ema_alpha=0.35`.
- Reset smoothing after 0.8 seconds of invalid face/blink input.
- Measure feature distance in the trained model's scaled feature space.
- Report reduced quality outside the calibrated range and reject samples beyond twice the calibrated threshold.
- Apply screen affine correction before smoothing and clamp only the final coordinates to the screen.
- Continue publishing gaze coordinates, blink state, face state, quality, calibration ID, layout version, and heartbeats through the existing loopback UDP protocol.

## Optional Quality Validation

The existing checkbox continues to control six-region validation.

- When validation is disabled, save after training and affine correction.
- When validation is enabled, keep the new model provisional until at least 5 of 6 typing regions pass.
- Do not replace the previously active calibration before the provisional model passes.
- If validation fails, offer only abnormal-region recapture. Merge the replacement samples with the provisional training set, retrain, correct bias, and revalidate the failed regions.
- Promote and save the calibration only after the cumulative result reaches 5/6.

## UI Flow

1. Connect the camera and retain the live preview.
2. Select whether post-calibration quality validation is enabled.
3. Press **快速校准**.
4. Choose **快速校准（推荐）** or **精确校准** in a calm full-screen submenu showing estimated duration.
5. Follow the full-screen calibration points and progress rings.
6. Return to the capture window when calibration is saved or when corrective action is required.

Escape cancels calibration without changing the current saved model.

## Data Storage

Use the separately approved portable data root:

```text
D:\CodeX\脑电软件设计\cap32_gaze_typing\release\纯眼动打字系统数据
```

Packaged applications share its `calibration`, `logs`, `sessions`, and `settings.json` contents. On first packaged launch, copy legacy `%APPDATA%\PureGazeTyping` data non-destructively when the portable root is empty.

## Compatibility

Bump the layout/calibration pipeline version so older lightweight models are not loaded as reference-pipeline models. Preserve old files on disk; only the active calibration pointer changes after a successful new calibration.

## Verification

- Unit-test both calibration profiles and point order.
- Unit-test robust feature filtering, sample balancing, affine correction, feature-range scoring, and smoother reset.
- Unit-test provisional validation so failed models never replace the active calibration.
- Unit-test failed-region retry and final promotion.
- Unit-test portable path resolution and non-destructive migration.
- Run all tests, build both executables, run packaged self-tests, and verify both resolve the same portable data root.
- Perform a camera/UDP smoke test while leaving the reference project unchanged.
