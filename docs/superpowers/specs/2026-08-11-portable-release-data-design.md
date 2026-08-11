# Portable Release Data Design

## Goal

Store all user-generated Pure Gaze Typing data beneath the visible release directory instead of `%APPDATA%`.

## Storage Layout

Both packaged applications use this shared root:

```text
D:\CodeX\脑电软件设计\cap32_gaze_typing\release\纯眼动打字系统数据
```

The root contains:

```text
calibration\
logs\
sessions\
settings.json
```

Keeping the data beside, rather than inside, `纯眼动打字系统` prevents a package replacement from deleting user data.

## Path Resolution

When frozen by PyInstaller, `AppPaths.default()` derives the `release` directory from `sys.executable` and uses its `纯眼动打字系统数据` child. Both executables therefore resolve the same data root despite living in separate application subdirectories.

When running from source or tests, the existing `%APPDATA%\PureGazeTyping` default remains available unless an explicit root is supplied.

## Migration

On the first packaged launch, if the release data root is empty and `%APPDATA%\PureGazeTyping` exists, copy its calibration, sessions, logs, and settings into the release data root. Never overwrite files already present in the release data root. The current `gaze-grid-v2` calibration must remain usable after migration.

## Packaging

The build output continues to contain only application files. Build and copy operations must not delete or replace the sibling `纯眼动打字系统数据` directory.

## Verification

- Unit-test frozen and source path resolution.
- Unit-test one-time non-destructive migration.
- Build both executables and run their self-tests.
- Verify both packaged executables resolve the same release data root.
- Verify the existing calibration is present in the new location.
