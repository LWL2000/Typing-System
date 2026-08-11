from pathlib import Path

from pure_gaze_typing.paths import AppPaths


def test_source_default_uses_appdata(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("APPDATA", str(tmp_path))

    paths = AppPaths.default(frozen=False)

    assert paths.root == (tmp_path / "PureGazeTyping").resolve()


def test_packaged_apps_share_release_data_root(tmp_path: Path):
    release = tmp_path / "release"
    package = release / "纯眼动打字系统"
    capture_exe = package / "眼动采集校准" / "眼动采集校准.exe"
    typing_exe = package / "纯眼动打字器" / "纯眼动打字器.exe"

    capture_paths = AppPaths.default(executable=capture_exe, frozen=True)
    typing_paths = AppPaths.default(executable=typing_exe, frozen=True)

    expected = (release / "纯眼动打字系统数据").resolve()
    assert capture_paths.root == expected
    assert typing_paths.root == expected


def test_migrate_legacy_copies_data_into_empty_portable_root(tmp_path: Path):
    legacy = tmp_path / "legacy"
    portable = AppPaths.for_root(tmp_path / "portable")
    (legacy / "calibration" / "cal-1").mkdir(parents=True)
    (legacy / "calibration" / "cal-1" / "model.pkl").write_bytes(b"model")
    (legacy / "calibration" / "current.json").write_text(
        '{"calibration_id":"cal-1"}', encoding="utf-8"
    )
    (legacy / "settings.json").write_text('{"dwell_seconds":1.2}', encoding="utf-8")

    migrated = portable.migrate_legacy(legacy)

    assert migrated
    assert (portable.calibration_dir / "cal-1" / "model.pkl").read_bytes() == b"model"
    assert (portable.calibration_dir / "current.json").is_file()
    assert portable.settings_file.read_text(encoding="utf-8") == '{"dwell_seconds":1.2}'


def test_migrate_legacy_never_overwrites_portable_files(tmp_path: Path):
    legacy = tmp_path / "legacy"
    portable = AppPaths.for_root(tmp_path / "portable")
    legacy.mkdir()
    portable.root.mkdir()
    (legacy / "settings.json").write_text("legacy", encoding="utf-8")
    portable.settings_file.write_text("portable", encoding="utf-8")
    (legacy / "logs").mkdir()
    (legacy / "logs" / "capture.log").write_text("legacy-log", encoding="utf-8")
    portable.logs_dir.mkdir()
    (portable.logs_dir / "capture.log").write_text("portable-log", encoding="utf-8")

    migrated = portable.migrate_legacy(legacy)

    assert not migrated
    assert portable.settings_file.read_text(encoding="utf-8") == "portable"
    assert (portable.logs_dir / "capture.log").read_text(encoding="utf-8") == "portable-log"
