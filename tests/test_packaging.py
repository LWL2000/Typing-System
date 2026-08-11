from pathlib import Path


def test_windows_powershell_build_script_is_ascii_safe():
    script = Path("packaging/build.ps1").read_bytes()
    assert script.decode("ascii")


def test_specs_do_not_collect_entire_scientific_and_qt_packages():
    for name in ("pure_gaze_capture.spec", "pure_gaze_typing.spec"):
        content = (Path("packaging") / name).read_text(encoding="utf-8")
        assert "collect_all" not in content
