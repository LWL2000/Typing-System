import logging
import json
from pathlib import Path

import pytest

from pure_gaze_typing.app_logging import configure_logging
from pure_gaze_typing.paths import AppPaths
from pure_gaze_typing.settings import TypingSettings, load_settings, save_settings


def test_default_settings_match_product_defaults(tmp_path: Path):
    settings = load_settings(tmp_path / "missing.json")
    assert settings == TypingSettings(True, 1.0, False)


def test_settings_round_trip_and_validation(tmp_path: Path):
    path = tmp_path / "settings.json"
    save_settings(path, TypingSettings(False, 1.4, True))
    assert load_settings(path) == TypingSettings(False, 1.4, True)
    with pytest.raises(ValueError, match="0.5"):
        TypingSettings(True, 0.4, False)


def test_old_settings_enable_adaptive_correction_by_default(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "show_gaze_point": False,
                "dwell_seconds": 1.2,
                "validate_calibration": True,
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(path)

    assert settings.adaptive_correction_enabled is True


def test_adaptive_setting_requires_boolean() -> None:
    with pytest.raises(ValueError, match="adaptive_correction_enabled"):
        TypingSettings(adaptive_correction_enabled="yes")


def test_malformed_settings_are_quarantined(tmp_path: Path):
    path = tmp_path / "settings.json"
    path.write_text("not json", encoding="utf-8")
    assert load_settings(path) == TypingSettings()
    assert not path.exists()
    assert (tmp_path / "settings.invalid.json").read_text(encoding="utf-8") == "not json"


def test_logging_creates_utf8_file_under_app_data(tmp_path: Path):
    paths = AppPaths.for_root(tmp_path)
    path = configure_logging(paths, "typing")
    logging.getLogger("pure_gaze_typing").info("眼动已连接")
    for handler in logging.getLogger("pure_gaze_typing").handlers:
        handler.flush()
    assert path.parent == tmp_path / "logs"
    assert "眼动已连接" in path.read_text(encoding="utf-8")
