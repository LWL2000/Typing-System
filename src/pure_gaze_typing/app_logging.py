from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import sys

from .paths import AppPaths


def configure_logging(paths: AppPaths, app_name: str) -> Path:
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    path = paths.logs_dir / f"{app_name}_{datetime.now():%Y%m%d_%H%M%S_%f}.log"
    logger = logging.getLogger("pure_gaze_typing")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in tuple(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return Path(path)
