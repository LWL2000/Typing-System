from __future__ import annotations

import multiprocessing

from pure_gaze_typing.capture_app import main


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
