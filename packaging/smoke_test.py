from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def validate_release(release: Path) -> None:
    required = (
        release / "眼动采集校准" / "眼动采集校准.exe",
        release / "眼动采集校准" / "_internal",
        release / "眼动采集校准" / "_internal" / "resources" / "face_landmarker.task",
        release / "纯眼动打字器" / "纯眼动打字器.exe",
        release / "纯眼动打字器" / "_internal",
        release / "使用说明.md",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("发布目录缺少：\n" + "\n".join(missing))
    for executable in (required[0], required[3]):
        if executable.stat().st_size <= 0:
            raise RuntimeError(f"可执行文件为空：{executable}")


def validate_executable(executable: Path) -> None:
    if not executable.is_file():
        raise RuntimeError(f"可执行文件不存在：{executable}")
    completed = subprocess.run(
        [str(executable), "--self-test"],
        cwd=executable.parent,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"自检失败（退出码 {completed.returncode}）：\n{completed.stdout}\n{completed.stderr}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--release", type=Path)
    group.add_argument("--exe", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.release is not None:
            validate_release(args.release.resolve())
        else:
            validate_executable(args.exe.resolve())
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
