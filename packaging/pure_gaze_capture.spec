# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


ROOT = Path(SPECPATH).resolve().parent
APP_NAME = "眼动采集校准"

datas = [(str(ROOT / "resources" / "face_landmarker.task"), "resources")]
binaries = []
hiddenimports = [
    "pure_gaze_typing.capture_app",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "eyetrax.gaze",
    "eyetrax.filters",
    "mediapipe.tasks.python.vision",
    "sklearn.linear_model",
]

datas += collect_data_files("mediapipe")
binaries += collect_dynamic_libs("mediapipe")
hiddenimports += collect_submodules("eyetrax")
hiddenimports += collect_submodules("mediapipe.tasks.python.vision")

for distribution in ("eyetrax", "mediapipe"):
    datas += copy_metadata(distribution)

a = Analysis(
    [str(ROOT / "packaging" / "capture_launcher.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["PySide6", "PyQt5"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
