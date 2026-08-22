"""Make ffmpeg reachable from Python regardless of how the process was launched."""

import os
import shutil
from pathlib import Path


def ensure_ffmpeg():
    """Put ffmpeg on PATH for this process.

    The transformers ASR pipeline shells out to ffmpeg to decode audio files.
    winget installs ffmpeg and updates the persistent PATH, but processes that
    were already running (VS Code and any terminal it spawned) keep their old
    copy of the environment until they restart. Recover the real location so
    this works without a restart.
    """
    if shutil.which("ffmpeg"):
        return

    candidates = []
    if os.name == "nt":
        # The persistent user PATH, which winget does update.
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                candidates += winreg.QueryValueEx(key, "Path")[0].split(os.pathsep)
        except OSError:
            pass
        # Fall back to the winget package location itself.
        local = os.environ.get("LOCALAPPDATA", "")
        if local:
            candidates += [
                str(p)
                for p in Path(local).glob("Microsoft/WinGet/Packages/Gyan.FFmpeg*/*/bin")
            ]

    for directory in candidates:
        if directory and shutil.which("ffmpeg", path=directory):
            os.environ["PATH"] = directory + os.pathsep + os.environ["PATH"]
            return

    raise SystemExit(
        "ffmpeg not found and is required to decode audio files.\n"
        "Install it with:  winget install Gyan.FFmpeg"
    )
