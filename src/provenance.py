from __future__ import annotations

import os
import subprocess
from pathlib import Path


def capture_git_provenance(root: Path | None = None) -> dict[str, str | bool]:
    root = root or Path.cwd()
    environment_commit = os.getenv("GIT_COMMIT")
    if environment_commit and environment_commit != "unknown":
        return {"git_commit": environment_commit, "git_dirty": False}

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": True}
