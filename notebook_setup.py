# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Colab / Kaggle: locate the ReproPilot repo, chdir, extend sys.path, and
# ``pip install -e .`` so project imports work when the notebook cwd
# is not the repo root (e.g. /kaggle/working) or cells are run out of order.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def candidate_directories() -> list[Path]:
    out: list[Path] = []
    home = Path.home()
    bases = [
        Path.cwd(),
        Path.cwd() / "ReproPilot",
        Path.cwd() / "repropilot",
        Path("/content/ReproPilot"),
        Path("/content/repropilot"),
        home / "SageMaker" / "ReproPilot",
        home / "SageMaker" / "repropilot",
        home / "SageMaker",
        Path("/kaggle/working/ReproPilot"),
        Path("/kaggle/working/repropilot"),
        Path("/kaggle/working"),
    ]
    for b in bases:
        out.extend([b, b / "ReproPilot", b / "repropilot"])

    kin = Path("/kaggle/input")
    if kin.is_dir():
        for ds in sorted(kin.iterdir()):
            out.extend([ds, ds / "ReproPilot", ds / "repropilot"])
    return out


def find_root() -> Path | None:
    for p in candidate_directories():
        try:
            if (p / "pyproject.toml").is_file() and (p / "models.py").is_file():
                return p.resolve()
        except OSError:
            continue
    return None


def apply_grpo_training_defaults() -> None:
    """Set recommended GRPO / reward env vars only when unset (Colab / local notebooks).

    Call after ``ensure_repropilot_on_path()`` so training cells pick up sane defaults:
    multi-turn in-process rollouts, mild reward squash, tie-break jitter for zero-std groups.
    """
    defaults: dict[str, str] = {
        "REPROPILOT_ROLLOUT_TURNS": "3",
        "REPROPILOT_ROLLOUT_GAMMA": "0.9",
        "REPROPILOT_REWARD_SQUASH": "1",
        "REPROPILOT_REWARD_SQUASH_SCALE": "2.5",
        "REPROPILOT_REWARD_JITTER": "1",
        "REPROPILOT_REWARD_JITTER_MAG": "0.001",
        "TRL_EXPERIMENTAL_SILENCE": "1",
    }
    for key, val in defaults.items():
        if key not in os.environ or os.environ.get(key, "").strip() == "":
            os.environ[key] = val


def ensure_repropilot_on_path(*, pip_install: bool = True) -> Path:
    """Return repo root. Idempotent: safe to call from every notebook section."""
    r = find_root()
    if r is None:
        raise RuntimeError(
            "ReproPilot repo not found (need pyproject.toml and models.py). "
            "On Kaggle: enable Internet and run the clone cell, or add this repo as a Dataset "
            "under /kaggle/input (see README)."
        )
    os.chdir(r)
    rs = str(r)
    if rs not in sys.path:
        sys.path.insert(0, rs)
    if pip_install:
        # SageMaker AMIs may ship GCC too old to *compile* NumPy 2.4+; pre-install a wheel.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-qqq", "-U", "pip", "setuptools", "wheel"],
            cwd=rs,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-qqq",
                "numpy>=1.26,<2.3",
                "--only-binary",
                "numpy",
            ],
            cwd=rs,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", "."],
            cwd=rs,
            check=True,
        )
    return r
