#!/usr/bin/env python3
"""Export ReproPilot evaluation plots into docs/submission_results."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.plots import generate_plots  # noqa: E402


def main() -> int:
    report = ROOT / "outputs" / "eval" / "smart_heldout.json"
    if not report.is_file():
        print(
            f"Missing report: {report}. Run scripts/generate_committed_submission_plots.py first."
        )
        return 1
    paths = generate_plots(report, ROOT / "docs" / "submission_results")
    print("Exported ReproPilot submission plots:")
    for path in paths:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
