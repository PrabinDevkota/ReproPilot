#!/usr/bin/env python3
"""Generate ReproPilot held-out reports and committable plot artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.random_policy import random_action
from baselines.smart_policy import smart_action
from evaluation.evaluate_policy import evaluate, write_report
from evaluation.plots import generate_plots


def main() -> int:
    out = ROOT / "outputs" / "eval"
    docs = ROOT / "docs" / "submission_results"
    out.mkdir(parents=True, exist_ok=True)
    smart = evaluate(lambda obs, rng: smart_action(obs, rng), split="heldout")
    random_report = evaluate(lambda obs, rng: random_action(obs, rng), split="heldout")
    smart_path = write_report(smart, out / "smart_heldout.json")
    write_report(random_report, out / "random_heldout.json")
    paths = generate_plots(smart_path, docs)
    print("Smart baseline:", smart.aggregate())
    print("Random baseline:", random_report.aggregate())
    print("Wrote plots:")
    for path in paths:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
