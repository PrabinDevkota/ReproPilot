"""Plot generation for ReproPilot evaluation reports."""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    import binascii

    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)


def _fallback_png(path: Path, labels: list[str], values: list[float], *, width: int = 720, height: int = 420) -> Path:
    """Write a small valid PNG without third-party plotting libraries."""
    bg = (255, 255, 255)
    axis = (30, 41, 59)
    colors = [(37, 99, 235), (22, 163, 74), (220, 38, 38), (147, 51, 234)]
    pixels = [[bg for _ in range(width)] for _ in range(height)]

    def rect(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(height, y1)):
            row = pixels[y]
            for x in range(max(0, x0), min(width, x1)):
                row[x] = color

    left, bottom, top, right = 60, height - 55, 35, width - 30
    rect(left, top, left + 2, bottom, axis)
    rect(left, bottom, right, bottom + 2, axis)
    maxv = max(1.0, max([abs(v) for v in values] or [1.0]))
    n = max(1, len(values))
    slot = max(24, (right - left) // n)
    for i, value in enumerate(values):
        bar_h = int((max(0.0, value) / maxv) * (bottom - top - 12))
        x0 = left + i * slot + slot // 5
        x1 = left + (i + 1) * slot - slot // 5
        rect(x0, bottom - bar_h, x1, bottom, colors[i % len(colors)])

    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend((r, g, b))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)
    return path


def _bar(path: Path, title: str, labels: list[str], values: list[float], *, ylim: tuple[float, float] | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return _fallback_png(path, labels, values)

    try:
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.bar(labels, values, color=["#2563eb", "#16a34a", "#dc2626", "#9333ea"][: len(labels)])
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelrotation=15)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        return path
    except Exception:
        return _fallback_png(path, labels, values)


def generate_plots(report_path: Path, out_dir: Path) -> list[Path]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    agg = data.get("aggregate", {})
    episodes = data.get("episodes", [])
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _bar(out_dir / "total_reward_before_after.png", "Average Total Reward", ["policy"], [float(agg.get("average_total_reward", 0.0))]),
        _bar(out_dir / "verdict_accuracy_before_after.png", "Verdict Accuracy", ["policy"], [float(agg.get("verdict_accuracy", 0.0))], ylim=(0, 1)),
        _bar(out_dir / "failure_type_accuracy.png", "Failure Type Accuracy", ["policy"], [float(agg.get("failure_type_accuracy", 0.0))], ylim=(0, 1)),
        _bar(out_dir / "evidence_validity_rate.png", "Evidence Validity Rate", ["policy"], [float(agg.get("evidence_validity_rate", 0.0))], ylim=(0, 1)),
        _bar(out_dir / "fabricated_evidence_rate.png", "Fabricated Evidence Rate", ["policy"], [float(agg.get("fabricated_evidence_rate", 0.0))], ylim=(0, 1)),
        _bar(out_dir / "checker_usage_rate.png", "Relevant Checker Usage", ["policy"], [float(agg.get("relevant_checker_usage_rate", 0.0))], ylim=(0, 1)),
    ]

    novelty_total = sum(1 for e in episodes if "novel" in str(e.get("scenario_id", "")))
    novelty_ok = sum(1 for e in episodes if "novel" in str(e.get("scenario_id", "")) and e.get("verdict_correct"))
    paths.append(_bar(out_dir / "novelty_calibration.png", "Novelty Calibration", ["novel"], [novelty_ok / max(1, novelty_total)], ylim=(0, 1)))

    by_type: dict[str, list[float]] = {}
    for e in episodes:
        sid = str(e.get("scenario_id", "unknown"))
        key = sid.split("_")[1] if sid.startswith("heldout_") and "_" in sid else sid.split("_")[0]
        by_type.setdefault(key, []).append(1.0 if e.get("verdict_family_correct") else 0.0)
    labels = sorted(by_type)[:10]
    vals = [sum(by_type[k]) / max(1, len(by_type[k])) for k in labels]
    paths.append(_bar(out_dir / "scenario_type_breakdown.png", "Scenario Type Breakdown", labels or ["none"], vals or [0.0], ylim=(0, 1)))

    channel_keys = [
        "verdict_accuracy",
        "failure_type_accuracy",
        "evidence_validity_rate",
        "relevant_checker_usage_rate",
        "fabricated_evidence_rate",
    ]
    paths.append(
        _bar(
            out_dir / "reward_channels.png",
            "Evaluation Channel Proxies",
            [k.replace("_", "\n") for k in channel_keys],
            [float(agg.get(k, 0.0)) for k in channel_keys],
            ylim=(0, 1),
        )
    )
    return paths


__all__ = ["generate_plots"]
