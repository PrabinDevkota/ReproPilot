"""Episode-level OpenEnv graders for ReproPilot."""

from __future__ import annotations

STRICT_MIN = 0.01
STRICT_MAX = 0.99


def _bounded(value: float) -> float:
    return min(max(round(float(value), 4), STRICT_MIN), STRICT_MAX)


def _rewards(trajectory: dict | None) -> list[float]:
    payload = trajectory or {}
    vals = payload.get("rewards")
    if isinstance(vals, list) and vals:
        return [float(v) for v in vals]
    if "score" in payload:
        return [float(payload["score"])]
    return [float(payload.get("reward", 0.5))]


def _score(trajectory: dict | None, *, evidence_weight: float) -> float:
    rewards = _rewards(trajectory)
    mean_reward = sum(rewards) / max(1, len(rewards))
    normalized = 0.5 + mean_reward / 6.0
    metadata = (trajectory or {}).get("metadata") or {}
    if metadata.get("fabricated_evidence"):
        normalized -= 0.25
    if metadata.get("relevant_checker_used"):
        normalized += evidence_weight
    return _bounded(normalized)


def metric_mismatch_grader(trajectory: dict | None = None) -> float:
    return _score(trajectory, evidence_weight=0.05)


def split_mismatch_grader(trajectory: dict | None = None) -> float:
    return _score(trajectory, evidence_weight=0.08)


def full_audit_grader(trajectory: dict | None = None) -> float:
    return _score(trajectory, evidence_weight=0.12)
