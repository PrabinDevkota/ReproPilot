"""Held-out evaluation for ReproPilot policies."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

try:
    from ..models import AgentAction, ReproPilotObservation
    from ..server.repropilot_environment import ReproPilotEnvironment
except ImportError:
    from models import AgentAction, ReproPilotObservation
    from server.repropilot_environment import ReproPilotEnvironment

Policy = Callable[[ReproPilotObservation, random.Random], AgentAction]
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EpisodeMetrics:
    scenario_id: str
    total_reward: float
    steps: int
    verdict_correct: bool
    verdict_family_correct: bool
    failure_type_correct: bool
    evidence_valid: bool
    fabricated_evidence: bool
    relevant_checker_used: bool
    reproduction_check_used: bool
    timeout: bool


@dataclass
class EvalReport:
    episodes: list[EpisodeMetrics] = field(default_factory=list)

    def aggregate(self) -> dict[str, float]:
        n = len(self.episodes) or 1
        return {
            "episodes": float(len(self.episodes)),
            "average_total_reward": sum(e.total_reward for e in self.episodes) / n,
            "verdict_accuracy": sum(e.verdict_correct for e in self.episodes) / n,
            "verdict_family_accuracy": sum(e.verdict_family_correct for e in self.episodes) / n,
            "failure_type_accuracy": sum(e.failure_type_correct for e in self.episodes) / n,
            "evidence_validity_rate": sum(e.evidence_valid for e in self.episodes) / n,
            "fabricated_evidence_rate": sum(e.fabricated_evidence for e in self.episodes) / n,
            "relevant_checker_usage_rate": sum(e.relevant_checker_used for e in self.episodes) / n,
            "reproduction_check_usage_rate": sum(e.reproduction_check_used for e in self.episodes) / n,
            "timeout_rate": sum(e.timeout for e in self.episodes) / n,
            "average_steps_to_verdict": sum(e.steps for e in self.episodes) / n,
        }


def _family(value: str | None) -> str:
    v = value or ""
    if v.startswith("NOT_SUPPORTED"):
        return "invalid"
    if v.startswith("INCONCLUSIVE") or v == "NOT_ENOUGH_EVIDENCE":
        return "inconclusive"
    if v in {"SUPPORTED_RESULT_AND_METHOD", "PLAUSIBLY_VALIDATED_NOVEL_METHOD"}:
        return "valid"
    return "unknown"


def run_episode(path: Path, policy: Policy, *, seed: int = 0, max_steps: int = 12) -> EpisodeMetrics:
    env = ReproPilotEnvironment(path)
    obs = env.reset()
    rng = random.Random(seed)
    total = 0.0
    done = False
    for _ in range(max_steps):
        action = policy(obs, rng)
        obs = env.step(action)
        total += float(obs.reward or 0.0)
        done = bool(obs.done)
        if done:
            break
    gold = env.hidden_gold
    st = env.audit_state
    submitted_evidence = []
    if st.action_history:
        submitted_evidence = list(st.action_history[-1].get("evidence_ids") or [])
    known_evidence = {e.id for e in st.evidence}
    fabricated = any(eid not in known_evidence for eid in submitted_evidence)
    evidence_valid = bool(submitted_evidence) and not fabricated
    checks = {c.check_name for c in st.checks}
    return EpisodeMetrics(
        scenario_id=st.scenario_id,
        total_reward=total,
        steps=st.step,
        verdict_correct=st.final_verdict == gold.gold_verdict,
        verdict_family_correct=_family(st.final_verdict.value if st.final_verdict else None) == _family(gold.gold_verdict.value),
        failure_type_correct=st.final_failure_type == gold.gold_failure_type,
        evidence_valid=evidence_valid,
        fabricated_evidence=fabricated,
        relevant_checker_used=bool(checks & set(gold.gold_required_checks)),
        reproduction_check_used=any(str(c.value) == "reproduction_check" for c in checks),
        timeout=not done or st.final_verdict is None,
    )


def evaluate(policy: Policy, *, split: str = "heldout", episodes_per_scenario: int = 1) -> EvalReport:
    report = EvalReport()
    for path in sorted((ROOT / "scenarios" / split).glob("*.json")):
        for i in range(episodes_per_scenario):
            report.episodes.append(run_episode(path, policy, seed=1000 + i))
    return report


def write_report(report: EvalReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"aggregate": report.aggregate(), "episodes": [asdict(e) for e in report.episodes]}, indent=2), encoding="utf-8")
    return path
