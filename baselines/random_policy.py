"""Type-valid random ReproPilot baseline."""

from __future__ import annotations

import random
from typing import Any

try:
    from ..models import ActionType, AgentAction, FailureType, ReproPilotObservation, ValidationVerdict
except ImportError:
    from models import ActionType, AgentAction, FailureType, ReproPilotObservation, ValidationVerdict


def random_action(obs: ReproPilotObservation, rng: random.Random) -> AgentAction:
    meta: dict[str, Any] = obs.metadata or {}
    action_type = rng.choice(list(ActionType))
    target_pool: list[str] = []
    if action_type == ActionType.inspect_paper_section:
        target_pool = list(meta.get("paper_section_ids") or [])
    elif action_type == ActionType.inspect_code_file:
        target_pool = list(meta.get("code_file_ids") or [])
    elif action_type == ActionType.inspect_config:
        target_pool = list(meta.get("config_ids") or [])
    elif action_type == ActionType.inspect_logs:
        target_pool = list(meta.get("log_ids") or [])
    elif action_type == ActionType.search_artifacts:
        target_pool = ["accuracy", "split", "seed", "ablation", "entropy"]
    elif action_type == ActionType.submit_verdict:
        return AgentAction(
            action_type=action_type,
            verdict=rng.choice(list(ValidationVerdict)),
            failure_type=rng.choice(list(FailureType)),
            evidence_ids=[],
            explanation="Random baseline verdict.",
        )
    return AgentAction(
        action_type=action_type,
        target_id=rng.choice(target_pool) if target_pool else None,
        explanation="Random baseline action.",
    )
