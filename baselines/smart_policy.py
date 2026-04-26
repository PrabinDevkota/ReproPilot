"""Rule-based ReproPilot audit baseline."""

from __future__ import annotations

import random
from typing import Any

try:
    from ..models import (
        ActionType,
        AgentAction,
        CheckName,
        FailureType,
        ReproPilotObservation,
        ValidationVerdict,
    )
except ImportError:
    from models import (
        ActionType,
        AgentAction,
        CheckName,
        FailureType,
        ReproPilotObservation,
        ValidationVerdict,
    )


_CHECK_TO_ACTION = {
    CheckName.metric_check.value: ActionType.run_metric_check,
    CheckName.split_check.value: ActionType.run_split_check,
    CheckName.leakage_check.value: ActionType.run_leakage_check,
    CheckName.seed_check.value: ActionType.run_seed_check,
    CheckName.ablation_check.value: ActionType.run_ablation_check,
    CheckName.reproduction_check.value: ActionType.run_reproduction_check,
    CheckName.paper_code_consistency_check.value: ActionType.run_paper_code_consistency_check,
    CheckName.dataset_provenance_check.value: ActionType.run_dataset_provenance_check,
    CheckName.hyperparameter_search_check.value: ActionType.run_hyperparameter_search_check,
    CheckName.baseline_fairness_check.value: ActionType.run_baseline_fairness_check,
    CheckName.statistical_significance_check.value: ActionType.run_statistical_significance_check,
    CheckName.implementation_completeness_check.value: ActionType.run_implementation_completeness_check,
}


def _history(meta: dict[str, Any]) -> set[str]:
    return {
        str(c.get("check_name"))
        for c in meta.get("checks") or []
        if isinstance(c, dict)
    }


def _failed_checks(meta: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        c
        for c in meta.get("checks") or []
        if isinstance(c, dict)
        and c.get("status") in {"failed", "inconclusive"}
        and c.get("failure_type")
    ]


def _observed_evidence(meta: dict[str, Any]) -> list[str]:
    return [
        str(e.get("id"))
        for e in meta.get("observed_evidence") or []
        if isinstance(e, dict)
    ]


def smart_action(
    obs: ReproPilotObservation, rng: random.Random | None = None
) -> AgentAction:
    meta: dict[str, Any] = obs.metadata or {}
    checks_done = _history(meta)
    step = int(meta.get("episode_step") or 0)

    if step == 0:
        return AgentAction(
            action_type=ActionType.read_claim,
            explanation="Start by reading the target claim.",
        )

    inspected_files = set(meta.get("inspected_code_file_ids") or [])
    inspected_cfgs = set(meta.get("inspected_config_ids") or [])
    inspected_logs = set(meta.get("inspected_log_ids") or [])

    for fid in meta.get("code_file_ids") or []:
        if fid not in inspected_files:
            return AgentAction(
                action_type=ActionType.inspect_code_file,
                target_id=fid,
                explanation="Inspect code before running methodology checks.",
            )

    for cid in meta.get("config_ids") or []:
        if cid not in inspected_cfgs:
            return AgentAction(
                action_type=ActionType.inspect_config,
                target_id=cid,
                explanation="Inspect config for split and metric.",
            )

    for lid in meta.get("log_ids") or []:
        if lid not in inspected_logs:
            return AgentAction(
                action_type=ActionType.inspect_logs,
                target_id=lid,
                explanation="Inspect logs for metric values and seed behavior.",
            )

    failed = _failed_checks(meta)
    if failed:
        evidence_ids = _observed_evidence(meta)
        failure = FailureType(str(failed[0].get("failure_type")))
        verdict = (
            ValidationVerdict.NOT_SUPPORTED_RESULT_MISMATCH
            if failure == FailureType.result_mismatch
            else ValidationVerdict.NOT_SUPPORTED_METHOD_INVALID
        )
        if failure == FailureType.missing_artifact:
            verdict = ValidationVerdict.INCONCLUSIVE_MISSING_ARTIFACTS
        return AgentAction(
            action_type=ActionType.submit_verdict,
            verdict=verdict,
            failure_type=failure,
            evidence_ids=evidence_ids,
            explanation="A deterministic checker found a methodology issue.",
        )

    if int(meta.get("steps_remaining") or 0) <= 2 and checks_done:
        evidence_ids = _observed_evidence(meta)
        verdict = (
            ValidationVerdict.PLAUSIBLY_VALIDATED_NOVEL_METHOD
            if int(meta.get("novelty_level") or 0) >= 4
            else ValidationVerdict.SUPPORTED_RESULT_AND_METHOD
        )
        return AgentAction(
            action_type=ActionType.submit_verdict,
            verdict=verdict,
            failure_type=FailureType.none,
            evidence_ids=evidence_ids,
            explanation="No checker has failed within the available audit budget.",
        )

    priority = [
        CheckName.metric_check.value,
        CheckName.split_check.value,
        CheckName.leakage_check.value,
        CheckName.seed_check.value,
        CheckName.ablation_check.value,
        CheckName.paper_code_consistency_check.value,
        CheckName.reproduction_check.value,
        CheckName.dataset_provenance_check.value,
        CheckName.hyperparameter_search_check.value,
        CheckName.baseline_fairness_check.value,
        CheckName.statistical_significance_check.value,
        CheckName.implementation_completeness_check.value,
    ]
    for name in priority:
        if name not in checks_done:
            return AgentAction(
                action_type=_CHECK_TO_ACTION[name],
                target_id=meta.get("claim_id"),
                explanation=f"Run {name} before verdict.",
            )

    evidence_ids = _observed_evidence(meta)
    if int(meta.get("novelty_level") or 0) >= 4:
        verdict = ValidationVerdict.PLAUSIBLY_VALIDATED_NOVEL_METHOD
    else:
        verdict = ValidationVerdict.SUPPORTED_RESULT_AND_METHOD
    return AgentAction(
        action_type=ActionType.submit_verdict,
        verdict=verdict,
        failure_type=FailureType.none,
        evidence_ids=evidence_ids,
        explanation="Available checks passed.",
    )
