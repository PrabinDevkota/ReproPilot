"""Interpretable ReproPilot reward channels."""

from __future__ import annotations

try:
    from .models import (
        AgentAction,
        AuditState,
        CheckName,
        FailureType,
        HiddenGold,
        RewardBreakdown,
        ValidationVerdict,
    )
except ImportError:
    from models import (
        AgentAction,
        AuditState,
        CheckName,
        FailureType,
        HiddenGold,
        RewardBreakdown,
        ValidationVerdict,
    )


INVALID_GOLD = {
    ValidationVerdict.NOT_SUPPORTED_METHOD_INVALID,
    ValidationVerdict.NOT_SUPPORTED_RESULT_MISMATCH,
}
INCONCLUSIVE_GOLD = {
    ValidationVerdict.INCONCLUSIVE_MISSING_ARTIFACTS,
    ValidationVerdict.INCONCLUSIVE_NEEDS_DOMAIN_REVIEW,
    ValidationVerdict.NOT_ENOUGH_EVIDENCE,
}
VALID_GOLD = {
    ValidationVerdict.SUPPORTED_RESULT_AND_METHOD,
    ValidationVerdict.PLAUSIBLY_VALIDATED_NOVEL_METHOD,
}


def _family(verdict: ValidationVerdict | None) -> str:
    if verdict in INVALID_GOLD:
        return "invalid"
    if verdict in INCONCLUSIVE_GOLD:
        return "inconclusive"
    if verdict in VALID_GOLD:
        return "valid"
    return "unknown"


def compute_terminal_reward(
    state: AuditState,
    gold: HiddenGold,
    action: AgentAction,
    *,
    hidden_access_attempt: bool = False,
    repeated_actions: int = 0,
) -> RewardBreakdown:
    verdict = action.verdict
    failure = action.failure_type or FailureType.unknown
    verdict_score = 0.0
    if verdict == gold.gold_verdict:
        verdict_score += 1.5
    elif _family(verdict) == _family(gold.gold_verdict):
        verdict_score += 0.5
    if _family(verdict) == "valid" and _family(gold.gold_verdict) == "invalid":
        verdict_score -= 1.0
    if _family(verdict) == "invalid" and _family(gold.gold_verdict) == "valid":
        verdict_score -= 0.7

    failure_score = 0.0
    if failure == gold.gold_failure_type:
        failure_score += 1.0
    elif failure != FailureType.unknown and failure != FailureType.none:
        if _family(gold.gold_verdict) == "invalid":
            failure_score += 0.3
        else:
            failure_score -= 0.5

    existing = {e.id: e for e in state.evidence}
    source_by_evidence = {e.id: e.source_id for e in state.evidence}
    fabricated = [eid for eid in action.evidence_ids if eid not in existing]
    unobserved = [eid for eid in action.evidence_ids if eid in existing and not existing[eid].observed]
    overlap_sources = {
        source_by_evidence[eid]
        for eid in action.evidence_ids
        if eid in source_by_evidence and source_by_evidence[eid] in set(gold.gold_evidence_source_ids)
    }
    evidence_score = 0.0
    if overlap_sources:
        evidence_score += 0.6
    if action.evidence_ids and not fabricated and not unobserved:
        evidence_score += 0.4
    evidence_score -= 1.0 * len(fabricated)
    evidence_score -= 0.5 * len(unobserved)

    ran_checks = {c.check_name for c in state.checks}
    checker_score = 0.0
    if set(gold.gold_required_checks) & ran_checks:
        checker_score += 0.3
    if ran_checks:
        checker_score += 0.2
    else:
        checker_score -= 0.2

    reproduction_score = 0.0
    repro = [c for c in state.checks if c.check_name == CheckName.reproduction_check]
    if repro:
        if gold.gold_failure_type in {FailureType.result_mismatch, FailureType.missing_artifact, FailureType.none}:
            reproduction_score += 0.5
        if gold.gold_metric_value is not None:
            reproduction_score += 0.3

    novelty_score = 0.0
    if state.target_claim.novelty_level >= 4 and gold.gold_verdict in VALID_GOLD:
        if verdict == ValidationVerdict.PLAUSIBLY_VALIDATED_NOVEL_METHOD:
            novelty_score += 0.6
        elif _family(verdict) == "invalid":
            novelty_score -= 0.7
    if gold.gold_failure_type == FailureType.missing_artifact and _family(verdict) == "inconclusive":
        novelty_score += 0.5

    efficiency = 0.1 if state.steps_remaining >= 3 and verdict is not None else 0.0
    efficiency -= 0.05 * repeated_actions
    anti = 0.0
    if hidden_access_attempt:
        anti -= 1.0
    if fabricated:
        anti -= 0.7
    if verdict is None:
        anti -= 0.5

    total = (
        verdict_score
        + failure_score
        + evidence_score
        + checker_score
        + reproduction_score
        + novelty_score
        + efficiency
        + anti
    )
    return RewardBreakdown(
        format_valid=0.2,
        verdict_correctness=verdict_score,
        failure_type=failure_score,
        evidence_grounding=evidence_score,
        checker_usage=checker_score,
        reproduction=reproduction_score,
        novelty_calibration=novelty_score,
        efficiency=efficiency,
        anti_hallucination=anti,
        final=total,
    )


def shaping_reward(*, valid: bool, repeated: bool = False, relevant: bool = False, hidden_access: bool = False) -> RewardBreakdown:
    score = 0.05 if valid else -0.3
    if relevant:
        score += 0.15
    if repeated:
        score -= 0.1
    if hidden_access:
        score -= 0.3
    return RewardBreakdown(format_valid=0.2 if valid else -0.3, shaping=score, anti_hallucination=-1.0 if hidden_access else 0.0, final=score)
