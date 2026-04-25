"""Deterministic methodology checkers for ReproPilot."""

from __future__ import annotations

import re
from uuid import uuid4

try:
    from .models import (
        AuditState,
        CheckName,
        CheckStatus,
        EvidenceItem,
        FailureType,
        Severity,
        ValidationCheck,
    )
except ImportError:
    from models import (
        AuditState,
        CheckName,
        CheckStatus,
        EvidenceItem,
        FailureType,
        Severity,
        ValidationCheck,
    )


def _norm_metric(value: str | None) -> str:
    v = (value or "").lower().replace("-", "").replace("_", "").replace(" ", "")
    if v in {"macrof1", "f1macro", "f1score", "f1"}:
        return "f1"
    if v in {"acc", "accuracy"}:
        return "accuracy"
    if v in {"auroc", "rocauc", "auc"}:
        return "auc"
    return v


def _source_texts(state: AuditState) -> list[tuple[str, str, str | None, bool]]:
    rows: list[tuple[str, str, str | None, bool]] = []
    for f in state.repo_files:
        rows.append((f.id, f.content, f.path, f.inspected))
    for s in state.paper_sections:
        rows.append((s.id, s.text, s.title, s.inspected))
    for c in state.configs:
        rows.append((c.id, f"{c.dataset} {c.split} {c.metric} {c.hyperparameters}", c.path, c.inspected))
    for l in state.logs:
        rows.append((l.id, f"{l.metric_name} {l.values} {l.split} {l.seed_values}", l.path, l.inspected))
    return rows


def _add_evidence(
    state: AuditState,
    source_id: str,
    source_path: str | None,
    finding: str,
    failure: FailureType,
    *,
    observed: bool,
) -> EvidenceItem:
    eid = f"ev_{failure.value}_{source_id}_{len(state.evidence)+1}"
    item = EvidenceItem(
        id=eid,
        source_id=source_id,
        source_path=source_path,
        quote_or_finding=finding,
        supports_failure_type=failure,
        observed=observed,
    )
    state.evidence.append(item)
    return item


def _record(
    state: AuditState,
    check_name: CheckName,
    status: CheckStatus,
    failure: FailureType,
    severity: Severity,
    message: str,
    evidence_ids: list[str],
) -> ValidationCheck:
    check = ValidationCheck(
        id=f"check_{check_name.value}_{uuid4().hex[:8]}",
        check_name=check_name,
        status=status,
        issue_found=status == CheckStatus.failed,
        failure_type=failure,
        severity=severity,
        evidence_ids=evidence_ids,
        message=message,
    )
    state.checks.append(check)
    state.last_action_result = message
    if status == CheckStatus.failed:
        state.claim_risk = min(100, state.claim_risk + 15)
    elif status == CheckStatus.passed:
        state.evidence_confidence = min(100, state.evidence_confidence + 12)
    return check


def metric_check(state: AuditState) -> ValidationCheck:
    claim_metric = _norm_metric(state.target_claim.claimed_metric_name)
    detected: tuple[str, str, str | None, bool] | None = None
    for sid, text, path, inspected in _source_texts(state):
        for raw in ("macro_f1", "macro-F1", "f1", "accuracy", "auc", "auroc"):
            if raw.lower() in text.lower():
                detected = (_norm_metric(raw), sid, path, inspected)
                break
        if detected:
            break
    if claim_metric and detected and claim_metric != detected[0]:
        ev = _add_evidence(
            state,
            detected[1],
            detected[2],
            f"Claim metric {claim_metric} differs from artifact metric {detected[0]}.",
            FailureType.metric_mismatch,
            observed=detected[3] or True,
        )
        return _record(state, CheckName.metric_check, CheckStatus.failed, FailureType.metric_mismatch, Severity.high, ev.quote_or_finding, [ev.id])
    return _record(state, CheckName.metric_check, CheckStatus.passed, FailureType.none, Severity.low, "Metric check passed or no conflicting metric found.", [])


def split_check(state: AuditState) -> ValidationCheck:
    claim_split = (state.target_claim.claimed_split or "").lower()
    for sid, text, path, inspected in _source_texts(state):
        t = text.lower()
        observed_split = None
        if "validation" in t or 'split="val"' in t or "split='val'" in t or "valid" in t:
            observed_split = "validation"
        elif "train" in t and "split" in t:
            observed_split = "train"
        elif "test" in t and "split" in t:
            observed_split = "test"
        if claim_split == "test" and observed_split in {"validation", "train"}:
            ev = _add_evidence(state, sid, path, f"Claim uses test split but artifact uses {observed_split}.", FailureType.split_mismatch, observed=inspected or True)
            return _record(state, CheckName.split_check, CheckStatus.failed, FailureType.split_mismatch, Severity.critical, ev.quote_or_finding, [ev.id])
    return _record(state, CheckName.split_check, CheckStatus.passed, FailureType.none, Severity.low, "Split check passed or no conflicting split found.", [])


def leakage_check(state: AuditState) -> ValidationCheck:
    patterns = [
        "train + test",
        "test + train",
        "concat([train, test",
        "concat([test, train",
        "fit(all_data",
        "fit(full_data",
        "fit(dataset)",
        "fit_transform(full",
        "scaler.fit(x)",
        "normalize(all",
        "use_test_in_training",
    ]
    for sid, text, path, inspected in _source_texts(state):
        low = text.lower().replace(" ", "")
        for pat in patterns:
            if pat.lower().replace(" ", "") in low:
                ev = _add_evidence(state, sid, path, f"Potential train/test leakage pattern found: {pat}.", FailureType.data_leakage, observed=inspected or True)
                return _record(state, CheckName.leakage_check, CheckStatus.failed, FailureType.data_leakage, Severity.critical, ev.quote_or_finding, [ev.id])
    return _record(state, CheckName.leakage_check, CheckStatus.passed, FailureType.none, Severity.low, "Leakage check passed.", [])


def seed_check(state: AuditState, tolerance: float = 0.003) -> ValidationCheck:
    claimed = state.target_claim.claimed_metric_value
    if claimed is None:
        return _record(state, CheckName.seed_check, CheckStatus.inconclusive, FailureType.unknown, Severity.low, "No claimed metric value to compare against seeds.", [])
    for log in state.logs:
        if len(log.seed_values) < 2:
            continue
        values = list(log.seed_values.values())
        best = max(values)
        mean = sum(values) / len(values)
        if abs(claimed - best) <= tolerance and abs(claimed - mean) > tolerance:
            ev = _add_evidence(state, log.id, log.path, f"Claimed value {claimed:.3f} matches best seed {best:.3f}, while seed mean is {mean:.3f}.", FailureType.cherry_picked_seed, observed=log.inspected or True)
            return _record(state, CheckName.seed_check, CheckStatus.failed, FailureType.cherry_picked_seed, Severity.high, ev.quote_or_finding, [ev.id])
    return _record(state, CheckName.seed_check, CheckStatus.passed, FailureType.none, Severity.low, "Seed check passed or insufficient seed spread.", [])


def ablation_check(state: AuditState) -> ValidationCheck:
    text = " ".join(t for _, t, _, _ in _source_texts(state)).lower()
    if "ablation" in text and ("changed lr and removed module" in text or "dropout,lr,batch_size" in text or "multiple changes" in text):
        ev = _add_evidence(state, state.target_claim.source_section_id or state.target_claim.id, None, "Ablation changes multiple factors at once.", FailureType.invalid_ablation, observed=True)
        return _record(state, CheckName.ablation_check, CheckStatus.failed, FailureType.invalid_ablation, Severity.high, ev.quote_or_finding, [ev.id])
    return _record(state, CheckName.ablation_check, CheckStatus.passed, FailureType.none, Severity.low, "Ablation check passed or no ablation claim found.", [])


def paper_code_consistency_check(state: AuditState) -> ValidationCheck:
    method = (state.target_claim.claimed_method or "").lower()
    tokens = [t for t in re.split(r"[^a-z0-9]+", method) if len(t) >= 4]
    code = " ".join(f.content.lower() for f in state.repo_files if f.artifact_type.value in {"code_file", "script"})
    missing = [t for t in tokens if t not in code]
    if tokens and len(missing) >= max(1, len(tokens) // 2):
        ev = _add_evidence(state, "repo", None, f"Method tokens missing from code: {', '.join(missing)}.", FailureType.paper_code_mismatch, observed=True)
        return _record(state, CheckName.paper_code_consistency_check, CheckStatus.failed, FailureType.paper_code_mismatch, Severity.medium, ev.quote_or_finding, [ev.id])
    return _record(state, CheckName.paper_code_consistency_check, CheckStatus.passed, FailureType.none, Severity.low, "Paper/code consistency check passed.", [])


def reproduction_check(state: AuditState, tolerance: float = 0.005) -> ValidationCheck:
    claim_metric = _norm_metric(state.target_claim.claimed_metric_name)
    claimed = state.target_claim.claimed_metric_value
    matching_logs = [l for l in state.logs if _norm_metric(l.metric_name) == claim_metric and l.values]
    if not matching_logs:
        ev = _add_evidence(state, "artifacts", None, "No matching logs or result artifacts for reproduction check.", FailureType.missing_artifact, observed=True)
        return _record(state, CheckName.reproduction_check, CheckStatus.inconclusive, FailureType.missing_artifact, Severity.medium, ev.quote_or_finding, [ev.id])
    if claimed is not None:
        observed = matching_logs[0].values[-1]
        if abs(claimed - observed) > tolerance:
            ev = _add_evidence(state, matching_logs[0].id, matching_logs[0].path, f"Claimed {claimed:.3f}, but reproduced/logged value is {observed:.3f}.", FailureType.result_mismatch, observed=matching_logs[0].inspected or True)
            return _record(state, CheckName.reproduction_check, CheckStatus.failed, FailureType.result_mismatch, Severity.high, ev.quote_or_finding, [ev.id])
    return _record(state, CheckName.reproduction_check, CheckStatus.passed, FailureType.none, Severity.low, "Reproduction check passed within tolerance.", [])
