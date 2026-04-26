"""Human-readable ReproPilot audit briefings."""

from __future__ import annotations

try:
    from .models import AuditState, FailureType, ValidationVerdict
except ImportError:
    from models import AuditState, FailureType, ValidationVerdict


def _preview(text: str, limit: int = 130) -> str:
    compact = " ".join((text or "").split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "..."


def build_briefing(state: AuditState) -> str:
    claim = state.target_claim
    section_lines = [
        f"- {s.id}: {s.title} [{s.section_type}] inspected={s.inspected} :: {_preview(s.text)}"
        for s in state.paper_sections
    ]
    file_lines = [
        f"- {f.id}: {f.path} [{f.artifact_type}] inspected={f.inspected} :: {_preview(f.content)}"
        for f in state.repo_files
    ]
    config_lines = [
        f"- {c.id}: {c.path} dataset={c.dataset or '?'} split={c.split or '?'} "
        f"metric={c.metric or '?'} seed={c.seed if c.seed is not None else '?'} inspected={c.inspected}"
        for c in state.configs
    ]
    log_lines = [
        f"- {log.id}: {log.path} metric={log.metric_name or '?'} values={len(log.values)} "
        f"split={log.split or '?'} seeds={len(log.seed_values)} inspected={log.inspected}"
        for log in state.logs
    ]
    check_lines = [
        f"- {c.id}: {c.check_name} status={c.status} issue_found={c.issue_found} "
        f"failure_type={c.failure_type} severity={c.severity} message={c.message or ''}"
        for c in state.checks
    ]
    evidence_lines = [
        f"- {e.id}: source={e.source_id} observed={e.observed} "
        f"failure={e.supports_failure_type} :: {_preview(e.quote_or_finding, 120)}"
        for e in state.evidence
        if e.observed
    ]

    return "\n".join(
        [
            "REPROPILOT RESEARCH AUDIT BRIEFING",
            "",
            f"Scenario: {state.scenario_id}",
            f"Steps remaining: {state.steps_remaining}",
            f"Claim risk: {state.claim_risk}/100",
            f"Evidence confidence: {state.evidence_confidence}/100",
            "",
            "TARGET CLAIM",
            f"- ID: {claim.id}",
            f"- Text: {claim.text}",
            f"- Type: {claim.claim_type}",
            f"- Claimed metric: {claim.claimed_metric_name or '?'} = {claim.claimed_metric_value if claim.claimed_metric_value is not None else '?'}",
            f"- Claimed dataset: {claim.claimed_dataset or '?'}",
            f"- Claimed split: {claim.claimed_split or '?'}",
            f"- Claimed method: {claim.claimed_method or '?'}",
            f"- Novelty level: {claim.novelty_level}/5",
            "",
            "PAPER SECTIONS",
            "\n".join(section_lines) if section_lines else "(none)",
            "",
            "REPOSITORY / ARTIFACTS",
            "\n".join(file_lines) if file_lines else "(none)",
            "",
            "CONFIGS",
            "\n".join(config_lines) if config_lines else "(none)",
            "",
            "LOGS / RESULT TABLES",
            "\n".join(log_lines) if log_lines else "(none)",
            "",
            "VALIDATION CHECKS",
            "\n".join(check_lines) if check_lines else "(none run yet)",
            "",
            "EVIDENCE COLLECTED",
            "\n".join(evidence_lines) if evidence_lines else "(none observed yet)",
            "",
            "LAST ACTION RESULT",
            state.last_action_result or "None",
            "",
            "LEGAL ACTIONS",
            'Return exactly one JSON object: {"action_type":"...","target_id":"...","secondary_id":"...","verdict":"...","failure_type":"...","evidence_ids":["..."],"explanation":"..."}',
            "",
            "VALID VERDICTS",
            "\n".join(f"- {v.value}" for v in ValidationVerdict),
            "",
            "VALID FAILURE TYPES",
            "\n".join(f"- {v.value}" for v in FailureType),
            "",
            "Important: Do not submit a verdict without inspecting evidence. Novel methods should not be rejected merely for being unfamiliar. If available checks pass but the method is novel, use PLAUSIBLY_VALIDATED_NOVEL_METHOD rather than claiming it is scientifically proven.",
        ]
    )
