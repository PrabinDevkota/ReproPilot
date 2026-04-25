"""Pydantic models for ReproPilot research-claim validation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

try:
    from openenv.core.env_server.types import Action as _OpenEnvAction
    from openenv.core.env_server.types import Observation as _OpenEnvObservation
except Exception:
    _OpenEnvAction = BaseModel  # type: ignore[assignment]
    _OpenEnvObservation = BaseModel  # type: ignore[assignment]


def _is_pydantic_model_class(cls: object) -> bool:
    try:
        return isinstance(cls, type) and issubclass(cls, BaseModel)
    except TypeError:
        return False


ActionBase = _OpenEnvAction if _is_pydantic_model_class(_OpenEnvAction) else BaseModel
ObservationBase = _OpenEnvObservation if _is_pydantic_model_class(_OpenEnvObservation) else BaseModel


class ValidationVerdict(StrEnum):
    SUPPORTED_RESULT_AND_METHOD = "SUPPORTED_RESULT_AND_METHOD"
    PLAUSIBLY_VALIDATED_NOVEL_METHOD = "PLAUSIBLY_VALIDATED_NOVEL_METHOD"
    NOT_SUPPORTED_RESULT_MISMATCH = "NOT_SUPPORTED_RESULT_MISMATCH"
    NOT_SUPPORTED_METHOD_INVALID = "NOT_SUPPORTED_METHOD_INVALID"
    INCONCLUSIVE_MISSING_ARTIFACTS = "INCONCLUSIVE_MISSING_ARTIFACTS"
    INCONCLUSIVE_NEEDS_DOMAIN_REVIEW = "INCONCLUSIVE_NEEDS_DOMAIN_REVIEW"
    NOT_ENOUGH_EVIDENCE = "NOT_ENOUGH_EVIDENCE"


class FailureType(StrEnum):
    none = "none"
    metric_mismatch = "metric_mismatch"
    split_mismatch = "split_mismatch"
    data_leakage = "data_leakage"
    cherry_picked_seed = "cherry_picked_seed"
    paper_code_mismatch = "paper_code_mismatch"
    invalid_ablation = "invalid_ablation"
    result_mismatch = "result_mismatch"
    missing_artifact = "missing_artifact"
    ambiguous_method = "ambiguous_method"
    unsupported_claim = "unsupported_claim"
    unknown = "unknown"


class ClaimType(StrEnum):
    metric_result = "metric_result"
    methodology_validity = "methodology_validity"
    ablation_claim = "ablation_claim"
    dataset_claim = "dataset_claim"
    theoretical_claim = "theoretical_claim"
    reproducibility_claim = "reproducibility_claim"


class ArtifactType(StrEnum):
    paper_section = "paper_section"
    code_file = "code_file"
    config_file = "config_file"
    log_file = "log_file"
    result_table = "result_table"
    dataset_card = "dataset_card"
    checkpoint = "checkpoint"
    script = "script"
    unknown = "unknown"


class CheckName(StrEnum):
    metric_check = "metric_check"
    split_check = "split_check"
    leakage_check = "leakage_check"
    seed_check = "seed_check"
    ablation_check = "ablation_check"
    reproduction_check = "reproduction_check"
    paper_code_consistency_check = "paper_code_consistency_check"


class CheckStatus(StrEnum):
    not_run = "not_run"
    passed = "passed"
    failed = "failed"
    inconclusive = "inconclusive"


class Severity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ActionType(StrEnum):
    read_claim = "read_claim"
    inspect_paper_section = "inspect_paper_section"
    inspect_code_file = "inspect_code_file"
    inspect_config = "inspect_config"
    inspect_logs = "inspect_logs"
    search_artifacts = "search_artifacts"
    run_metric_check = "run_metric_check"
    run_split_check = "run_split_check"
    run_leakage_check = "run_leakage_check"
    run_seed_check = "run_seed_check"
    run_ablation_check = "run_ablation_check"
    run_paper_code_consistency_check = "run_paper_code_consistency_check"
    run_reproduction_check = "run_reproduction_check"
    mark_inconclusive = "mark_inconclusive"
    submit_verdict = "submit_verdict"
    do_nothing = "do_nothing"


class ResearchClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    claim_type: ClaimType
    claimed_metric_name: str | None = None
    claimed_metric_value: float | None = None
    claimed_split: str | None = None
    claimed_dataset: str | None = None
    claimed_method: str | None = None
    source_section_id: str | None = None
    novelty_level: int = Field(default=0, ge=0, le=5)


class PaperSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    text: str
    section_type: str = "unknown"
    inspected: bool = False


class RepoFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: str
    artifact_type: ArtifactType
    content: str
    inspected: bool = False


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: str
    dataset: str | None = None
    split: str | None = None
    metric: str | None = None
    seed: int | None = None
    hyperparameters: dict[str, str] = Field(default_factory=dict)
    inspected: bool = False


class ExperimentLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    path: str
    metric_name: str | None = None
    values: list[float] = Field(default_factory=list)
    split: str | None = None
    seed_values: dict[str, float] = Field(default_factory=dict)
    inspected: bool = False


class ValidationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    check_name: CheckName
    status: CheckStatus = CheckStatus.not_run
    issue_found: bool = False
    failure_type: FailureType = FailureType.none
    severity: Severity = Severity.low
    evidence_ids: list[str] = Field(default_factory=list)
    message: str | None = None


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source_id: str
    source_path: str | None = None
    quote_or_finding: str
    supports_failure_type: FailureType = FailureType.unknown
    observed: bool = False


class AuditState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    step: int = 0
    steps_remaining: int = Field(default=12, ge=0)
    target_claim: ResearchClaim
    paper_sections: list[PaperSection] = Field(default_factory=list)
    repo_files: list[RepoFile] = Field(default_factory=list)
    configs: list[ExperimentConfig] = Field(default_factory=list)
    logs: list[ExperimentLog] = Field(default_factory=list)
    checks: list[ValidationCheck] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    final_verdict: ValidationVerdict | None = None
    final_failure_type: FailureType | None = None
    claim_risk: int = Field(default=50, ge=0, le=100)
    evidence_confidence: int = Field(default=0, ge=0, le=100)
    last_action_result: str | None = None
    action_history: list[dict[str, Any]] = Field(default_factory=list)
    episode_active: bool = True


class HiddenGold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold_verdict: ValidationVerdict
    gold_failure_type: FailureType
    gold_evidence_source_ids: list[str] = Field(default_factory=list)
    gold_required_checks: list[CheckName] = Field(default_factory=list)
    gold_metric_name: str | None = None
    gold_metric_value: float | None = None
    notes: str | None = None


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    split: str
    state: AuditState
    hidden_gold: HiddenGold
    tags: list[str] = Field(default_factory=list)


class AgentAction(ActionBase):
    action_type: ActionType = ActionType.do_nothing
    target_id: str | None = None
    secondary_id: str | None = None
    verdict: ValidationVerdict | None = None
    failure_type: FailureType | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str | None = None
    generated_code: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _default_action_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and "action_type" not in data:
            data = {**data, "action_type": "do_nothing"}
        return data


class ReproPilotObservation(ObservationBase):
    done: bool = False
    reward: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    echoed_message: str = ""
    message_length: int = 0


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_valid: float = 0.0
    verdict_correctness: float = 0.0
    failure_type: float = 0.0
    evidence_grounding: float = 0.0
    checker_usage: float = 0.0
    reproduction: float = 0.0
    novelty_calibration: float = 0.0
    efficiency: float = 0.0
    anti_hallucination: float = 0.0
    shaping: float = 0.0
    final: float = 0.0


WorldState = AuditState
