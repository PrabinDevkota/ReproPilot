"""ReproPilot OpenEnv environment: research claim auditing over synthetic artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except Exception:
    class Environment:  # type: ignore[no-redef]
        pass

    class State:  # type: ignore[no-redef]
        def __init__(self, episode_id: str | None = None, step_count: int = 0) -> None:
            self.episode_id = episode_id
            self.step_count = step_count

try:
    from ..briefing import build_briefing
    from ..checkers import (
        ablation_check,
        leakage_check,
        metric_check,
        paper_code_consistency_check,
        reproduction_check,
        seed_check,
        split_check,
    )
    from ..models import (
        ActionType,
        AgentAction,
        ArtifactType,
        AuditState,
        CheckName,
        FailureType,
        HiddenGold,
        RepoFile,
        ReproPilotObservation,
        RewardBreakdown,
        Scenario,
        ValidationVerdict,
    )
    from ..rewards import action_shaping_reward, compute_terminal_reward, shaping_reward
except ImportError:
    from briefing import build_briefing
    from checkers import (
        ablation_check,
        leakage_check,
        metric_check,
        paper_code_consistency_check,
        reproduction_check,
        seed_check,
        split_check,
    )
    from models import (
        ActionType,
        AgentAction,
        ArtifactType,
        AuditState,
        CheckName,
        FailureType,
        HiddenGold,
        RepoFile,
        ReproPilotObservation,
        RewardBreakdown,
        Scenario,
        ValidationVerdict,
    )
    from rewards import action_shaping_reward, compute_terminal_reward, shaping_reward


def _default_scenario_path() -> Path:
    root = Path(__file__).resolve().parents[1] / "scenarios" / "train"
    preferred = root / "split_mismatch_test_vs_val_001.json"
    if preferred.exists():
        return preferred
    found = sorted(root.glob("*.json"))
    return found[0] if found else Path(__file__).resolve().parents[1] / "scenarios" / "phase2_core.json"


class ReproPilotEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, scenario_path: str | Path | None = None) -> None:
        self._scenario_path = Path(scenario_path) if scenario_path else _default_scenario_path()
        self._scenario: Scenario | None = None
        self._state_obj = State(episode_id=str(uuid4()), step_count=0)
        self._last_reward_breakdown: RewardBreakdown | None = None
        self._hidden_access_attempt = False

    @staticmethod
    def load_scenario_from_json(path: str | Path) -> Scenario:
        return Scenario.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @property
    def audit_state(self) -> AuditState:
        if self._scenario is None:
            raise RuntimeError("Environment not reset.")
        return self._scenario.state

    @property
    def hidden_gold(self) -> HiddenGold:
        if self._scenario is None:
            raise RuntimeError("Environment not reset.")
        return self._scenario.hidden_gold

    @property
    def state(self) -> State:
        return self._state_obj

    def reset(self) -> ReproPilotObservation:  # type: ignore[override]
        self._scenario = self.load_scenario_from_json(self._scenario_path)
        self._scenario.state.step = 0
        self._scenario.state.episode_active = True
        self._scenario.state.final_verdict = None
        self._scenario.state.final_failure_type = None
        self._scenario.state.last_action_result = "Audit started."
        self._state_obj = State(episode_id=str(uuid4()), step_count=0)
        self._hidden_access_attempt = False
        self._last_reward_breakdown = None
        return self._observation(0.0, False)

    def step(self, action: AgentAction) -> ReproPilotObservation:  # type: ignore[override]
        if self._scenario is None:
            self.reset()
        st = self.audit_state
        if not st.episode_active:
            return self._observation(-0.2, True)

        self._state_obj.step_count += 1
        st.step += 1
        st.steps_remaining = max(0, st.steps_remaining - 1)
        action_dict = action.model_dump(mode="json")
        st.action_history.append(action_dict)

        hidden_access = self._is_hidden_access(action)
        self._hidden_access_attempt = self._hidden_access_attempt or hidden_access
        repeated = st.action_history.count(action_dict) > 1

        if hidden_access:
            st.last_action_result = "Rejected: attempted access to hidden/gold answer fields."
            bd = action_shaping_reward(st, self.hidden_gold, action, valid=False, repeated=repeated, hidden_access=True)
            self._last_reward_breakdown = bd
            return self._maybe_timeout_or_observation(bd)

        at = action.action_type
        relevant = False
        valid = True
        if at == ActionType.read_claim:
            relevant = self._read_claim()
        elif at == ActionType.inspect_paper_section:
            valid, relevant = self._inspect_paper_section(action.target_id)
        elif at == ActionType.inspect_code_file:
            valid, relevant = self._inspect_file(action.target_id, {ArtifactType.code_file, ArtifactType.script})
        elif at == ActionType.inspect_config:
            valid, relevant = self._inspect_config(action.target_id)
        elif at == ActionType.inspect_logs:
            valid, relevant = self._inspect_log(action.target_id)
        elif at == ActionType.inspect_result_table:
            valid, relevant = self._inspect_file(action.target_id, {ArtifactType.result_table})
        elif at == ActionType.inspect_dataset_card:
            valid, relevant = self._inspect_file(action.target_id, {ArtifactType.dataset_card})
        elif at == ActionType.inspect_checkpoint:
            valid, relevant = self._inspect_file(action.target_id, {ArtifactType.checkpoint})
        elif at == ActionType.search_artifacts:
            valid, relevant = self._search(action.target_id or action.explanation or "")
        elif at == ActionType.compare_claim_to_artifacts:
            valid, relevant = self._compare_claim_to_artifacts()
        elif at == ActionType.run_metric_check:
            metric_check(st)
            relevant = CheckName.metric_check in self.hidden_gold.gold_required_checks
        elif at == ActionType.run_split_check:
            split_check(st)
            relevant = CheckName.split_check in self.hidden_gold.gold_required_checks
        elif at == ActionType.run_leakage_check:
            leakage_check(st)
            relevant = CheckName.leakage_check in self.hidden_gold.gold_required_checks
        elif at == ActionType.run_seed_check:
            seed_check(st)
            relevant = CheckName.seed_check in self.hidden_gold.gold_required_checks
        elif at == ActionType.run_ablation_check:
            ablation_check(st)
            relevant = CheckName.ablation_check in self.hidden_gold.gold_required_checks
        elif at == ActionType.run_paper_code_consistency_check:
            paper_code_consistency_check(st)
            relevant = CheckName.paper_code_consistency_check in self.hidden_gold.gold_required_checks
        elif at == ActionType.run_reproduction_check:
            reproduction_check(st)
            relevant = CheckName.reproduction_check in self.hidden_gold.gold_required_checks
        elif at == ActionType.synthesize_findings:
            valid, relevant = self._synthesize_findings()
        elif at == ActionType.mark_inconclusive:
            st.last_action_result = "Marked current audit as inconclusive pending more artifacts."
            relevant = self.hidden_gold.gold_failure_type in {FailureType.missing_artifact, FailureType.ambiguous_method}
        elif at == ActionType.submit_verdict:
            return self._submit_verdict(action, repeated)
        elif at == ActionType.do_nothing:
            st.last_action_result = "No action taken."
        else:
            valid = False
            st.last_action_result = f"Unsupported action: {at}"

        bd = action_shaping_reward(st, self.hidden_gold, action, valid=valid, repeated=repeated, relevant=relevant)
        self._last_reward_breakdown = bd
        return self._maybe_timeout_or_observation(bd)

    def _maybe_timeout_or_observation(self, bd: RewardBreakdown) -> ReproPilotObservation:
        st = self.audit_state
        if st.steps_remaining <= 0 and st.final_verdict is None:
            st.episode_active = False
            timeout_bd = bd.model_copy(update={"efficiency": -0.5, "final": bd.final - 0.5})
            self._last_reward_breakdown = timeout_bd
            st.last_action_result = (st.last_action_result or "") + " Episode timed out without verdict."
            return self._observation(timeout_bd.final, True)
        return self._observation(bd.final, False)

    def _submit_verdict(self, action: AgentAction, repeated: bool) -> ReproPilotObservation:
        st = self.audit_state
        st.final_verdict = action.verdict
        st.final_failure_type = action.failure_type
        st.episode_active = False
        st.last_action_result = f"Submitted verdict={action.verdict} failure_type={action.failure_type}."
        bd = compute_terminal_reward(
            st,
            self.hidden_gold,
            action,
            hidden_access_attempt=self._hidden_access_attempt,
            repeated_actions=1 if repeated else 0,
        )
        self._last_reward_breakdown = bd
        return self._observation(bd.final, True)

    def _read_claim(self) -> bool:
        st = self.audit_state
        already = any(a.get("action_type") == ActionType.read_claim.value for a in st.action_history[:-1])
        st.last_action_result = f"Claim details: {st.target_claim.text}"
        st.evidence_confidence = min(100, st.evidence_confidence + (5 if not already else 0))
        return not already

    def _inspect_paper_section(self, sid: str | None) -> tuple[bool, bool]:
        for i, sec in enumerate(self.audit_state.paper_sections):
            if sec.id == sid:
                already = sec.inspected
                self.audit_state.paper_sections[i] = sec.model_copy(update={"inspected": True})
                self.audit_state.last_action_result = f"Inspected paper section {sid}: {sec.text[:600]}"
                return True, not already
        self.audit_state.last_action_result = f"Invalid paper section id: {sid}"
        return False, False

    def _inspect_file(self, fid: str | None, allowed: set[ArtifactType]) -> tuple[bool, bool]:
        for i, file in enumerate(self.audit_state.repo_files):
            if file.id == fid:
                if file.artifact_type not in allowed:
                    self.audit_state.last_action_result = f"File {fid} is not an allowed artifact type for this action."
                    return False, False
                already = file.inspected
                self.audit_state.repo_files[i] = file.model_copy(update={"inspected": True})
                self.audit_state.last_action_result = f"Inspected {file.path}: {file.content[:900]}"
                return True, not already
        self.audit_state.last_action_result = f"Invalid repository file id: {fid}"
        return False, False

    def _inspect_config(self, cid: str | None) -> tuple[bool, bool]:
        for i, cfg in enumerate(self.audit_state.configs):
            if cfg.id == cid:
                already = cfg.inspected
                self.audit_state.configs[i] = cfg.model_copy(update={"inspected": True})
                self.audit_state.last_action_result = f"Config {cid}: dataset={cfg.dataset}, split={cfg.split}, metric={cfg.metric}, seed={cfg.seed}, hyperparameters={cfg.hyperparameters}"
                return True, not already
        self.audit_state.last_action_result = f"Invalid config id: {cid}"
        return False, False

    def _inspect_log(self, lid: str | None) -> tuple[bool, bool]:
        for i, log in enumerate(self.audit_state.logs):
            if log.id == lid:
                already = log.inspected
                self.audit_state.logs[i] = log.model_copy(update={"inspected": True})
                self.audit_state.last_action_result = f"Log {lid}: metric={log.metric_name}, values={log.values}, split={log.split}, seed_values={log.seed_values}"
                return True, not already
        self.audit_state.last_action_result = f"Invalid log id: {lid}"
        return False, False

    def _search(self, query: str) -> tuple[bool, bool]:
        q = query.lower().strip()
        if not q:
            self.audit_state.last_action_result = "Search requires a non-empty query."
            return False, False
        hits: list[str] = []
        for sec in self.audit_state.paper_sections:
            if q in sec.text.lower() or q in sec.title.lower():
                hits.append(f"{sec.id}:{sec.title}")
        for file in self.audit_state.repo_files:
            if q in file.content.lower() or q in file.path.lower():
                hits.append(f"{file.id}:{file.path}")
        self.audit_state.last_action_result = "Search hits: " + (", ".join(hits[:12]) if hits else "(none)")
        return True, bool(hits)

    def _compare_claim_to_artifacts(self) -> tuple[bool, bool]:
        st = self.audit_state
        before = len(st.checks)
        ran: list[str] = []
        if st.target_claim.claimed_metric_name:
            metric_check(st)
            ran.append(CheckName.metric_check.value)
        if st.target_claim.claimed_split:
            split_check(st)
            ran.append(CheckName.split_check.value)
        if st.logs:
            reproduction_check(st)
            ran.append(CheckName.reproduction_check.value)
        if st.target_claim.claimed_method:
            paper_code_consistency_check(st)
            ran.append(CheckName.paper_code_consistency_check.value)
        if not ran:
            st.last_action_result = "No comparable claim fields or artifacts were available."
            return False, False
        relevant = bool(set(self.hidden_gold.gold_required_checks) & {c.check_name for c in st.checks[before:]})
        st.last_action_result = "Compared claim to artifacts using: " + ", ".join(ran)
        return True, relevant

    def _synthesize_findings(self) -> tuple[bool, bool]:
        st = self.audit_state
        failed = [c for c in st.checks if c.issue_found]
        observed = [e for e in st.evidence if e.observed]
        if failed:
            st.last_action_result = "Synthesis: failing checks found: " + ", ".join(
                f"{c.check_name.value}:{c.failure_type.value}" for c in failed[-4:]
            )
            return True, True
        if observed or st.checks:
            st.last_action_result = f"Synthesis: {len(st.checks)} checks run, {len(observed)} evidence items observed, no failing check yet."
            return True, True
        st.last_action_result = "Synthesis needs at least one inspection, check, or evidence item."
        return True, False

    def _is_hidden_access(self, action: AgentAction) -> bool:
        hay = " ".join(
            str(x or "")
            for x in [action.target_id, action.secondary_id, action.explanation, action.generated_code]
        ).lower()
        return any(token in hay for token in ("hidden", "gold", "answer", "label"))

    def _metadata(self) -> dict[str, Any]:
        st = self.audit_state
        checks = [c.model_dump(mode="json") for c in st.checks]
        observed_evidence = [e.model_dump(mode="json") for e in st.evidence if e.observed]
        return {
            "scenario_id": st.scenario_id,
            "episode_step": st.step,
            "steps_remaining": st.steps_remaining,
            "claim_id": st.target_claim.id,
            "claim_type": st.target_claim.claim_type.value,
            "claim_metric": st.target_claim.claimed_metric_name,
            "claim_split": st.target_claim.claimed_split,
            "novelty_level": st.target_claim.novelty_level,
            "paper_section_ids": [s.id for s in st.paper_sections],
            "code_file_ids": [f.id for f in st.repo_files if f.artifact_type in {ArtifactType.code_file, ArtifactType.script}],
            "config_ids": [c.id for c in st.configs],
            "log_ids": [l.id for l in st.logs],
            "result_table_ids": [f.id for f in st.repo_files if f.artifact_type == ArtifactType.result_table],
            "dataset_card_ids": [f.id for f in st.repo_files if f.artifact_type == ArtifactType.dataset_card],
            "checkpoint_ids": [f.id for f in st.repo_files if f.artifact_type == ArtifactType.checkpoint],
            "inspected_paper_section_ids": [s.id for s in st.paper_sections if s.inspected],
            "inspected_code_file_ids": [f.id for f in st.repo_files if f.inspected],
            "inspected_config_ids": [c.id for c in st.configs if c.inspected],
            "inspected_log_ids": [l.id for l in st.logs if l.inspected],
            "checks": checks,
            "observed_evidence": observed_evidence,
            "final_verdict": st.final_verdict.value if st.final_verdict else None,
            "final_failure_type": st.final_failure_type.value if st.final_failure_type else None,
            "step_ok": True,
            "reward_breakdown": self._last_reward_breakdown.model_dump() if self._last_reward_breakdown else {},
            "suggested_action_families": self._suggested_action_families(st),
        }

    def _suggested_action_families(self, st: AuditState) -> list[str]:
        suggestions: list[str] = []
        if not st.checks:
            suggestions.append("inspect_or_search_before_verdict")
        if st.target_claim.claimed_split:
            suggestions.append("split_consistency")
        if st.target_claim.claimed_metric_name:
            suggestions.append("metric_consistency")
        if st.logs:
            suggestions.append("reproduction_or_result_check")
        if st.target_claim.claimed_method:
            suggestions.append("paper_code_consistency")
        return suggestions

    def _observation(self, reward: float, done: bool) -> ReproPilotObservation:
        text = build_briefing(self.audit_state)
        if len(text) > 48_000:
            text = text[:47_999] + "..."
        return ReproPilotObservation(
            echoed_message=text,
            message_length=len(text),
            reward=reward,
            done=done,
            metadata=self._metadata(),
        )
