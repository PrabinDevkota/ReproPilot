from __future__ import annotations

import random
from pathlib import Path

from baselines.random_policy import random_action
from baselines.smart_policy import smart_action
from briefing import build_briefing
from checkers import (
    baseline_fairness_check,
    dataset_provenance_check,
    hyperparameter_search_check,
    implementation_completeness_check,
    leakage_check,
    metric_check,
    reproduction_check,
    seed_check,
    split_check,
    statistical_significance_check,
)
from evaluation.evaluate_policy import evaluate
from models import ActionType, AgentAction, FailureType, Scenario, ValidationVerdict
from server.repropilot_environment import ReproPilotEnvironment

ROOT = Path(__file__).resolve().parents[1]


def scenario_path(name: str) -> Path:
    return ROOT / "scenarios" / "train" / name


def test_load_repropilot_scenario_json() -> None:
    scen = Scenario.model_validate_json(scenario_path("split_mismatch_test_vs_val_001.json").read_text(encoding="utf-8"))
    assert scen.hidden_gold.gold_failure_type == FailureType.split_mismatch


def test_hidden_gold_not_in_briefing() -> None:
    scen = Scenario.model_validate_json(scenario_path("split_mismatch_test_vs_val_001.json").read_text(encoding="utf-8"))
    text = build_briefing(scen.state).lower()
    assert "hidden_gold" not in text
    assert "gold_verdict" not in text
    assert "target claim" in text


def test_agent_action_schema_valid() -> None:
    action = AgentAction(action_type=ActionType.run_split_check, target_id="claim_001")
    assert action.action_type == ActionType.run_split_check


def test_checkers_detect_core_failures() -> None:
    metric_env = ReproPilotEnvironment(scenario_path("metric_mismatch_f1_vs_accuracy_001.json"))
    metric_env.reset()
    assert metric_check(metric_env.audit_state).failure_type == FailureType.metric_mismatch

    split_env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    split_env.reset()
    assert split_check(split_env.audit_state).failure_type == FailureType.split_mismatch

    leak_env = ReproPilotEnvironment(scenario_path("leakage_scaler_fit_full_data_001.json"))
    leak_env.reset()
    assert leakage_check(leak_env.audit_state).failure_type == FailureType.data_leakage

    seed_env = ReproPilotEnvironment(scenario_path("cherry_picked_best_seed_001.json"))
    seed_env.reset()
    assert seed_check(seed_env.audit_state).failure_type == FailureType.cherry_picked_seed


def test_new_complex_scenarios_trigger_specialized_checkers() -> None:
    cases = [
        ("dataset_provenance_private_test_030.json", dataset_provenance_check, FailureType.dataset_provenance_issue),
        ("hyperparameter_tuned_on_test_032.json", hyperparameter_search_check, FailureType.hyperparameter_search_bias),
        ("baseline_default_settings_034.json", baseline_fairness_check, FailureType.baseline_unfairness),
        ("statistical_sota_single_run_036.json", statistical_significance_check, FailureType.statistical_underpower),
        ("implementation_stub_038.json", implementation_completeness_check, FailureType.incomplete_implementation),
    ]
    for file_name, checker, expected_failure in cases:
        env = ReproPilotEnvironment(scenario_path(file_name))
        env.reset()
        check = checker(env.audit_state)
        assert check.failure_type == expected_failure
        assert check.evidence_ids


def test_all_scenarios_load_and_accept_planning_action() -> None:
    for split in ("train", "heldout"):
        for path in (ROOT / "scenarios" / split).glob("*.json"):
            scen = Scenario.model_validate_json(path.read_text(encoding="utf-8"))
            assert scen.split == split
            env = ReproPilotEnvironment(path)
            obs = env.reset()
            assert obs.metadata["scenario_id"] == scen.scenario_id
            obs = env.step(AgentAction(action_type=ActionType.plan_next_check, target_id="claim_001"))
            assert obs.reward is not None


def test_valid_novel_method_not_flagged_invalid() -> None:
    env = ReproPilotEnvironment(scenario_path("valid_novel_entropy_gated_routing_001.json"))
    env.reset()
    assert reproduction_check(env.audit_state).issue_found is False


def test_transition_and_submit_verdict() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    obs = env.reset()
    obs = env.step(AgentAction(action_type=ActionType.inspect_code_file, target_id="file_eval"))
    assert "file_eval" in obs.metadata["inspected_code_file_ids"]
    obs = env.step(AgentAction(action_type=ActionType.run_split_check, target_id="claim_001"))
    ev_ids = [e["id"] for e in obs.metadata["observed_evidence"]]
    obs = env.step(
        AgentAction(
            action_type=ActionType.submit_verdict,
            verdict=ValidationVerdict.NOT_SUPPORTED_METHOD_INVALID,
            failure_type=FailureType.split_mismatch,
            evidence_ids=ev_ids,
        )
    )
    assert obs.done
    assert (obs.reward or 0) > 1.0
    assert "verdict_correctness" in obs.metadata["reward_breakdown"]


def test_dense_action_shaping_prefers_relevant_checks_over_read_claim() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    read_obs = env.step(AgentAction(action_type=ActionType.read_claim, target_id="claim_001"))

    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    check_obs = env.step(AgentAction(action_type=ActionType.run_split_check, target_id="claim_001"))

    assert (check_obs.reward or 0.0) > (read_obs.reward or 0.0)
    assert check_obs.metadata["reward_breakdown"]["checker_usage"] > 0


def test_do_nothing_is_penalized_for_single_step_training() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    obs = env.step(AgentAction(action_type=ActionType.do_nothing))
    assert (obs.reward or 0.0) < 0


def test_compare_claim_to_artifacts_runs_multiple_checks() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    obs = env.step(AgentAction(action_type=ActionType.compare_claim_to_artifacts, target_id="claim_001"))
    check_names = {c["check_name"] for c in obs.metadata["checks"]}
    assert "split_check" in check_names
    assert "metric_check" in check_names
    assert (obs.reward or 0.0) > 0.2


def test_synthesize_findings_rewards_existing_audit_progress() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    env.step(AgentAction(action_type=ActionType.run_split_check, target_id="claim_001"))
    obs = env.step(AgentAction(action_type=ActionType.synthesize_findings))
    assert (obs.reward or 0.0) > 0
    assert "Synthesis:" in (obs.metadata.get("last_action_result") or obs.echoed_message)


def test_audit_experiment_design_runs_methodology_checks() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    obs = env.step(AgentAction(action_type=ActionType.audit_experiment_design, target_id="claim_001"))
    check_names = {c["check_name"] for c in obs.metadata["checks"]}
    assert {"split_check", "leakage_check", "hyperparameter_search_check", "baseline_fairness_check", "statistical_significance_check"} <= check_names
    assert (obs.reward or 0.0) > 0.2


def test_plan_and_rank_evidence_actions_are_stateful() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    plan_obs = env.step(AgentAction(action_type=ActionType.plan_next_check))
    assert "Next" in plan_obs.echoed_message
    env.step(AgentAction(action_type=ActionType.run_split_check, target_id="claim_001"))
    rank_obs = env.step(AgentAction(action_type=ActionType.rank_evidence))
    assert (rank_obs.reward or 0.0) > 0
    assert "Top evidence" in rank_obs.echoed_message


def test_hidden_gold_access_penalized() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    obs = env.step(AgentAction(action_type=ActionType.search_artifacts, target_id="hidden gold answer"))
    assert (obs.reward or 0) < 0


def test_fabricated_evidence_penalty() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    env.reset()
    obs = env.step(
        AgentAction(
            action_type=ActionType.submit_verdict,
            verdict=ValidationVerdict.NOT_SUPPORTED_METHOD_INVALID,
            failure_type=FailureType.split_mismatch,
            evidence_ids=["fake_ev"],
        )
    )
    assert obs.metadata["reward_breakdown"]["evidence_grounding"] < 0


def test_random_and_smart_policies_run() -> None:
    env = ReproPilotEnvironment(scenario_path("split_mismatch_test_vs_val_001.json"))
    obs = env.reset()
    assert isinstance(random_action(obs, random.Random(0)), AgentAction)

    done = False
    for _ in range(12):
        obs = env.step(smart_action(obs, random.Random(0)))
        done = obs.done
        if done:
            break
    assert done
    assert env.audit_state.final_verdict is not None


def test_heldout_eval_runs() -> None:
    report = evaluate(lambda obs, rng: smart_action(obs, rng), split="heldout")
    agg = report.aggregate()
    assert agg["episodes"] >= 8
    assert "verdict_accuracy" in agg
