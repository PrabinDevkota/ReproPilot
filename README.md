---
title: ReproPilot Environment Server
emoji: "🔬"
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - research-audit
  - reproducibility
---

# ReproPilot

ReproPilot is an OpenEnv environment for training LLM agents to audit research-paper claims against supplied artifacts. The agent receives a research audit briefing, chooses structured JSON actions, inspects paper/code/config/log evidence, runs deterministic methodology checks, and submits a grounded verdict.

The goal is not to prove scientific truth from scratch. The goal is to teach a model to stop guessing and instead perform a disciplined reproducibility audit over the evidence it can see.

## Why It Matters

AI systems can generate plausible papers, benchmarks, and result tables. Human reviewers still have to answer basic but time-consuming questions:

- Does the claimed metric match the artifact logs?
- Does the claim say test while the code uses validation?
- Was the test set used for hyperparameter search?
- Are baselines compared fairly?
- Is the claimed method actually implemented?
- Is a strong result supported by enough runs, seeds, and evidence?

ReproPilot turns those questions into an RL environment with interpretable rewards.

## Current Scale

- 52 total scenarios
- 39 training scenarios
- 13 heldout scenarios
- 18 regression tests
- 29 legal JSON actions
- 12 deterministic audit checks

## Architecture

```text
JSON scenarios
  -> Pydantic audit state
  -> plain-text research briefing
  -> structured AgentAction JSON
  -> OpenEnv transition
  -> deterministic checks + evidence updates
  -> shaped and terminal rewards
  -> GRPO/SFT training in Colab
  -> heldout evaluation
```

Core files:

- `models.py` defines claims, artifacts, evidence, checks, actions, verdicts, and scenarios.
- `checkers.py` implements deterministic validation checks.
- `server/repropilot_environment.py` implements the OpenEnv environment.
- `rewards.py` implements dense action shaping and final verdict rewards.
- `notebooks/trainer.ipynb` trains a policy with Unsloth + TRL GRPO against the deployed Hugging Face Space.
- `evaluation/evaluate_policy.py` runs baseline or trained policy evaluation.
- `tests/test_repropilot_core.py` verifies schemas, checkers, transitions, rewards, and heldout evaluation.

## Failure Modes

ReproPilot covers common research-audit failures:

- `metric_mismatch`
- `split_mismatch`
- `data_leakage`
- `cherry_picked_seed`
- `paper_code_mismatch`
- `invalid_ablation`
- `result_mismatch`
- `missing_artifact`
- `dataset_provenance_issue`
- `hyperparameter_search_bias`
- `baseline_unfairness`
- `statistical_underpower`
- `incomplete_implementation`
- `ambiguous_method`
- `unsupported_claim`

## Legal Actions

Artifact inspection:

- `read_claim`
- `inspect_paper_section`
- `inspect_code_file`
- `inspect_config`
- `inspect_logs`
- `inspect_result_table`
- `inspect_dataset_card`
- `inspect_checkpoint`
- `search_artifacts`

Composite audit actions:

- `compare_claim_to_artifacts`
- `audit_experiment_design`
- `rank_evidence`
- `plan_next_check`
- `synthesize_findings`

Deterministic checks:

- `run_metric_check`
- `run_split_check`
- `run_leakage_check`
- `run_seed_check`
- `run_ablation_check`
- `run_paper_code_consistency_check`
- `run_reproduction_check`
- `run_dataset_provenance_check`
- `run_hyperparameter_search_check`
- `run_baseline_fairness_check`
- `run_statistical_significance_check`
- `run_implementation_completeness_check`

Episode control:

- `mark_inconclusive`
- `submit_verdict`
- `do_nothing`

Example action:

```json
{
  "action_type": "run_split_check",
  "target_id": "claim_001",
  "explanation": "The claim reports test accuracy, so verify the artifact split."
}
```

Example final verdict:

```json
{
  "action_type": "submit_verdict",
  "verdict": "NOT_SUPPORTED_METHOD_INVALID",
  "failure_type": "split_mismatch",
  "evidence_ids": ["ev_split_mismatch_file_eval_1"],
  "explanation": "The paper claims test accuracy, but the evaluation code loads the validation split."
}
```

## Reward Design

ReproPilot uses two reward layers.

Dense non-terminal shaping rewards useful audit behavior:

- valid JSON actions
- relevant artifact inspection
- relevant deterministic checks
- composite audit actions such as experiment-design audit
- evidence ranking and synthesis
- avoiding repeated idle/read-only behavior
- avoiding hidden gold access

Terminal rewards score final answer quality:

- verdict correctness
- failure type correctness
- evidence grounding
- checker usage
- reproduction behavior
- novelty calibration
- efficiency
- anti-hallucination

This design gives GRPO useful reward variance before the final verdict, which prevents collapse into safe but useless actions.

## Training

The active training notebook is:

```text
notebooks/trainer.ipynb
```

It is designed for Google Colab:

- uses `!pip install` only
- loads `unsloth/Qwen2.5-3B-Instruct` by default
- trains a LoRA adapter
- uses a small SFT warm-start for JSON/action routing
- runs staged GRPO against the deployed Hugging Face Space
- logs reward components and plots learning curves

Training stages:

- SFT warm-start: teaches objective-to-action routing.
- Stage B: higher-temperature exploration.
- Stage C: lower-temperature refinement.
- Stage D: low-temperature stabilization.

Notebook outputs:

- `reward_log.csv`
- `before_after.png`
- `reward_curve.png`
- `components.png`
- `reward_std.png`
- `kl_curve.png`
- `rolling_reward.png`
- final LoRA adapter
- run manifest

## Local Quickstart

Install dependencies:

```bash
uv sync --extra dev
```

Run the server:

```bash
uv run server --port 8000
```

Run tests:

```bash
uv run --extra dev pytest -q
```

Run heldout evaluation:

```python
from baselines.smart_policy import smart_action
from evaluation.evaluate_policy import evaluate

report = evaluate(lambda obs, rng: smart_action(obs, rng), split="heldout")
print(report.aggregate())
```

## Hugging Face Space

This repository is configured as a Docker Space. The Space exposes the OpenEnv API and web interface:

- `/health`
- `/reset`
- `/step`
- `/web`

The Colab trainer calls `/reset` to get the research audit briefing and `/step` to score model-generated JSON actions.

## Evaluation Story

A typical ReproPilot episode:

1. The briefing says a method achieves 91.2% test accuracy.
2. A weak model may immediately claim the result is supported.
3. A trained ReproPilot policy inspects the evaluation code and config.
4. It notices `split="validation"` while the claim says `test`.
5. It runs `run_split_check`.
6. It submits a verdict with `split_mismatch` and grounded evidence.
7. The reward breakdown credits verdict correctness, failure type, evidence, and checker use.

## Hackathon Pitch

ReproPilot makes reproducibility auditing trainable. It is not a chatbot that gives opinions about papers. It is an interactive environment where an LLM must take auditable steps, gather evidence, run checks, and justify a final verdict. The reward function is transparent, the scenarios are inspectable, and heldout cases test whether the policy learned the audit process rather than memorizing one bug pattern.
