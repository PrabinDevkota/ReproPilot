---
title: ReproPilot Environment Server
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - research-audit
---

# ReproPilot

**ReproPilot** is an OpenEnv environment for training LLM agents to audit research-paper claims. The agent receives a plain-text research audit briefing, inspects paper sections, code files, configs, logs, and artifacts, runs deterministic methodology checks, then submits an evidence-backed verdict.

ReproPilot does **not** prove scientific truth. It validates a focused claim against available artifacts and objective failure modes: metric mismatch, split mismatch, train/test leakage, cherry-picked seeds, paper-code mismatch, invalid ablations, result mismatch, missing artifacts, and calibrated support for novel methods.

## Architecture

`JSON scenarios -> Pydantic audit state -> research briefing -> structured action -> checker/world transition -> interpretable rewards -> TRL/GRPO training -> held-out evaluation`

The project keeps the original OpenEnv/FastAPI shape but replaces the domain with a research-claim validation environment.

## Legal Actions

- `read_claim`
- `inspect_paper_section`
- `inspect_code_file`
- `inspect_config`
- `inspect_logs`
- `inspect_result_table`
- `inspect_dataset_card`
- `inspect_checkpoint`
- `search_artifacts`
- `compare_claim_to_artifacts`
- `audit_experiment_design`
- `rank_evidence`
- `plan_next_check`
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
- `synthesize_findings`
- `mark_inconclusive`
- `submit_verdict`
- `do_nothing`

Example check action:

```json
{
  "action_type": "run_split_check",
  "target_id": "claim_001",
  "explanation": "The claim mentions test accuracy, so I need to verify the evaluation split."
}
```

Example final verdict:

```json
{
  "action_type": "submit_verdict",
  "verdict": "NOT_SUPPORTED_METHOD_INVALID",
  "failure_type": "split_mismatch",
  "evidence_ids": ["ev_split_mismatch_file_eval_1"],
  "explanation": "The paper claims test accuracy, but evaluate.py loads the validation split."
}
```

## Reward Channels

Terminal verdict rewards expose:

- `verdict_correctness`
- `failure_type`
- `evidence_grounding`
- `checker_usage`
- `reproduction`
- `novelty_calibration`
- `efficiency`
- `anti_hallucination`

Small shaping rewards encourage valid actions, first-time inspections, and relevant checker use. Hidden gold labels are never included in observations.

## Scenarios

Synthetic scenarios live in:

- `scenarios/train/`
- `scenarios/heldout/`

The active set includes metric mismatch, split mismatch, leakage, cherry-picked seed, result mismatch, missing artifact, paper-code mismatch, invalid ablation, valid standard method, and valid novel method cases.

## Demo Story

1. A base model sees: "Model achieves 91.2% test accuracy."
2. It prematurely says the claim is supported.
3. A ReproPilot-trained agent inspects `repo/evaluate.py`.
4. It finds `load_dataset(split="validation")` while the paper says `test`.
5. It runs `run_split_check`.
6. It submits `NOT_SUPPORTED_METHOD_INVALID` with `split_mismatch` evidence.
7. Reward breakdown shows high verdict, evidence, and checker rewards.
8. Held-out plots show improvement in verdict accuracy and evidence validity.

## Quickstart

```bash
uv sync
uv run server --port 8000
```

Run tests:

```bash
uv run pytest -q
```

Run held-out evaluation from Python:

```python
from baselines.smart_policy import smart_action
from evaluation.evaluate_policy import evaluate

report = evaluate(lambda obs, rng: smart_action(obs, rng), split="heldout")
print(report.aggregate())
```

Generate rejection-sampled SFT traces:

```bash
uv run python -m training.generate_sft_data
```

## Hackathon Positioning

AI can generate convincing research papers and results, but validation is still manual. ReproPilot turns research-claim validation into an RL environment. The agent must inspect paper sections, code, configs, and logs; run objective methodology checks; and submit evidence-backed verdicts. Rewards are given for correct verdicts, correct failure types, grounded evidence, and calibrated uncertainty. Through GRPO, the model learns to stop guessing and audit claims systematically.
