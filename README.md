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

ReproPilot is an OpenEnv-compliant environment for training LLM agents on reproducibility investigation and professional research-workflow tasks. The agent audits research-paper claims against supplied artifacts, chooses structured JSON actions, gathers evidence, runs deterministic methodology checks, and submits a grounded verdict.

The goal is not to prove scientific truth from scratch. The goal is to train agents to stop guessing and instead perform disciplined, multi-step reproducibility audits over the evidence they can actually inspect.

## Quick Read

**Problem:** LLMs can sound confident about research results without checking whether the paper, code, logs, configs, and dataset evidence agree.

**Environment:** ReproPilot gives the agent a research audit briefing, artifact IDs, and a 12-step budget. The agent must inspect evidence, run checks, plan next steps, and submit a verdict as structured JSON.

**Results:** GRPO training changes behavior from random or shallow action selection into evidence-seeking audit behavior. The plots below show trained stages B/C/D strongly outperforming the random baseline, with reward improving while loss remains controlled.

**Why it matters:** Reviewers, research engineers, ML platform teams, and anyone evaluating AI-generated research claims need agents that can audit evidence instead of writing plausible but unverified summaries.

Try it through the [Hugging Face Space](https://huggingface.co/spaces/riwaj43adz/repro).

## Why this matters

LLMs often produce plausible research and debugging narratives without doing the careful investigation that reproducibility work requires. In real reviews, the hard questions are operational:

- Does the claimed metric match the logs?
- Does the paper say test while the code uses validation?
- Was the test set used for hyperparameter search?
- Are baselines compared fairly?
- Is the claimed method actually implemented?
- Is a strong result supported by enough runs, seeds, and evidence?

ReproPilot targets that capability gap. It trains agents to inspect evidence, choose valid actions, avoid idle or reward-hacking behavior, and improve over long-running environment interactions.

## Hackathon Theme Fit

ReproPilot fits **Theme #3.1: Professional Tasks / World Modeling**. The agent interacts with a dynamic environment, maintains state across a multi-step audit, and must decide what evidence to inspect before issuing a final judgment.

It also exercises long-horizon planning: the agent has up to 12 steps to move from a paper claim to artifact inspection, methodology checks, evidence synthesis, and a calibrated verdict.

## Environment Overview

The environment is a reproducibility investigation loop. At reset, the agent receives a plain-text research audit briefing with the target claim, paper sections, artifact identifiers, available configs/logs/result tables, and validation checks already run. The briefing is intentionally close to what a research engineer or reviewer would face: enough evidence to investigate, but not a hidden answer.

On each step, the agent chooses one structured action. It can read the claim, inspect a paper section, inspect code/config/log/result artifacts, search artifacts, compare claims to artifacts, audit experiment design, rank evidence, plan the next check, synthesize findings, run deterministic checks, or submit a final verdict.

The environment returns the next observation, a reward, a done flag, and metadata containing scenario state, observed evidence, checker results, and reward breakdown. Success means submitting the right verdict and failure type with valid evidence after using relevant checks, while avoiding fabricated evidence, hidden gold access, repeated idle actions, and unsupported claims.

ReproPilot follows the OpenEnv reset / step / state pattern:

- `openenv.yaml` declares the environment metadata, tasks, runtime, port, and max steps.
- `server/repropilot_environment.py` implements the OpenEnv environment class.
- `server/app.py` exposes the OpenEnv HTTP/web interface through FastAPI.
- `pyproject.toml` depends on `openenv-core[core]>=0.2.3`; the committed lockfile resolves `openenv-core==0.2.3`.

Hugging Face Space deployment:

- [Hugging Face Space](https://huggingface.co/spaces/riwaj43adz/repro)

## Current Scale

- 52 total scenarios
- 39 training scenarios
- 13 heldout scenarios
- 18 regression tests
- 29 legal JSON actions
- 12 deterministic audit checks

## Reward Design

The reward is interpretable and intentionally shaped around reproducibility work rather than surface-level answers.

- **Objective progress reward:** credits moving toward the correct verdict, correct failure type, valid evidence, and relevant checker usage.
- **Environment feedback reward:** gives dense step-level feedback for useful inspection, comparison, audit, planning, ranking, and synthesis actions.
- **Valid action / formatting reward:** rewards valid structured JSON actions and penalizes malformed or unsupported actions.
- **Audit-policy / evidence-seeking reward:** rewards inspecting the right artifacts, running deterministic checks, ranking evidence, and grounding the final verdict in observed evidence.
- **Anti-idle / anti-gaming reward:** penalizes repeated actions, `do_nothing`, hidden/gold-answer access attempts, fabricated evidence, premature verdicts, and timeout without a verdict.

![Reward component breakdown](assets/reward_component_breakdown.png)
**Figure: Reward component breakdown.** The reward combines objective progress, environment feedback, valid action formatting, audit-policy behavior, and anti-idle incentives so the agent is rewarded for reproducibility work rather than superficial answers.

## Training Setup

Training uses Hugging Face tooling with TRL GRPO, plus a small SFT warm-start for JSON/action routing. The active notebook is [notebooks/trainer.ipynb](notebooks/trainer.ipynb), designed for Google Colab with `!pip install` setup, an `unsloth/Qwen2.5-3B-Instruct` default model, LoRA adapter training, staged GRPO, and reward-component logging.

The training loop connects to the deployed Hugging Face Space environment rather than only training on a static dataset: the notebook calls `/reset` for a research audit briefing and `/step` to score model-generated JSON actions.

Training stages:

- **SFT warm-start:** teaches objective-to-action routing.
- **Stage B:** higher-temperature exploration.
- **Stage C:** lower-temperature refinement.
- **Stage D:** low-temperature stabilization.

Notebook outputs include reward logs, learning curves, reward component plots, final LoRA adapter artifacts, and a run manifest.

## Results

The key comparison is simple: the random baseline often receives negative mean reward because it wastes steps or submits weak verdicts, while trained GRPO stages learn to collect evidence and use the audit tools before answering.

![Baseline vs trained reward](assets/baseline_vs_trained_reward.png)
**Figure 1. Baseline vs trained reward.** The trained GRPO stages achieve much higher mean reward than the random baseline on the deployed Hugging Face Space.

![Reward and loss training progress](assets/reward_loss_training_progress.png)
**Figure 2. Reward and loss training progress.** Rolling reward improves quickly and remains high, while rolling mean loss stays controlled across training steps.

![Training loss curve](assets/training_loss_curve.png)
**Figure 3. Training loss curve.** GRPO training loss remains bounded across stages, providing the required loss evidence from a real training run.

![Training signal - rolling reward averages](assets/training_signal_rolling_reward_averages.png)
**Figure 4. Training signal - rolling reward averages.** Total reward rises quickly and stabilizes, while objective and environment reward components improve during training.

![Total reward by step](assets/reward_curve_by_step.png)
**Figure 5. Total reward by step.** Rewards remain consistently positive across trained stages B, C, and D after the initial learning phase.

After training, the behavior changes from guessing to investigation. A random or untrained baseline often burns steps, repeats low-value actions, or submits unsupported verdicts. The trained policy more reliably inspects artifacts, runs the relevant deterministic checks, gathers evidence, and submits a verdict tied to observed evidence.

In a typical split-mismatch case, the trained policy does not stop at the paper claim. It inspects the evaluation code/config, discovers that the artifact uses validation while the paper claims test performance, runs the split check, and submits a grounded `split_mismatch` verdict. That is the behavior ReproPilot is designed to reward.

## Additional Diagnostics

<details>
<summary>Additional diagnostics</summary>

![Reward vs loss](assets/reward_vs_loss.png)
**Diagnostic: Reward vs loss.** Most later-stage samples cluster around low loss and positive reward, with early-stage outliers visible.

</details>

## How to Run

Install dependencies:

```bash
uv sync --extra dev
```

Run the OpenEnv server locally:

```bash
uv run server --port 8000
```

Open the local web interface:

```text
http://127.0.0.1:8000/web
```

Smoke-test the HTTP endpoints:

```bash
uv run python scripts/http_endpoint_smoke.py --local
```

Run a short local episode demo:

```bash
uv run python scripts/demo_repropilot.py
```

Run tests:

```bash
uv run --extra dev pytest -q
```

Run heldout evaluation from Python:

```python
from baselines.smart_policy import smart_action
from evaluation.evaluate_policy import evaluate

report = evaluate(lambda obs, rng: smart_action(obs, rng), split="heldout")
print(report.aggregate())
```

Run training:

- Open [notebooks/trainer.ipynb](notebooks/trainer.ipynb) in Colab.
- Set the Hugging Face Space URL in the notebook.
- Run the SFT warm-start and GRPO stages B/C/D.

If testing against the deployed Space, use:

```bash
uv run python scripts/http_endpoint_smoke.py --url https://riwaj43adz-repro.hf.space
```

## Submission Links

| Item | Link or path |
| --- | --- |
| Hugging Face Space | [https://huggingface.co/spaces/riwaj43adz/repro](https://huggingface.co/spaces/riwaj43adz/repro) |
| Training Notebook (with logs & figures) | [notebooks/trainer.ipynb](notebooks/trainer.ipynb) |
| Mini-blog / video / slides | [Blog.md](Blog.md) |
| OpenEnv manifest | [openenv.yaml](openenv.yaml) |

## Engineering Notes

ReproPilot separates environment/server code from client, training, and evaluation code. The OpenEnv environment lives in [server/repropilot_environment.py](server/repropilot_environment.py), the FastAPI/OpenEnv server wrapper lives in [server/app.py](server/app.py), baseline policies live under [baselines](baselines), and heldout evaluation lives in [evaluation/evaluate_policy.py](evaluation/evaluate_policy.py).

The environment uses standard reset / step / state behavior. `reset` loads a scenario and returns the initial research audit briefing. `step` validates and applies a structured action, updates the audit state, returns the next briefing, and attaches reward metadata. The [openenv.yaml](openenv.yaml) manifest declares the OpenEnv runtime, task families, max steps, and grader entry points.

<details>
<summary>Example structured actions</summary>

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

</details>

## Evaluation Story

A typical ReproPilot episode starts with a briefing saying a method achieves 91.2% test accuracy. A weak model may immediately claim the result is supported. A trained ReproPilot policy inspects the evaluation code and config, notices `split="validation"` while the claim says `test`, runs `run_split_check`, and submits a `split_mismatch` verdict with grounded evidence.

That is the core story: problem -> environment -> reward -> training -> better reproducibility behavior.
