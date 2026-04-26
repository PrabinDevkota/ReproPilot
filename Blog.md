# ReproPilot: Teaching LLMs To Audit Research Before They Trust It

## The Story

This is a fictional story, but it is uncomfortably easy to imagine. A highly cited medical researcher spends decades building a reputation strong enough that reviewers treat his papers as low-risk. When AI research tools arrive, he uses them the way everyone around him does: first as assistants, then as accelerators, and eventually as quiet co-authors in the research process. They summarize literature, suggest drug-repurposing leads, generate experiment plans, and produce clean explanations that sound correct. While studying a rare disease, his lab reports a promising compound. The paper claims the treatment can control the disease, the plots look clean, the code folder exists, and the author's name carries enough weight that nobody slows down to verify every artifact. The claim enters the literature with the confidence of a serious result.

Years later, the same researcher develops that rare disease. His own proposed treatment is considered, but the evidence behind it was weaker than it looked. The benchmark split did not match the claim, some artifacts were missing, and the AI-assisted analysis had produced a convincing chain of reasoning without a reproducible one. The tragedy is not that AI was used; the tragedy is that nobody forced the claim through a systematic audit before trusting it. ReproPilot is built around that failure mode: when scientific claims become easier to generate, verification has to become easier to train.

## The Problem

LLMs are good at producing confident scientific narratives. They are much worse at the dull but crucial work of verification. A paper claim is rarely just one sentence; it is tied to code, configs, logs, datasets, seeds, result tables, and implementation details. If those artifacts disagree, the model should notice before it writes a polished answer.

ReproPilot targets that exact gap. We want an agent that behaves less like a hype machine and more like a skeptical reviewer: inspect first, check evidence, then answer.

| Research-audit failure | What a weak LLM often does | What ReproPilot trains |
| --- | --- | --- |
| Paper says test, code uses validation | Trusts the paper sentence | Runs a split check |
| Metric name changes across artifacts | Summarizes the result anyway | Compares claim metric to logs/configs |
| Result has one cherry-picked seed | Treats the number as stable | Checks seed evidence |
| Method is claimed but not implemented | Trusts method section prose | Inspects code and paper-code consistency |
| Evidence is missing | Fills gaps with plausible language | Marks uncertainty or missing artifacts |

## What We Built

ReproPilot is an OpenEnv environment for research-claim auditing. On reset, the agent receives a research audit briefing with the target claim, artifact IDs, available paper sections, repository files, configs, logs, evidence collected so far, and remaining steps. On each step, the agent must return one structured JSON action.

The action space is intentionally broader than a toy benchmark. The agent can inspect artifacts, run deterministic methodology checks, compare claims to artifacts, audit experiment design, rank evidence, synthesize findings, and submit a final verdict. The final answer must cite observed evidence rather than invented evidence IDs.

| Environment piece | What it means |
| --- | --- |
| Observation | Plain-text audit briefing plus metadata |
| Action | One JSON `AgentAction` |
| State | Claim, artifacts, checks, evidence, final verdict |
| Reward | Dense audit progress plus terminal verdict score |
| Goal | Correct verdict, correct failure type, real evidence |

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

## Why This Is An RL Environment

This is not a static classification dataset. The model has to interact with the environment. It decides what to inspect, which check to run, when to synthesize, and when enough evidence exists for a final verdict. A good answer requires process, not just prediction.

That interaction creates realistic failure modes. The agent can submit too early, cite evidence it has not observed, fabricate evidence IDs, repeat low-value actions, run irrelevant checks, or trust a result without checking the artifact trail. The environment rewards the audit process itself, not only the final answer.

## Actions And Checks

ReproPilot includes 29 legal actions and 12 deterministic audit checks. The checks cover common reproducibility and methodology failures: metric mismatch, split mismatch, leakage, cherry-picked seeds, paper-code mismatch, invalid ablations, result mismatch, missing artifacts, dataset provenance issues, hyperparameter search bias, unfair baselines, statistical underpower, and incomplete implementations.

| Action family | Examples | Why it matters |
| --- | --- | --- |
| Inspection | inspect code/config/log/result artifacts | Forces evidence gathering |
| Deterministic checks | split, metric, leakage, seed, reproduction | Gives objective audit tools |
| Composite audit | compare claim to artifacts, audit experiment design | Encourages multi-step reasoning |
| Evidence work | rank evidence, synthesize findings | Trains grounded summaries |
| Final judgment | submit verdict, mark inconclusive | Rewards calibrated conclusions |

This makes the environment meaningfully challenging: the model has to reason about methodology, evidence, and uncertainty rather than just pick a label.

## Reward Design

The reward signal has two layers. Dense shaping rewards guide the agent during the episode: valid JSON, useful inspection, relevant checks, planning, ranking, synthesis, and anti-idle behavior. Terminal rewards score the final audit: verdict correctness, failure type, evidence grounding, checker usage, reproduction behavior, novelty calibration, efficiency, and anti-hallucination.

The key design choice is that reward is not just pass/fail at the end. A sparse final reward made early runs collapse into safe actions. Dense audit-progress reward gives GRPO enough signal to learn useful behavior before the final verdict.

| Reward component | What it teaches |
| --- | --- |
| Format reward | Emit valid structured JSON |
| Objective reward | Pick the right action family for the audit objective |
| Environment reward | Do actions the live OpenEnv environment considers useful |
| Audit-policy reward | Inspect evidence and run meaningful checks |
| Anti-idle reward | Avoid repeated or useless actions |

## Training Setup

We trained a LoRA adapter with OpenEnv, Unsloth, and Hugging Face TRL GRPO. The active training notebook runs in Google Colab, talks to the deployed Hugging Face Space through `/reset` and `/step`, and trains against live environment rewards rather than a static answer file.

Training has three stages. First, a small SFT warm-start teaches JSON formatting and objective-to-action routing. Then GRPO Stage B explores at higher temperature. Stages C and D lower temperature and learning rate to stabilize the behavior.

| Stage | Purpose |
| --- | --- |
| SFT warm-start | Learn valid JSON and basic action routing |
| GRPO B | Explore audit actions against live rewards |
| GRPO C | Refine toward higher-value checks |
| GRPO D | Stabilize lower-temperature behavior |

## What The Agent Learned

Early behavior looked like a normal untrained LLM: long reasoning traces, invalid formats, and safe-but-useless actions. After the warm-start and GRPO stages, the policy became much more tool-oriented. When the prompt asked for code inspection, it inspected code. When the task required split or metric verification, it selected the corresponding check. When the audit needed broader methodology review, it used the experiment-design action instead of guessing from the claim text.

The two most important figures for the submission are the reward-vs-baseline plot and the reward/loss training plot. The first shows that trained GRPO stages outperform shallow baselines. The second shows that reward improves during training while loss remains controlled, which is the simplest proof that the model actually trained against the environment.

![Baseline vs trained reward](assets/baseline_vs_trained_reward.png)

**Figure 1. Baseline vs trained reward.** The trained stages achieve higher reward than the random baseline by learning to inspect evidence and run checks before answering.

![Reward and loss training progress](assets/reward_loss_training_progress.png)

**Figure 2. Reward and loss training progress.** Rolling reward improves across training while loss stays bounded, showing useful optimization rather than random drift.

Qualitatively, the behavior changed from "sound confident" to "investigate first." In a split-mismatch scenario, the trained policy inspects the artifacts, catches that the code/config uses validation while the paper claims test accuracy, runs the split check, and submits a grounded `split_mismatch` verdict.

## Why It Matters

Scientific AI should not only generate ideas. It should also learn to challenge them. ReproPilot asks an LLM to slow down, inspect artifacts, run checks, and admit uncertainty when the evidence is incomplete.

The same pattern matters outside this benchmark: medical literature review, ML reproducibility, benchmark validation, grant review, compliance audits, and scientific assistants that need to verify before advising. The point is not to replace expert reviewers. The point is to give them a tireless first-pass auditor that knows how to look for the boring details humans miss when a paper looks polished.

## What Comes Next

The current version uses synthetic scenarios with realistic artifact patterns. Next, we want to add real-world benchmark reproductions, longer multi-turn websocket rollouts during training, richer dataset-provenance checks, more adversarial reward-hacking cases, and larger heldout scenario families.

ReproPilot is a small step toward LLMs that do not just sound scientific, but know how to verify scientific claims.
