#!/usr/bin/env python3
"""Step-by-step ReproPilot demo for the split-mismatch scenario."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import ActionType, AgentAction, FailureType, ValidationVerdict
from server.repropilot_environment import ReproPilotEnvironment


def main() -> int:
    env = ReproPilotEnvironment(ROOT / "scenarios" / "train" / "split_mismatch_test_vs_val_001.json")
    obs = env.reset()
    print(obs.echoed_message.splitlines()[0])
    for action in [
        AgentAction(action_type=ActionType.read_claim),
        AgentAction(action_type=ActionType.inspect_code_file, target_id="file_eval"),
        AgentAction(action_type=ActionType.run_split_check, target_id="claim_001"),
    ]:
        obs = env.step(action)
        print(f"{action.action_type}: reward={obs.reward:.3f}")
        print(env.audit_state.last_action_result)
    evidence_ids = [e.id for e in env.audit_state.evidence if e.observed]
    obs = env.step(
        AgentAction(
            action_type=ActionType.submit_verdict,
            verdict=ValidationVerdict.NOT_SUPPORTED_METHOD_INVALID,
            failure_type=FailureType.split_mismatch,
            evidence_ids=evidence_ids,
            explanation="The paper claims test accuracy, but evaluate.py uses validation.",
        )
    )
    print(f"submit_verdict: reward={obs.reward:.3f} done={obs.done}")
    print(obs.metadata["reward_breakdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
