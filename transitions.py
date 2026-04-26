"""Transition-layer compatibility helpers for ReproPilot.

The transition implementation lives on ``ReproPilotEnvironment.step`` so OpenEnv
can call it directly. This module gives the project an explicit transition
surface for tests, docs, and future extension.
"""

try:
    from .models import AgentAction, ReproPilotObservation
    from .server.repropilot_environment import ReproPilotEnvironment
except ImportError:
    from models import AgentAction, ReproPilotObservation
    from server.repropilot_environment import ReproPilotEnvironment


def apply_action(
    env: ReproPilotEnvironment, action: AgentAction
) -> ReproPilotObservation:
    return env.step(action)


__all__ = ["apply_action"]
