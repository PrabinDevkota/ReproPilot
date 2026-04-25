"""ReproPilot Environment Client."""

from __future__ import annotations

from typing import Any

try:
    from openenv.client import EnvClient
except ImportError:
    try:
        from openenv.client.client import EnvClient
    except ImportError:
        from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AgentAction, ReproPilotObservation


class ReproPilotEnv(EnvClient[AgentAction, ReproPilotObservation, State]):
    """WebSocket client for persistent ReproPilot episodes."""

    def _step_payload(self, action: AgentAction) -> dict[str, Any]:
        return action.model_dump(mode="json", exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ReproPilotObservation]:
        obs_data = payload.get("observation", {})
        observation = ReproPilotObservation(
            echoed_message=obs_data.get("echoed_message", ""),
            message_length=obs_data.get("message_length", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: dict[str, Any]) -> State:
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))
