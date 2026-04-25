"""Import shim for source-tree usage before editable install."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in __path__:
    __path__.append(str(_ROOT))

from models import AgentAction, ReproPilotObservation  # noqa: E402

try:
    from client import ReproPilotEnv  # noqa: E402
except Exception:  # pragma: no cover
    ReproPilotEnv = None  # type: ignore[assignment]

__all__ = ["AgentAction", "ReproPilotObservation"]
if ReproPilotEnv is not None:
    __all__.append("ReproPilotEnv")
