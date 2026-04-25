"""Compatibility exports for ReproPilot reward functions."""

try:
    from ..rewards import compute_terminal_reward, shaping_reward
except ImportError:
    from rewards import compute_terminal_reward, shaping_reward

__all__ = ["compute_terminal_reward", "shaping_reward"]
