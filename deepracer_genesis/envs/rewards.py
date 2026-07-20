"""Swappable reward functions, written in plain torch and passed as parameters.

Each fn maps the env to named per-step (N,) terms weighted by ``reward_scales``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from .base_env import DeepRacerEnv

#: a reward fn maps the env to ``{term_name: (N,) tensor}``
RewardFn = Callable[["DeepRacerEnv"], "dict[str, torch.Tensor]"]


def deepracer(env: "DeepRacerEnv") -> dict[str, torch.Tensor]:
    """Compute the built-in DeepRacer shaping: progress-dominated with stability.

    Args:
        env: The live DeepRacerEnv, providing the driving-direction-aware (N,)
            per-car state tensors (lateral, half_width, d_progress, v_forward,
            heading_err, actions, last_actions, dt) and the ``cfg`` dict
            (wheel_margin, max_speed) read here.

    Returns:
        A mapping from term name to its (N,) per-car reward tensor: progress,
        speed, centered, heading, steering, action_rate, and off_track. The
        weighted sum under the spec's ``reward_scales`` is the step reward.
    """
    on_track = env.lateral.abs() < (env.half_width - env.cfg["termination"]["wheel_margin"])
    return {
        "progress": env.d_progress,
        "speed": env.v_forward.clamp(0.0, env.cfg["action"]["max_speed"]) * env.dt,
        "centered": torch.exp(-((env.lateral / env.half_width.clamp(min=0.1)) ** 2)) * env.dt,
        "heading": -env.heading_err.abs() * env.dt,
        "steering": -env.actions[:, 0].abs() * env.dt,
        "action_rate": -((env.actions - env.last_actions) ** 2).sum(dim=1) * env.dt,
        "off_track": (~on_track).float() * env.dt,
    }
