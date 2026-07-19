"""Reward functions: swappable, written in plain torch — passed as parameters.

A reward function maps the env to a dict of NAMED PER-STEP TERMS — (N,) CUDA
tensors, one value per parallel car. Which terms count, and how much, is the
`reward_scales` dict (spec-level, sweepable); the weighted sum is the step
reward and every term is logged per episode (`Episode/rew_<name>`).

Write your own and pass it straight in — no registration:

    import torch
    from deepracer_genesis.envs.rewards import RewardFn  # (just the type alias)

    def time_trial(env) -> dict[str, torch.Tensor]:
        return {
            "progress": env.d_progress,                       # meters this step
            "alive": torch.full_like(env.d_progress, -env.dt) # ticking clock
        }

    ... >> RewardShaping(fn=time_trial, scales={"progress": 10.0, "alive": 1.0})

Everything is batched torch on GPU — same speed as the built-in. Useful env
attributes (all (N,) tensors, driving-direction aware): d_progress, lateral,
half_width, heading_err, v_forward, v_lateral, yaw_rate, actions, last_actions,
dt, plus anything else on DeepRacerEnv.

The spec records a reward fn by its NAME (``__qualname__``) for the run-dir id
and the run record — rename the fn (or change its scales) to get a new run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:
    from .base_env import DeepRacerEnv

#: a reward fn maps the env to ``{term_name: (N,) tensor}``
RewardFn = Callable[["DeepRacerEnv"], "dict[str, torch.Tensor]"]


def deepracer(env: "DeepRacerEnv") -> dict[str, torch.Tensor]:
    """The default shaping: progress-dominated with stability terms."""
    on_track = env.lateral.abs() < (env.half_width - env.cfg["wheel_margin"])
    return {
        "progress": env.d_progress,
        "speed": env.v_forward.clamp(0.0, env.cfg["max_speed"]) * env.dt,
        "centered": torch.exp(-((env.lateral / env.half_width.clamp(min=0.1)) ** 2)) * env.dt,
        "heading": -env.heading_err.abs() * env.dt,
        "steering": -env.actions[:, 0].abs() * env.dt,
        "action_rate": -((env.actions - env.last_actions) ** 2).sum(dim=1) * env.dt,
        "off_track": (~on_track).float() * env.dt,
    }
