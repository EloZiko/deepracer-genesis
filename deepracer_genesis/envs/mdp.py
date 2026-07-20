"""Define the MDP interface over the raw simulator: reward (R) and termination (T).

Each function reads the live env's per-step attributes and writes its per-env buffers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import rules

if TYPE_CHECKING:
    from .deepracer_env import DeepRacerEnv


def compute_reward(env: "DeepRacerEnv") -> None:
    """Accumulate the weighted reward terms into the env's reward buffer.

    Zeroes ``env.rew_buf``, then adds each ``scale * terms[name]`` into it and ``episode_sums``.

    Args:
        env: The live DeepRacer env whose ``reward_terms`` callable, per-term
            ``reward_scales``, ``rew_buf``, and ``episode_sums`` are read/written.

    Raises:
        KeyError: If ``reward_scales`` references a term name the reward fn did
            not produce.
    """
    terms = env.reward_terms(env)          # named per-step terms (rewards.py)
    env.rew_buf.zero_()
    for name, scale in env.reward_scales.items():
        try:
            r = terms[name] * scale
        except KeyError:
            raise KeyError(
                f"reward_scales references term {name!r} but the reward fn "
                f"produced {sorted(terms)}") from None
        env.rew_buf += r
        env.episode_sums[name] += r


def check_termination(env: "DeepRacerEnv") -> None:
    """Set the termination (and, under the CMDP framing, cost) buffers for the step.

    Evaluates the off-track and flipped predicates and writes the per-env termination buffers.

    Args:
        env: The live DeepRacer env; reads ``lateral``, ``half_width``, ``up_z``,
            ``v_forward``, ``episode_length_buf``, ``max_episode_length``,
            ``emit_cost``, ``cost_fn``, and ``cfg``, and writes the termination
            (and cost) buffers listed above.
    """
    cfg = env.cfg
    off = rules.is_off_track(env.lateral, env.half_width, cfg["termination"]["off_track_margin"])
    flipped = rules.is_flipped(env.up_z)
    env.flipped_buf = flipped
    env.time_out_buf = env.episode_length_buf >= env.max_episode_length
    if env.emit_cost:
        # CMDP framing: offtrack is a COST, not a termination — declare
        # "violate at most `budget`" instead of hand-tuning a penalty.
        # Only unrecoverable states terminate (flip, or far off the road).
        hard_off = rules.is_off_track(env.lateral, env.half_width,
                                      cfg["termination"]["off_track_margin"] + 0.4)
        env.offtrack_buf = hard_off
        env.reset_buf = hard_off | flipped | env.time_out_buf
        cost = off.float() + flipped.float()
        if env.cost_fn == "offtrack_or_overspeed":
            cost += (env.v_forward > cfg["termination"].get("overspeed_limit", 3.5)).float()
        elif env.cost_fn == "crash":
            cost = flipped.float() + hard_off.float()
        env.cost_buf = cost
        env.cost_episode_sum += cost
    else:
        env.offtrack_buf = off
        env.reset_buf = off | flipped | env.time_out_buf
        # terminal penalty for genuine failures (not timeouts)
        env.rew_buf += (off | flipped).float() * cfg["termination"]["crash_penalty"]
