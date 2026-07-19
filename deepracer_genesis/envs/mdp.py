"""The RL interface over the raw simulator: the reward (R) and termination (T)
of the MDP.

Each function takes the live env, reads its per-step kinematic/track-frame
attributes, and writes its per-env buffers (``rew_buf``, ``reset_buf``,
``offtrack_buf``, …). Pulling this out of the env body keeps the "what counts
as reward / a crash / a timeout" logic readable on its own; the actual reward
*terms* live in :mod:`deepracer_genesis.envs.rewards`, the predicates in
:mod:`deepracer_genesis.envs.rules`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import rules

if TYPE_CHECKING:
    from .deepracer_env import DeepRacerEnv


def compute_reward(env: "DeepRacerEnv") -> None:
    """Weighted sum of the reward fn's named terms into ``env.rew_buf`` (and the
    per-term episode sums). Raises if a scale references a term the fn omits."""
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
    """Set the termination/cost buffers for the step just taken.

    Writes ``offtrack_buf`` / ``flipped_buf`` / ``time_out_buf`` / ``reset_buf``.
    In the plain-reward path off-track|flip add ``crash_penalty`` and terminate;
    under the CMDP framing (``emit_cost``) off-track becomes a COST, only
    unrecoverable states (flip, or far off the road) terminate, and the cost
    stream is accumulated for the Lagrangian.
    """
    cfg = env.cfg
    off = rules.is_off_track(env.lateral, env.half_width, cfg["off_track_margin"])
    flipped = rules.is_flipped(env.up_z)
    env.flipped_buf = flipped
    env.time_out_buf = env.episode_length_buf >= env.max_episode_length
    if env.emit_cost:
        # CMDP framing: offtrack is a COST, not a termination — declare
        # "violate at most `budget`" instead of hand-tuning a penalty.
        # Only unrecoverable states terminate (flip, or far off the road).
        hard_off = rules.is_off_track(env.lateral, env.half_width,
                                      cfg["off_track_margin"] + 0.4)
        env.offtrack_buf = hard_off
        env.reset_buf = hard_off | flipped | env.time_out_buf
        cost = off.float() + flipped.float()
        if env.cost_fn == "offtrack_or_overspeed":
            cost += (env.v_forward > cfg.get("overspeed_limit", 3.5)).float()
        elif env.cost_fn == "crash":
            cost = flipped.float() + hard_off.float()
        env.cost_buf = cost
        env.cost_episode_sum += cost
    else:
        env.offtrack_buf = off
        env.reset_buf = off | flipped | env.time_out_buf
        # terminal penalty for genuine failures (not timeouts)
        env.rew_buf += (off | flipped).float() * cfg["crash_penalty"]
