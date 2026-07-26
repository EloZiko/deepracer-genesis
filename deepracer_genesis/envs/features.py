"""Build the policy/CNN feature vector from the live env state.

Ships ``ClassicFeatures`` and ``PerceptionFeatures``; pass a FeatureSet subclass.
"""

from __future__ import annotations

import math

import torch

#: fixed normalization constants + physical caps — the single source of truth
#: is physics/limits.py; re-exported here so `from ...features import MAX_SPEED`
#: (etc.) keeps working. NOMINAL_WHEELBASE is now the exact URDF wheelbase
#: (0.163974, was a rounded 0.164).
from ..physics.limits import (  # noqa: E402
    A_LAT_NORM, BETA_NORM, CURVATURE_NORM, MAX_SPEED, MAX_STEER_RAD, MIN_SPEED,
    YAW_RATE_NORM, WHEELBASE_M as NOMINAL_WHEELBASE,
)


def resolve_feature_set(feature_set):
    """Return the FeatureSet class to use, defaulting ``None`` to ClassicFeatures.

    Args:
        feature_set: A FeatureSet subclass, or None for the default.

    Returns:
        The FeatureSet subclass (ClassicFeatures when given None).
    """
    return feature_set if feature_set is not None else ClassicFeatures


def feature_dim(feature_set, *, lookahead_k: int = 10,
                params: dict | None = None) -> int:
    """Return a feature set's vector width without building an env.

    Args:
        feature_set: A FeatureSet subclass, or None for the default.
        lookahead_k: Number of lookahead waypoints the set may use.
        params: Set-specific parameters.

    Returns:
        The state-vector width the set produces.
    """
    return resolve_feature_set(feature_set).dim_for(
        lookahead_k=lookahead_k, params=dict(params or {}))


def feature_layout(feature_set, *, lookahead_k: int = 10,
                   params: dict | None = None) -> str:
    """Return a human-readable channel layout (model cards / dataset metas).

    Args:
        feature_set: A FeatureSet subclass, or None for the default.
        lookahead_k: Number of lookahead waypoints the set may use.
        params: Set-specific parameters.

    Returns:
        A string describing each channel of the state vector.
    """
    return resolve_feature_set(feature_set).layout_for(
        lookahead_k=lookahead_k, params=dict(params or {}))


class FeatureSet:
    """One feature-vector recipe bound to a live env.

    Subclasses implement :meth:`compute` and clear history in :meth:`reset`.

    Attributes:
        env: the live env the features are read from.
        params: set-specific configuration for this recipe.
        cnn_target_slice: channel range a deployed CNN must predict, or None.
    """

    def __init__(self, env, params: dict):
        self.env = env
        self.params = params

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def compute(self) -> torch.Tensor:
        """Return the (N, dim) feature tensor for the current step."""
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear any per-env history for freshly respawned envs."""

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        raise NotImplementedError

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        raise NotImplementedError

    #: (start, end) slice of channels a deployed CNN must predict; None if
    #: the whole vector is privileged-only (no pixel-readable subset)
    cnn_target_slice: tuple[int, int] | None = None


class ClassicFeatures(FeatureSet):
    """The original vector: pose/velocity + body-frame look-ahead waypoints.

    Attributes:
        env: the live env the features are read from.
        params: set-specific configuration for this recipe.
    """

    @property
    def dim(self) -> int:
        return self.dim_for(lookahead_k=self.env.lookahead_k, params=self.params)

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        return 8 + 2 * lookahead_k

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        return ("v_forward/4, v_lateral, yaw_rate/5, lateral/half_width, "
                "sin(heading_err), cos(heading_err), last_action[2], "
                f"lookahead_rel_x[{lookahead_k}]/scale, "
                f"lookahead_rel_y[{lookahead_k}]/scale")

    def compute(self) -> torch.Tensor:
        env = self.env
        cy, sy = torch.cos(env.yaw), torch.sin(env.yaw)
        la_idx = env.track.lookahead(env.wp_idx, env.lookahead_k,
                                     env.cfg["obs"]["lookahead_stride"],
                                     dir_sign=env.dir_sign)
        la_pts = env.track.lookahead_points(la_idx)              # (N, K, 2)
        rel = la_pts - env.base_pos[:, None, :2]
        rel_x = rel[..., 0] * cy[:, None] + rel[..., 1] * sy[:, None]
        rel_y = -rel[..., 0] * sy[:, None] + rel[..., 1] * cy[:, None]
        la_scale = env.cfg["obs"]["lookahead_scale"]
        return torch.cat(
            [
                (env.v_forward / env.cfg["action"]["max_speed"]).unsqueeze(1),
                env.v_lateral.unsqueeze(1),
                (env.yaw_rate / YAW_RATE_NORM).unsqueeze(1),
                (env.lateral * env.dir_sign
                 / env.half_width.clamp(min=0.1)).unsqueeze(1),
                torch.sin(env.heading_err).unsqueeze(1),
                torch.cos(env.heading_err).unsqueeze(1),
                env.actions,
                rel_x / la_scale,
                rel_y / la_scale,
            ],
            dim=1,
        )


class PerceptionFeatures(FeatureSet):
    """CNN targets + action-conditioned error channels.

    Optional params: ``horizons``, ``k_prev``, ``k_speed``, ``k_steer``.

    Attributes:
        env: the live env the features are read from.
        params: set-specific configuration for this recipe.
        horizons: look-ahead distances (m) for curvature target channels.
        k_prev: number of previous actions kept in the history.
        k_speed: number of past speed-error samples kept.
        k_steer: number of past steer- and yaw-error samples kept.
    """

    def __init__(self, env, params: dict):
        super().__init__(env, params)
        self.horizons = tuple(params.get("horizons", (1.0, 3.0)))
        self.k_prev = int(params.get("k_prev", 2))
        self.k_speed = int(params.get("k_speed", 2))
        self.k_steer = int(params.get("k_steer", 8))
        n, dev = env.num_envs, env.device
        self._prev_actions = torch.zeros(n, self.k_prev, 2, device=dev)
        self._speed_err = torch.zeros(n, self.k_speed, device=dev)
        self._steer_err = torch.zeros(n, self.k_steer, device=dev)
        self._yaw_err = torch.zeros(n, self.k_steer, device=dev)

    # ---- static shape info -------------------------------------------
    @staticmethod
    def _counts(params: dict) -> tuple[int, int, int, int]:
        h = len(tuple(params.get("horizons", (1.0, 3.0))))
        return (5 + h, int(params.get("k_prev", 2)),
                int(params.get("k_speed", 2)), int(params.get("k_steer", 8)))

    @property
    def dim(self) -> int:
        return self.dim_for(lookahead_k=0, params=self.params)

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        targets, k_prev, k_speed, k_steer = cls._counts(params)
        return targets + 2 * k_prev + k_speed + 2 * k_steer

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        h = tuple(params.get("horizons", (1.0, 3.0)))
        targets, k_prev, k_speed, k_steer = cls._counts(params)
        return (f"CNN targets[{targets}]: lateral_offset, heading_err/pi, "
                f"speed/4, yaw_rate/5, sideslip_beta/{BETA_NORM}, "
                f"curvature@{list(h)}m/{CURVATURE_NORM} | policy-only: "
                f"prev_action[{k_prev}x2], speed_error[{k_speed}], "
                f"steer_response_error[{k_steer}], yaw_error[{k_steer}]")

    @property
    def cnn_target_slice(self) -> tuple[int, int]:
        return (0, self._counts(self.params)[0])

    # ---- per-step ------------------------------------------------------
    def compute(self) -> torch.Tensor:
        env = self.env

        # -- CNN targets: what a camera could tell you ------------------
        lateral = env.lateral * env.dir_sign / env.half_width.clamp(min=0.1)
        heading = env.heading_err / math.pi
        speed = env.v_forward / MAX_SPEED
        yaw_rate = env.yaw_rate / YAW_RATE_NORM
        beta = torch.atan2(env.v_lateral,
                           env.v_forward.clamp(min=0.05)) / BETA_NORM
        kappa = env.track.curvature_ahead(
            env.progress_m, self.horizons, env.dir_sign) / CURVATURE_NORM

        # -- error channels vs the FIXED nominal bicycle ----------------
        steer_cmd = env.actions[:, 0] * MAX_STEER_RAD
        speed_cmd = MIN_SPEED + (env.actions[:, 1] + 1) * 0.5 * (MAX_SPEED - MIN_SPEED)
        yaw_expected = env.v_forward * torch.tan(steer_cmd) / NOMINAL_WHEELBASE
        a_lat_expected = env.v_forward * yaw_expected            # v^2 tan(d)/L
        a_lat_actual = env.v_forward * env.yaw_rate

        speed_err = (speed_cmd - env.v_forward) / MAX_SPEED
        steer_err = ((a_lat_expected - a_lat_actual) / A_LAT_NORM).clamp(-2, 2)
        yaw_err = ((yaw_expected - env.yaw_rate) / YAW_RATE_NORM).clamp(-2, 2)

        # -- roll the histories (newest first) --------------------------
        self._prev_actions = torch.cat(
            [env.actions.unsqueeze(1), self._prev_actions[:, :-1]], dim=1)
        self._speed_err = torch.cat(
            [speed_err.unsqueeze(1), self._speed_err[:, :-1]], dim=1)
        self._steer_err = torch.cat(
            [steer_err.unsqueeze(1), self._steer_err[:, :-1]], dim=1)
        self._yaw_err = torch.cat(
            [yaw_err.unsqueeze(1), self._yaw_err[:, :-1]], dim=1)

        return torch.cat(
            [
                lateral.unsqueeze(1), heading.unsqueeze(1),
                speed.unsqueeze(1), yaw_rate.unsqueeze(1), beta.unsqueeze(1),
                kappa,
                self._prev_actions.reshape(env.num_envs, -1),
                self._speed_err,
                self._steer_err,
                self._yaw_err,
            ],
            dim=1,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self._prev_actions[env_ids] = 0.0
        self._speed_err[env_ids] = 0.0
        self._steer_err[env_ids] = 0.0
        self._yaw_err[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Composable feature blocks: a feature vector is an ordered *selection* of these
# (Part K.2). Each block is one named, normalized group of channels; an
# experiment picks the blocks it wants via ``SelectFeatures`` (see
# ``experiments/base_feature_vector.py``). The block computes read the same live
# env state ClassicFeatures does, so selecting every block in order reproduces
# the classic vector exactly.

from dataclasses import dataclass          # noqa: E402
from typing import Callable, Union         # noqa: E402


@dataclass(frozen=True)
class FeatureBlock:
    """One named group of feature channels.

    Attributes:
        width: Channel count — an int, or a fn of ``lookahead_k`` for
            lookahead-sized blocks.
        compute: ``env -> (N, width)`` tensor for the current step.
        layout: Human-readable channel label(s); ``{k}`` expands to lookahead_k.
    """

    width: Union[int, Callable[[int], int]]
    compute: Callable[["object"], torch.Tensor]
    layout: str


def _block_width(width, lookahead_k: int) -> int:
    return width(lookahead_k) if callable(width) else width


def _lookahead_xy(env) -> torch.Tensor:
    """Body-frame (rel_x, rel_y) of the upcoming waypoints, (N, 2K)."""
    cy, sy = torch.cos(env.yaw), torch.sin(env.yaw)
    la_idx = env.track.lookahead(env.wp_idx, env.lookahead_k,
                                 env.cfg["obs"]["lookahead_stride"],
                                 dir_sign=env.dir_sign)
    la_pts = env.track.lookahead_points(la_idx)
    rel = la_pts - env.base_pos[:, None, :2]
    rel_x = rel[..., 0] * cy[:, None] + rel[..., 1] * sy[:, None]
    rel_y = -rel[..., 0] * sy[:, None] + rel[..., 1] * cy[:, None]
    la_scale = env.cfg["obs"]["lookahead_scale"]
    return torch.cat([rel_x / la_scale, rel_y / la_scale], dim=1)


#: the block vocabulary experiments select from (order in a selection matters)
FEATURE_BLOCKS: dict[str, FeatureBlock] = {
    "v_forward": FeatureBlock(
        1, lambda e: (e.v_forward / e.cfg["action"]["max_speed"]).unsqueeze(1),
        "v_forward/max_speed"),
    "v_lateral": FeatureBlock(1, lambda e: e.v_lateral.unsqueeze(1), "v_lateral"),
    "yaw_rate": FeatureBlock(
        1, lambda e: (e.yaw_rate / YAW_RATE_NORM).unsqueeze(1), "yaw_rate/norm"),
    "lateral": FeatureBlock(
        1, lambda e: (e.lateral * e.dir_sign / e.half_width.clamp(min=0.1)).unsqueeze(1),
        "lateral/half_width"),
    "heading": FeatureBlock(
        2, lambda e: torch.cat([torch.sin(e.heading_err).unsqueeze(1),
                                torch.cos(e.heading_err).unsqueeze(1)], dim=1),
        "sin(heading_err), cos(heading_err)"),
    "last_action": FeatureBlock(2, lambda e: e.actions, "last_action[2]"),
    "lookahead_xy": FeatureBlock(
        lambda k: 2 * k, _lookahead_xy,
        "lookahead_rel_x[{k}]/scale, lookahead_rel_y[{k}]/scale"),
}


class SelectFeatures(FeatureSet):
    """A feature vector assembled from a chosen, ordered list of blocks.

    ``params['features']`` is a tuple of :data:`FEATURE_BLOCKS` names; the
    vector is their concatenation in that order. This is how an experiment
    chooses exactly which features it uses (Part K.2) without a bespoke
    ``FeatureSet`` subclass.

    Attributes:
        env: the live env the features are read from.
        params: must contain ``features`` (the ordered block names).
    """

    @staticmethod
    def _names(params: dict) -> tuple[str, ...]:
        names = tuple(params.get("features", ()))
        if not names:
            raise ValueError(
                "SelectFeatures needs params['features'] = a non-empty tuple of "
                f"block names from {sorted(FEATURE_BLOCKS)}")
        unknown = [n for n in names if n not in FEATURE_BLOCKS]
        if unknown:
            raise ValueError(
                f"unknown feature block(s) {unknown}; available: "
                f"{sorted(FEATURE_BLOCKS)}")
        return names

    @property
    def dim(self) -> int:
        return self.dim_for(lookahead_k=self.env.lookahead_k, params=self.params)

    @classmethod
    def dim_for(cls, *, lookahead_k: int, params: dict) -> int:
        return sum(_block_width(FEATURE_BLOCKS[n].width, lookahead_k)
                   for n in cls._names(params))

    @classmethod
    def layout_for(cls, *, lookahead_k: int, params: dict) -> str:
        return ", ".join(FEATURE_BLOCKS[n].layout.format(k=lookahead_k)
                         for n in cls._names(params))

    def compute(self) -> torch.Tensor:
        return torch.cat([FEATURE_BLOCKS[n].compute(self.env)
                          for n in self._names(self.params)], dim=1)
