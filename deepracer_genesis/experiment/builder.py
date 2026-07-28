"""Builder: turn a validated ExperimentSpec into the live Genesis sim.

The only layer that imports the heavy sim dependency.
"""

from __future__ import annotations

from ..configs.cfgs import get_env_cfg
from ..envs import DeepRacerEnv
from .spec import ExperimentSpec

_NEUTRAL_PHYSICS = {
    "friction_range": (1.0, 1.0),
    "mass_shift_kg": 0.0,
    "com_shift_m": 0.0,
    "steer_kp_scale": (1.0, 1.0),
    "wheel_kv_scale": (1.0, 1.0),
    "armature_range": (0.0, 0.0),
    "track_width_scale": (1.0, 1.0),
}


def _ensure_genesis(backend: str = "gpu"):
    """Initialize Genesis on the given backend (SSOT gs.init, Part M)."""
    from .._gs import ensure_init
    ensure_init(backend)


class Builder:
    """Turn a validated ExperimentSpec into the live Genesis sim (built once).

    Attributes:
        spec: The validated experiment spec driving the sim build.
    """

    def __init__(self, spec: ExperimentSpec) -> None:
        spec.validate()
        self.spec: ExperimentSpec = spec
        self._sim: DeepRacerEnv | None = None

    def sim_cfg(self) -> dict:
        """Translate the EnvSpec (+DR spec) into the sim's config dict.

        Returns:
            The nested env config the sim is constructed from.
        """
        env = self.spec.env
        obs_dr = self.spec.obs_dr
        randomize = bool(obs_dr.physics or obs_dr.camera_jitter)
        track = list(env.tracks) if len(env.tracks) > 1 else env.tracks[0]
        cfg = get_env_cfg(vision=(env.modality == "camera"), track=track,
                          randomize=randomize, backend=env.backend, view=env.view)
        cfg["vision"]["camera_res"] = tuple(env.resolution)
        cfg["vision"]["camera_fov"] = env.fov
        cfg["obs"]["lookahead_k"] = env.lookahead_k
        cfg["obs"]["feature_set"] = env.feature_set
        cfg["obs"]["feature_params"] = dict(env.feature_params)
        cfg["spawn"]["random_start"] = env.random_start
        cfg["spawn"]["random_direction"] = env.random_direction
        cfg["reward"]["reward"] = env.reward   # a callable (or None -> deepracer default)
        cfg["reward"]["reward_scale_overrides"] = dict(env.reward_scales)
        if env.render == "nyx":
            cfg["vision"]["vision_renderer"] = "nyx"
        if randomize:
            rand = dict(_NEUTRAL_PHYSICS)
            rand.update(obs_dr.physics)
            rand["camera_pitch_jitter_deg"] = obs_dr.camera_jitter.get("pitch_deg", 0.0)
            rand["camera_pos_jitter_m"] = obs_dr.camera_jitter.get("pos_m", 0.0)
            rand["randomize"] = True
            cfg["rand"] = rand
        if obs_dr.appearance:
            cfg["vision"]["appearance"] = dict(obs_dr.appearance)
        if self.spec.policy is not None and self.spec.policy.actions:
            cfg["action"]["action_table"] = [list(a) for a in self.spec.policy.actions]
        if env.emits_cost:
            cfg["reward"]["emit_cost"] = True
            cfg["reward"]["cost_fn"] = env.cost_fn
        return cfg

    def sim(self, extra_cfg: dict | None = None) -> DeepRacerEnv:
        """Build (once) and return the Genesis sim.

        Args:
            extra_cfg: Merged into the spec-derived config on FIRST construction
                only (e.g. spectator camera, env-side DR); ignored afterwards.

        Returns:
            The cached DeepRacerEnv.
        """
        if self._sim is None:
            _ensure_genesis(self.spec.env.backend)
            cfg = self.sim_cfg()
            if extra_cfg:
                cfg.update(extra_cfg)
            self._sim = DeepRacerEnv(num_envs=self.spec.env.num_envs, env_cfg=cfg)
        return self._sim
