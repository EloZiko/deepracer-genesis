"""The batched DeepRacer environment on Genesis (shared base).

``DeepRacerEnv`` is the base class *and* the factory: constructing it dispatches
by ``env_cfg["vision"]`` to :class:`~deepracer_genesis.envs.vision_env.VisionDeepRacerEnv`
or :class:`~deepracer_genesis.envs.vector_env.VectorDeepRacerEnv`. The base owns
the control/observation loop; the two subclasses differ only in the observation
contract (camera group + image buffers). All rendering — and every ``if vision``
branch — lives behind the injected :class:`~deepracer_genesis.envs.renderers.Renderer`
strategy; the car behind :class:`~deepracer_genesis.envs.entities.Car`.

Exposes the rsl-rl-lib 5.x VecEnv contract (TensorDict observation groups, no
``reset()`` from the runner — done envs respawn inside ``step()``) and is wrapped
for TorchRL by :class:`~deepracer_genesis.envs.torchrl_env.TorchRLDeepRacerEnv`.

Observation groups:
    state: ``(N, D)`` vector (:mod:`deepracer_genesis.envs.features`).
    camera: ``(N, 3, H, W)`` float in ``[0, 1]`` (vision envs only).

Actions:
    ``(N, 2)`` normalized ``[steering, throttle]`` in ``[-1, 1]`` → DeepRacer
    ``Box([-30deg, 0.1 m/s], [+30deg, 4.0 m/s])``; or ``(N,)`` action-table
    indices for discrete policies.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

import genesis as gs

from . import mdp, rules
from .scene import build_scene
from .track import MultiTrack
from ..physics.limits import YAW_RATE_NORM
from ..randomization.domain_rand import randomize_physics


class DeepRacerEnv:
    def __new__(cls, num_envs: int, env_cfg: dict, show_viewer: bool = False,
                device=None):
        # constructing the base dispatches to the vision/vector subclass
        if cls is DeepRacerEnv:
            from .vector_env import VectorDeepRacerEnv
            from .vision_env import VisionDeepRacerEnv
            cls = VisionDeepRacerEnv if env_cfg["vision"] else VectorDeepRacerEnv
        return object.__new__(cls)

    def __init__(self, num_envs: int, env_cfg: dict, show_viewer: bool = False,
                 device=None) -> None:
        self.device = torch.device(device) if device is not None else gs.device
        self.cfg = env_cfg
        self.num_envs = num_envs
        self.num_actions = 2
        self.vision = env_cfg["vision"]
        # discrete action support: a (K, 2) table of [steer, speed] pairs.
        # step() accepts index tensors (N,) and looks them up — every consumer
        # (training, eval, collection) stays agnostic.
        table = env_cfg.get("action_table")
        self.action_table = (torch.tensor(table, dtype=torch.float32,
                                          device=self.device)
                             if table else None)

        self.dt = env_cfg["dt"] * env_cfg["decimation"]  # control dt
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        names = env_cfg["track"] if isinstance(env_cfg["track"], (list, tuple)) else [env_cfg["track"]]
        self.track = MultiTrack(names, num_envs, self.device)

        build_scene(self, env_cfg, show_viewer)      # -> self.renderer, scene, car, track_entity
        self.car.configure(env_cfg, self.device)
        # mirror the car's dof handles on the env (domain randomization reads them)
        self.steer_dofs = self.car.steer_dofs
        self.wheel_dofs = self.car.wheel_dofs
        self.wheel_radius = self.car.wheel_radius
        self.steering_model = self.car.steering_model
        self.renderer.finalize(self, env_cfg)        # attach cameras, appearance/obs state
        self._init_buffers(env_cfg)

    # ------------------------------------------------------------------ hooks
    def _init_obs_buffers(self, env_cfg: dict) -> None:
        """Subclass hook: preallocate camera image buffers (vision only)."""

    def _observe_camera(self) -> None:
        """Subclass hook: refresh the camera observation (vision only)."""

    def _obs_groups(self) -> dict:
        """Observation groups for :meth:`get_observations` (base: state only)."""
        return {"state": self.state_buf}

    @property
    def spec_cam(self):
        """The spectator debug camera (or None); lives on the renderer."""
        return self.renderer.spec_cam

    # ---------------------------------------------------------------- buffers
    def _init_buffers(self, env_cfg: dict) -> None:
        """Per-env episode state (reward/termination/DR buffers, the state
        vector) + reward-function resolution."""
        N = self.num_envs
        self.episode_length_buf = torch.zeros(N, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(N, device=self.device, dtype=torch.bool)
        self.rew_buf = torch.zeros(N, device=self.device)
        self.time_out_buf = torch.zeros(N, device=self.device, dtype=torch.bool)
        self.actions = torch.zeros(N, 2, device=self.device)
        self.last_actions = torch.zeros(N, 2, device=self.device)
        self.progress_m = torch.zeros(N, device=self.device)
        self.laps = torch.zeros(N, device=self.device)
        # per-env driving direction: +1 follows waypoint order (counter-
        # clockwise on re:Invent tracks), -1 drives the track reversed.
        self.dir_sign = torch.ones(N, device=self.device)
        self.offtrack_buf = torch.zeros(N, device=self.device, dtype=torch.bool)
        self.flipped_buf = torch.zeros(N, device=self.device, dtype=torch.bool)
        self.emit_cost = bool(env_cfg.get("emit_cost", False))
        self.cost_fn = env_cfg.get("cost_fn") or "offtrack"
        if self.emit_cost:
            self.cost_buf = torch.zeros(N, device=self.device)
            self.cost_episode_sum = torch.zeros(N, device=self.device)
        self.extras = {"log": {}}

        self.lookahead_k = env_cfg["lookahead_k"]
        self.num_state_obs = 8 + 2 * self.lookahead_k
        self.state_buf = torch.zeros(N, self.num_state_obs, device=self.device)
        self._init_obs_buffers(env_cfg)

        from .rewards import deepracer
        reward_fn = env_cfg.get("reward_fn") or deepracer   # None -> default
        self.reward_terms = reward_fn
        overrides = env_cfg.get("reward_scale_overrides", {})
        if reward_fn is deepracer:
            # tweaks merge over the tuned defaults
            self.reward_scales = dict(env_cfg["reward_scales"])
            self.reward_scales.update(overrides)
        else:
            # custom fn: its scales stand alone (defaults reference terms the
            # custom fn does not produce)
            if not overrides:
                name = getattr(reward_fn, "__qualname__", reward_fn)
                raise ValueError(
                    f"custom reward fn {name!r} needs explicit scales "
                    "(RewardShaping(fn=..., scales={...}))")
            self.reward_scales = dict(overrides)
        if self.emit_cost:
            # the cost stream replaces the offtrack shaping term (plan: pull
            # offtrack/crash OUT of the reward, constrain them instead)
            self.reward_scales.pop("off_track", None)
        self.episode_sums = {k: torch.zeros(N, device=self.device) for k in self.reward_scales}

        self.reset_idx(torch.arange(N, device=self.device))
        self._post_physics()

    # ------------------------------------------------------------------- step
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Advance every env by one control step (``decimation`` physics steps).

        Returns:
            ``(obs_tensordict, reward (N,), done (N,), extras)``; ``extras["log"]``
            carries per-episode stats at reset boundaries and
            ``extras["time_outs"]`` flags truncations.
        """
        if self.action_table is not None and actions.dim() == 1:
            actions = self.action_table[actions.long()]
        self.actions = torch.clip(actions, -1.0, 1.0)
        steer = self.actions[:, 0:1] * math.radians(self.cfg["max_steering_deg"])
        speed = self.cfg["min_speed"] + (self.actions[:, 1:2] + 1) * 0.5 * (
            self.cfg["max_speed"] - self.cfg["min_speed"])
        self.car.drive(steer, speed)
        for _ in range(self.cfg["decimation"]):
            self.scene.step()

        self.episode_length_buf += 1
        self._post_physics()
        mdp.compute_reward(self)
        mdp.check_termination(self)

        # pre-reset snapshot: reset_idx destroys these for done rows, but
        # wrappers/evaluators need the values of the step that just happened
        self.step_info = {
            "progress_delta": self.d_progress.clone(),
            "laps": self.laps.clone(),
            "offtrack": self.offtrack_buf.clone(),
            "flipped": self.flipped_buf.clone(),
            "time_out": self.time_out_buf.clone(),
            "terminal_state": self.state_buf.clone(),
        }
        if self.emit_cost:
            self.step_info["cost"] = self.cost_buf.clone()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self._post_physics(env_ids)

        self.last_actions[:] = self.actions
        self.extras["time_outs"] = self.time_out_buf
        # fence the genesis<->torch stream boundary: quadrants kernels run on
        # their own CUDA stream and consume torch tensors (controls, reset poses,
        # DR draws) that torch's stream-ordered allocator may otherwise recycle
        # while still in flight -> sporadic CUDA_ERROR_ILLEGAL_ADDRESS in long
        # runs. One device sync per control step bounds the race.
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    # ---------------------------------------------------------- post-physics
    def _post_physics(self, env_ids: torch.Tensor | None = None) -> None:
        """Refresh cached kinematics + track-frame quantities + observations."""
        pos, quat, vel, ang = self.car.kinematics()

        self.base_pos = pos
        self.yaw = rules.yaw_from_quat(quat)
        cy, sy = torch.cos(self.yaw), torch.sin(self.yaw)
        self.v_forward = vel[:, 0] * cy + vel[:, 1] * sy
        self.v_lateral = -vel[:, 0] * sy + vel[:, 1] * cy
        self.yaw_rate = ang[:, 2]
        self.up_z = rules.up_z_from_quat(quat)   # flip detection

        loc = self.track.localize(pos[:, :2])
        self.wp_idx = loc["wp_idx"]
        self.lateral = loc["lateral"]
        self.half_width = loc["half_width"]
        # all track-frame quantities are expressed in the env's own driving
        # direction (dir_sign): a reversed car aligned with the reversed tangent
        # has heading_err 0 and accumulates positive progress
        rev = (self.dir_sign < 0).float()
        self.heading_err = rules.wrap(self.yaw - loc["track_yaw"] - rev * math.pi)
        new_progress = loc["progress_m"]
        d = new_progress - self.progress_m
        L = self.track.total_len_env
        wrapped = torch.where(d > 0.5 * L, d - L, torch.where(d < -0.5 * L, d + L, d))
        self.d_progress = wrapped * self.dir_sign
        if env_ids is not None and len(env_ids) > 0:
            self.d_progress[env_ids] = 0.0
        # wrap through the finish line while moving forward = one lap completed
        self.laps += ((d.abs() > 0.5 * L) & (self.d_progress > 0)).float()
        self.progress_m = new_progress

        # ---- state obs ----
        la_idx = self.track.lookahead(self.wp_idx, self.lookahead_k,
                                      self.cfg["lookahead_stride"],
                                      dir_sign=self.dir_sign)
        la_pts = self.track.lookahead_points(la_idx)             # (N, K, 2)
        rel = la_pts - pos[:, None, :2]
        rel_x = rel[..., 0] * cy[:, None] + rel[..., 1] * sy[:, None]
        rel_y = -rel[..., 0] * sy[:, None] + rel[..., 1] * cy[:, None]
        la_scale = self.cfg["lookahead_scale"]
        self.state_buf = torch.cat(
            [
                (self.v_forward / self.cfg["max_speed"]).unsqueeze(1),
                self.v_lateral.unsqueeze(1),
                (self.yaw_rate / YAW_RATE_NORM).unsqueeze(1),
                # signed offset in the car's own left/right (flips when reversed)
                (self.lateral * self.dir_sign
                 / self.half_width.clamp(min=0.1)).unsqueeze(1),
                torch.sin(self.heading_err).unsqueeze(1),
                torch.cos(self.heading_err).unsqueeze(1),
                self.actions,
                rel_x / la_scale,
                rel_y / la_scale,
            ],
            dim=1,
        )
        if self.cfg.get("obs_noise", 0.0) > 0:
            self.state_buf += torch.randn_like(self.state_buf) * self.cfg["obs_noise"]

        self._observe_camera()

    # ------------------------------------------------------------------ reset
    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """Respawn the given envs and resample their per-episode DR draws.

        Spawns are random waypoints (+ lateral/yaw noise), direction is coin-
        flipped under ``random_direction``, physics/camera-mount/world-color
        randomizations are redrawn, and episode logs are emitted to
        ``extras["log"]``.
        """
        n = len(env_ids)
        if n == 0:
            return
        pos_xy, yaw = self.track.spawn_pose(
            env_ids, self.cfg["random_start"],
            lateral_noise=self.cfg["spawn_lateral_noise"], yaw_noise=self.cfg["spawn_yaw_noise"])
        if self.cfg.get("random_direction", False):
            # coin-flip the driving direction each episode; the spawn faces the
            # chosen direction and all track-frame quantities follow it
            flip = torch.rand(n, device=self.device) < 0.5
            self.dir_sign[env_ids] = torch.where(flip, -1.0, 1.0)
            yaw = yaw + flip.float() * math.pi

        self.renderer.resample_appearance(env_ids)   # world-color DR (vision only)

        qpos = torch.zeros(n, 13, device=self.device)
        qpos[:, 0:2] = pos_xy
        qpos[:, 2] = self.cfg["spawn_height"]
        qpos[:, 3] = torch.cos(yaw / 2)
        qpos[:, 6] = torch.sin(yaw / 2)
        self.car.reset_pose(qpos, env_ids)

        if self.cfg.get("randomize", False):
            randomize_physics(self, env_ids)
            self.renderer.randomize_mount(self, env_ids)   # camera-mount DR (Madrona only)

        # episode logging
        self.extras["log"] = {}
        for key, sums in self.episode_sums.items():
            self.extras["log"][f"Episode/rew_{key}"] = sums[env_ids].mean()
            sums[env_ids] = 0.0
        self.extras["log"]["Episode/length"] = self.episode_length_buf[env_ids].float().mean()
        if self.emit_cost:
            self.extras["log"]["Episode/cost"] = self.cost_episode_sum[env_ids].mean()
            self.cost_episode_sum[env_ids] = 0.0

        self.episode_length_buf[env_ids] = 0
        self.laps[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.progress_m[env_ids] = self.track.localize(pos_xy, envs_idx=env_ids)["progress_m"]
        # same stream fence as step(): reset poses/DR draws are torch temporaries
        # consumed by genesis kernels (see step() comment)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ----------------------------------------------------------- observations
    def get_observations(self) -> TensorDict:
        """Current observation TensorDict (``state`` [+ ``camera``] groups)."""
        return TensorDict(self._obs_groups(), batch_size=[self.num_envs], device=self.device)

    def render_topdown(self):
        """(N, H, W, 3) uint8 per-env bird's-eye view (validation only)."""
        return self.renderer.topdown(self)

    def render_spectator(self):
        """(H, W, 3) uint8 high-res bird's-eye view showing all envs' cars."""
        return self.renderer.spectator(self)
