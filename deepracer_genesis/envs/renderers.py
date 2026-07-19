"""Rendering strategy for the DeepRacer env.

The env holds ONE ``Renderer`` and never branches on ``vision`` itself: the
strategy decides whether there is a camera observation, which Genesis scene
renderer to use, how to build the cameras/lights/sensors, how to produce the
per-step image, and how to render the debug (spectator / top-down) views.

- :class:`NullRenderer` — feature / no-vision. No camera observation; the
  optional spectator debug view still works (rasterizer). Top-down is
  vision-only.
- :class:`MadronaRenderer` — batch-renderer camera obs (+ camera-mount DR).
- :class:`NyxRenderer` — Nyx path-tracer sensor obs (true texture colors).

``make_renderer(env_cfg)`` picks the strategy from the config.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch

import genesis as gs

if TYPE_CHECKING:
    from .base_env import DeepRacerEnv

# RGB <-> YIQ (luma / chroma) for the world-color DR remap
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114],
                         [0.596, -0.274, -0.322],
                         [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.tensor([[1.0, 0.956, 0.621],
                         [1.0, -0.272, -0.647],
                         [1.0, -1.106, 1.703]])


def _u(lo, hi, shape, device):
    return lo + (hi - lo) * torch.rand(shape, device=device)


def camera_offset_T(pitch_deg: float) -> np.ndarray:
    """camera_link frame → Genesis camera frame (camera looks along -z)."""
    base = np.array([
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    p = math.radians(pitch_deg)  # positive pitches the view down
    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(p), -math.sin(p)],
        [0.0, math.sin(p), math.cos(p)],
    ])
    T = np.eye(4)
    T[:3, :3] = base @ rx
    return T


def _track_extent(track):
    """``(center_xy, max_extent)`` of a Track's centerline."""
    c = track.center.mean(dim=0)
    extent = (track.center.max(dim=0).values - track.center.min(dim=0).values).max()
    return c, extent


def make_renderer(env_cfg: dict) -> "Renderer":
    """Pick the rendering strategy from the config."""
    if not env_cfg["vision"]:
        return NullRenderer()
    if env_cfg.get("vision_renderer", "batch") == "nyx":
        return NyxRenderer()
    return MadronaRenderer()


class Renderer:
    """Base strategy: no camera observation, optional spectator debug view."""

    has_camera: bool = False
    merge_fixed_links: bool = True
    _scene_batch_renderer: bool = False
    _spectator_debug: bool = False

    def scene_renderer(self):
        """The Genesis scene renderer (BatchRenderer only for Madrona obs)."""
        return (gs.renderers.BatchRenderer(use_rasterizer=True)
                if self._scene_batch_renderer else gs.renderers.Rasterizer())

    # ---------------------------------------------------------- build lifecycle
    def build(self, env: "DeepRacerEnv", env_cfg: dict) -> None:
        """Pre-build: add cameras / lights / sensors to ``env.scene``."""
        self.spec_cam = None
        if env_cfg.get("spectator", False):
            # high-res bird's-eye view (rasterizer, true colors, all cars in one
            # image). With a BatchRenderer active it must be a debug camera to
            # stay off the batch pipeline (Madrona sets _spectator_debug=True).
            c, extent = _track_extent(env.track.tracks[0])
            c = c.cpu().numpy()
            sw, sh = env_cfg.get("spectator_res", (1280, 960))
            self.spec_cam = env.scene.add_camera(
                res=(sw, sh),
                pos=(float(c[0]), float(c[1]), float(extent) * 1.1),
                lookat=(float(c[0]), float(c[1]), 0.0),
                up=(0.0, 1.0, 0.0), fov=60, GUI=False, debug=self._spectator_debug)
        self._build(env, env_cfg)

    def _build(self, env: "DeepRacerEnv", env_cfg: dict) -> None:
        """Subclass hook: add the observation camera / sensors + top-down cam."""

    def finalize(self, env: "DeepRacerEnv", env_cfg: dict) -> None:
        """Post-build: attach cameras, set poses, init appearance/obs state."""

    # ------------------------------------------------- per-step / per-episode
    def render(self, env: "DeepRacerEnv"):
        """``(full_image, obs_image)`` both ``(N, 3, H, W)``, or ``(None, None)``."""
        return None, None

    def resample_appearance(self, env_ids: torch.Tensor) -> None:
        """Per-episode world-color redraw (vision renderers only)."""

    def randomize_mount(self, env: "DeepRacerEnv", env_ids: torch.Tensor) -> None:
        """Per-episode camera-mount jitter (Madrona only)."""

    # ------------------------------------------------------------- debug views
    def topdown(self, env: "DeepRacerEnv") -> torch.Tensor:
        raise NotImplementedError("top-down view requires a vision renderer")

    def spectator(self, env: "DeepRacerEnv") -> np.ndarray:
        assert self.spec_cam is not None, "spectator camera not enabled (cfg['spectator'])"
        rgb = np.asarray(self.spec_cam.render(rgb=True)[0])
        return rgb.reshape(rgb.shape[-3:])


class NullRenderer(Renderer):
    """Feature / no-vision: state observations only (no camera)."""


class _CameraRenderer(Renderer):
    """Shared vision base: world-color remap, pixel noise, policy-res downscale."""

    has_camera = True

    def finalize(self, env: "DeepRacerEnv", env_cfg: dict) -> None:
        self.rg_swap = bool(env_cfg.get("madrona_rg_swap", False))
        self._device = env.device
        appearance = env_cfg.get("appearance") or {}
        self.world_color_s = float(appearance.get("world_color", 0.0))
        self.policy_res = env_cfg.get("policy_res") or env_cfg["camera_res"]
        self._camera_res = env_cfg["camera_res"]
        self._pixel_noise = float(env_cfg.get("pixel_noise", 0.0))
        if self.world_color_s > 0:
            # per-env, EPISODE-static color remap (resampled each reset): each
            # agent sees the same world through its own random palette
            n = env.num_envs
            self.color_mat = torch.eye(3, device=env.device).repeat(n, 1, 1)
            self.color_bias = torch.zeros(n, 1, 3, device=env.device)

    def _acquire_rgb(self, env: "DeepRacerEnv") -> torch.Tensor:
        raise NotImplementedError

    def render(self, env: "DeepRacerEnv"):
        rgb = self._acquire_rgb(env)                          # (N, H, W, 3) uint8
        imgf = rgb.float().div_(255.0)
        if self.world_color_s > 0:
            # color remap in native NHWC: (N, H*W, 3) is a free view here, and
            # the tall-skinny batched GEMM is ~10x cheaper than any NCHW form
            n, h, w, c = imgf.shape
            imgf = ((imgf.view(n, h * w, c) @ self.color_mat.transpose(1, 2)
                     + self.color_bias).clamp_(0.0, 1.0).view(n, h, w, c))
        img = imgf.permute(0, 3, 1, 2)
        if self._pixel_noise > 0:
            img = (img + torch.randn_like(img) * self._pixel_noise).clamp(0, 1)
        if tuple(self.policy_res) != tuple(self._camera_res):
            # rendering above the policy's resolution (demo videos); the policy
            # still receives a downscaled frame
            pw, ph = self.policy_res
            obs = torch.nn.functional.interpolate(img, size=(ph, pw), mode="area")
        else:
            obs = img
        return img, obs

    def resample_appearance(self, env_ids: torch.Tensor) -> None:
        """Fresh per-env color remap: hue rotation + saturation/value scaling +
        channel mixing + bias, strength-scaled. Invertible by construction, so
        distinct scene features stay distinct — the world looks different, the
        task stays readable."""
        if self.world_color_s <= 0:
            return
        n = len(env_ids)
        s = self.world_color_s
        dev = self._device
        theta = (torch.rand(n, device=dev) * 2 - 1) * math.pi * s
        cos, sin = torch.cos(theta), torch.sin(theta)
        rot = torch.zeros(n, 3, 3, device=dev)
        rot[:, 0, 0] = 1.0
        rot[:, 1, 1] = cos; rot[:, 1, 2] = -sin
        rot[:, 2, 1] = sin; rot[:, 2, 2] = cos
        sat = 1.0 + (torch.rand(n, device=dev) * 2 - 1) * 0.6 * s
        val = 1.0 + (torch.rand(n, device=dev) * 2 - 1) * 0.35 * s
        scale = torch.zeros(n, 3, 3, device=dev)
        scale[:, 0, 0] = val
        scale[:, 1, 1] = sat; scale[:, 2, 2] = sat
        m = _YIQ2RGB.to(dev) @ scale @ rot @ _RGB2YIQ.to(dev)
        mix = torch.randn(n, 3, 3, device=dev) * 0.08 * s
        self.color_mat[env_ids] = m + mix
        self.color_bias[env_ids] = ((torch.rand(n, 1, 3, device=dev) * 2 - 1) * 0.12 * s)


class MadronaRenderer(_CameraRenderer):
    """Batch-renderer camera obs + camera-mount domain randomization."""

    merge_fixed_links = True
    _scene_batch_renderer = True
    _spectator_debug = True

    def _build(self, env: "DeepRacerEnv", env_cfg: dict) -> None:
        env.scene.add_light(pos=(0.0, 0.0, 10.0), dir=(0.4, 0.3, -1.0),
                            directional=True, castshadow=False,
                            intensity=float(env_cfg.get("light_intensity", 6.0)))
        res = env_cfg["camera_res"]  # (W, H)
        self.cam = env.scene.add_camera(res=res, fov=env_cfg["camera_fov"], GUI=False)
        self.top_cam = None
        if env_cfg.get("topdown_camera", False):
            # per-env bird's-eye pose over each env's own track variant
            centers, heights = [], []
            for t in env.track.tracks:
                c, extent = _track_extent(t)
                centers.append(c)
                heights.append(extent * 1.2)
            ev = env.track.variant_idx
            self._top_center = torch.stack(centers)[ev]          # (N, 2)
            self._top_height = torch.stack(heights)[ev]          # (N,)
            c0 = centers[0].cpu().numpy()
            self.top_cam = env.scene.add_camera(
                res=res, pos=(float(c0[0]), float(c0[1]), float(heights[0])),
                lookat=(float(c0[0]), float(c0[1]), 0.0),
                up=(0.0, 1.0, 0.0), fov=60, GUI=False)

    def finalize(self, env: "DeepRacerEnv", env_cfg: dict) -> None:
        super().finalize(env, env_cfg)
        self.cam_offset_T = camera_offset_T(env_cfg.get("camera_pitch_deg", 0.0))
        self.cam.attach(env.car.get_link("camera_link"), self.cam_offset_T)
        if self.top_cam is not None:
            pos = torch.cat([self._top_center, self._top_height[:, None]], dim=1)
            lookat = torch.cat([self._top_center,
                                torch.zeros(env.num_envs, 1, device=env.device)], dim=1)
            up = torch.tensor([[0.0, 1.0, 0.0]], device=env.device).expand(env.num_envs, 3)
            self.top_cam.set_pose(pos=pos, lookat=lookat, up=up)

    def _acquire_rgb(self, env: "DeepRacerEnv") -> torch.Tensor:
        self.cam.move_to_attach()
        rgb = self.cam.render(rgb=True)[0]                       # (N, H, W, 3) uint8 cuda
        if self.rg_swap:
            rgb = rgb[..., [1, 0, 2]]
        return rgb

    def randomize_mount(self, env: "DeepRacerEnv", env_ids: torch.Tensor) -> None:
        cfg = env.cfg["rand"]
        jitter_deg = cfg.get("camera_pitch_jitter_deg", 0.0)
        jitter_pos = cfg.get("camera_pos_jitter_m", 0.0)
        if jitter_deg <= 0 and jitter_pos <= 0:
            return
        cam = self.cam
        base = torch.as_tensor(self.cam_offset_T, dtype=torch.float32, device=env.device)
        if cam._attached_offset_T.dim() == 2:
            cam._attached_offset_T = base.expand(env.num_envs, 4, 4).clone()
        n = len(env_ids)
        p = torch.deg2rad(_u(-jitter_deg, jitter_deg, (n,), env.device))
        rx = torch.zeros(n, 4, 4, device=env.device)
        rx[:, 0, 0] = 1.0
        rx[:, 3, 3] = 1.0
        rx[:, 1, 1] = torch.cos(p)
        rx[:, 1, 2] = -torch.sin(p)
        rx[:, 2, 1] = torch.sin(p)
        rx[:, 2, 2] = torch.cos(p)
        T = base.expand(n, 4, 4).clone() @ rx
        T[:, :3, 3] += _u(-jitter_pos, jitter_pos, (n, 3), env.device)
        cam._attached_offset_T[env_ids] = T

    def topdown(self, env: "DeepRacerEnv") -> torch.Tensor:
        assert self.top_cam is not None
        rgb = self.top_cam.render(rgb=True)[0]
        return rgb[..., [1, 0, 2]] if self.rg_swap else rgb


class NyxRenderer(_CameraRenderer):
    """Nyx path-tracer sensor obs (true texture colors)."""

    merge_fixed_links = False   # the Nyx exporter refuses merged fixed links
    _scene_batch_renderer = False
    _spectator_debug = False

    def _build(self, env: "DeepRacerEnv", env_cfg: dict) -> None:
        import gs_nyx.nyx_py_renderer as npr
        import gs_nyx.nyx_py_sdk as nps
        from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

        sun = {"type": "directional", "dir": (0.4, 0.3, -1.0), "color": (1.0, 1.0, 1.0),
               "intensity": float(env_cfg.get("nyx_light_intensity", 3.0)), "shadow": False}
        mode = getattr(npr.ERenderMode, env_cfg.get("nyx_mode", "Forward"))
        res = env_cfg["camera_res"]
        # denoise/AA off: their temporal history smears moving objects across
        # frames — bad for RL observations and for validation diffs
        common = dict(spp=int(env_cfg.get("nyx_spp", 4)), render_mode=mode, lights=[sun],
                      denoise=False, anti_aliasing=nps.EAntiAliasing.Off)
        # same link->camera mount transform as the Madrona path (looks along -z
        # of offset_T incl. the downward pitch); sensors ignore pos/euler offset
        self.nyx_cam = env.scene.add_sensor(NyxCameraOptions(
            res=res, fov=env_cfg["camera_fov"],
            entity_idx=env.car.idx,
            link_idx_local=env.car.get_link("camera_link").idx_local,
            offset_T=camera_offset_T(env_cfg.get("camera_pitch_deg", 0.0)),
            **common))
        self.nyx_top = None
        if env_cfg.get("topdown_camera", False):
            c, extent = _track_extent(env.track.tracks[0])
            c = c.cpu().numpy()
            self.nyx_top = env.scene.add_sensor(NyxCameraOptions(
                res=res, fov=60,
                pos=(float(c[0]), float(c[1]), float(extent) * 1.2),
                lookat=(float(c[0]), float(c[1]), 0.0), up=(0.0, 1.0, 0.0),
                **common))

    def _acquire_rgb(self, env: "DeepRacerEnv") -> torch.Tensor:
        return self.nyx_cam.read().rgb[..., :3]                  # (N, H, W, 3) uint8 cuda

    def topdown(self, env: "DeepRacerEnv") -> torch.Tensor:
        assert self.nyx_top is not None
        return self.nyx_top.read().rgb[..., :3]
