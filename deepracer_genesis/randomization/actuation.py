"""Actuation- and image-space DR *definitions* as TorchRL transforms (Part L).

These are torchrl-pipeline classes, but their DR definitions live here so the
randomization folder is the single home for "what can be randomized". The
pipeline still imports them via a thin re-export in
``deepracer_genesis.experiment.transforms``.
"""

from __future__ import annotations

import math

import torch
from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.utils import _get_reset, _set_missing_tolerance

from .visual import RGB2YIQ, YIQ2RGB


class ImageAug(Transform):
    """Per-step image-space DR on a float [0,1] (*B, C, H, W) key.

    Params resample per step per sub-env across brightness/contrast/saturation/hue/blur/cutout/noise.

    Attributes:
        aug: augmentation config mapping each effect name to its sampling range or scale.
    """

    def __init__(self, aug: dict, in_keys=("camera",), out_keys=None):
        """Store the augmentation config; default out_keys to in_keys."""
        out_keys = list(out_keys or in_keys)
        super().__init__(in_keys=list(in_keys), out_keys=out_keys)
        self.aug = dict(aug)

    def _u(self, lo, hi, n, device):
        """Sample n broadcastable uniforms in [lo, hi)."""
        return lo + (hi - lo) * torch.rand(n, 1, 1, 1, device=device)

    def _apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        """Apply the sampled augmentations and clamp back to [0, 1]."""
        lead = img.shape[:-3]
        c, h, w = img.shape[-3:]
        x = img.reshape(-1, c, h, w).clone()
        n, dev = x.shape[0], x.device
        a = self.aug

        if "brightness" in a:
            x = x * self._u(*a["brightness"], n, dev)
        if "contrast" in a:
            mean = x.mean(dim=(-3, -2, -1), keepdim=True)
            x = (x - mean) * self._u(*a["contrast"], n, dev) + mean
        if "saturation" in a:
            gray = x.mean(dim=-3, keepdim=True)
            x = gray + (x - gray) * self._u(*a["saturation"], n, dev)
        if a.get("hue"):
            theta = (torch.rand(n, device=dev) * 2 - 1) * a["hue"] * 2 * math.pi
            cos, sin = torch.cos(theta), torch.sin(theta)
            rot = torch.zeros(n, 3, 3, device=dev)
            rot[:, 0, 0] = 1.0
            rot[:, 1, 1] = cos; rot[:, 1, 2] = -sin
            rot[:, 2, 1] = sin; rot[:, 2, 2] = cos
            m = YIQ2RGB.to(dev) @ rot @ RGB2YIQ.to(dev)            # (n,3,3)
            x = torch.einsum("nij,njhw->nihw", m, x)
        if a.get("blur"):
            sigma = float(torch.rand(()).item()) * a["blur"]
            if sigma > 0.05:
                k = self._gaussian_kernel(sigma, dev)
                pad = k.shape[-1] // 2
                blurred = torch.nn.functional.conv2d(
                    x, k.expand(c, 1, -1, -1), padding=pad, groups=c)
                mask = (torch.rand(n, 1, 1, 1, device=dev) < 0.5).float()
                x = mask * blurred + (1 - mask) * x
        if a.get("cutout"):
            active = torch.rand(n, device=dev) < a["cutout"]
            if active.any():
                x = self._cutout(x, active)
        if a.get("noise"):
            x = x + torch.randn_like(x) * a["noise"]

        return x.clamp(0.0, 1.0).reshape(*lead, c, h, w)

    @staticmethod
    def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
        """Build a normalized 2D gaussian conv kernel for the given sigma."""
        radius = max(1, int(2 * sigma))
        xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        k1 = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
        k1 = k1 / k1.sum()
        return (k1[:, None] * k1[None, :])[None, None]

    @staticmethod
    def _cutout(x: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Zero a random rectangle in each active sub-env's image."""
        n, _, h, w = x.shape
        dev = x.device
        ph = torch.randint(h // 6, h // 3 + 1, (n,), device=dev)
        pw = torch.randint(w // 6, w // 3 + 1, (n,), device=dev)
        cy = torch.randint(0, h, (n,), device=dev)
        cx = torch.randint(0, w, (n,), device=dev)
        ys = torch.arange(h, device=dev)[None, :, None]
        xs = torch.arange(w, device=dev)[None, None, :]
        inside = ((ys >= (cy - ph // 2)[:, None, None]) & (ys < (cy + ph // 2)[:, None, None])
                  & (xs >= (cx - pw // 2)[:, None, None]) & (xs < (cx + pw // 2)[:, None, None]))
        keep = ~(inside & active[:, None, None])
        return x * keep.unsqueeze(1)

    def _reset(self, tensordict, tensordict_reset):
        """Populate out_keys on the reset tensordict."""
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    _reset_on_native_autoreset = _reset


class ActionNoiseDelay(Transform):
    """Actuation DR: k-step command latency, then per-channel gaussian noise.

    Runs on the inverse action path; the ring buffer holds the last k commands, zeroed on reset.

    Attributes:
        steer_noise: gaussian noise scale applied to the steering channel.
        speed_noise: gaussian noise scale applied to the speed channel.
        delay_steps: number of steps commands are delayed before taking effect.
        buf: per-env ring buffer of recent commands, present only when delay is enabled.
    """

    def __init__(self, n_envs: int, steer_noise=0.0, speed_noise=0.0,
                 delay_steps=0, device="cpu"):   # real callers pass the env device
        """Configure noise scales and allocate the per-env delay buffer."""
        super().__init__(in_keys_inv=["action"], out_keys_inv=["action"])
        self.steer_noise = steer_noise
        self.speed_noise = speed_noise
        self.delay_steps = int(delay_steps)
        if self.delay_steps > 0:
            self.register_buffer(
                "buf", torch.zeros(n_envs, self.delay_steps, 2, device=device))

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        """Delay via the ring buffer, add noise, clamp to [-1, 1]."""
        out = action
        if self.delay_steps > 0:
            out = self.buf[:, -1].clone()
            self.buf.copy_(torch.cat([action.unsqueeze(1), self.buf[:, :-1]], dim=1))
        noise = torch.stack([
            torch.randn(out.shape[0], device=out.device) * self.steer_noise,
            torch.randn(out.shape[0], device=out.device) * self.speed_noise,
        ], dim=1)
        return (out + noise).clamp(-1.0, 1.0)

    def _reset(self, tensordict, tensordict_reset):
        """Zero the delay buffer for freshly reset sub-envs."""
        if self.delay_steps > 0:
            mask = _get_reset("_reset", tensordict).reshape(-1)
            self.buf[mask] = 0.0
        return tensordict_reset

    _reset_on_native_autoreset = _reset
