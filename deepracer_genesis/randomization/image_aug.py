"""Pure image-space DR applied env-side on the camera obs.

Two groups of stateless, per-frame effects (mirrors the former
``actuation.ImageAug``):

- *appearance* — brightness, contrast, saturation, hue, blur, cutout, noise.
- *photometric / geometric sensor block* (Part P.2) — barrel distortion, gamma,
  per-channel white balance, vignetting, and brightness-dependent shot noise.
  These model where the sim2real camera gap actually lives (no auto-exposure,
  colour cast, wide-angle distortion, sensor shot noise).

Every effect is opt-in via its key in the ``aug`` dict; an absent key is a
no-op, so old configs behave identically. Temporal effects (observation latency
/ frame drop) are stateful and live env-side, not here.
"""

from __future__ import annotations

import math

import torch

from .visual import RGB2YIQ, YIQ2RGB


def _u(lo, hi, n, device):
    """Sample n image-broadcastable uniforms in [lo, hi)."""
    return lo + (hi - lo) * torch.rand(n, 1, 1, 1, device=device)


def _gaussian_kernel(sigma: float, device) -> torch.Tensor:
    """Build a normalized 2D gaussian conv kernel for the given sigma."""
    radius = max(1, int(2 * sigma))
    xs = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k1 = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
    k1 = k1 / k1.sum()
    return (k1[:, None] * k1[None, :])[None, None]


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
    lo_y, hi_y = (cy - ph // 2)[:, None, None], (cy + ph // 2)[:, None, None]
    lo_x, hi_x = (cx - pw // 2)[:, None, None], (cx + pw // 2)[:, None, None]
    inside = (ys >= lo_y) & (ys < hi_y) & (xs >= lo_x) & (xs < hi_x)
    keep = ~(inside & active[:, None, None])
    return x * keep.unsqueeze(1)


def _radial_grid(h: int, w: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a base sampling grid ``(h, w, 2)`` in [-1, 1] and its radius^2.

    The grid is in ``grid_sample`` (x, y) order; ``r2`` is 0 at the centre and
    1 at the corners.
    """
    ys = torch.linspace(-1.0, 1.0, h, device=device)
    xs = torch.linspace(-1.0, 1.0, w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack((gx, gy), dim=-1)            # (h, w, 2), (x, y)
    r2 = (gx ** 2 + gy ** 2) / 2.0                  # 0 centre .. 1 corner
    return base, r2


def _barrel_distort(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Apply per-image radial (barrel/pincushion) distortion.

    Args:
        x: Image batch ``(n, c, h, w)``.
        k: Per-image radial coefficient ``(n,)``; positive = pincushion,
            negative = barrel.

    Returns:
        The resampled batch, same shape as ``x`` (edges held, not wrapped).
    """
    n, _, h, w = x.shape
    base, r2 = _radial_grid(h, w, x.device)         # (h, w, 2), (h, w)
    factor = 1.0 + k.view(n, 1, 1) * r2[None]       # (n, h, w)
    grid = base[None] * factor[..., None]           # (n, h, w, 2)
    return torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _crop_resize(x: torch.Tensor, frac: float) -> torch.Tensor:
    """Random per-image crop (up to ``frac`` off) resized back to full size.

    Models field-of-view / principal-point jitter without touching camera
    intrinsics: each image zooms into a random sub-window and is resampled to
    the original resolution.

    Args:
        x: Image batch ``(n, c, h, w)``.
        frac: Max fraction of the frame croppable per image (0 disables).

    Returns:
        The cropped-and-resized batch, same shape as ``x``.
    """
    n, _, h, w = x.shape
    dev = x.device
    scale = 1.0 - torch.rand(n, device=dev) * frac      # remaining window (<=1)
    max_t = 1.0 - scale                                  # keep the window on-frame
    tx = (torch.rand(n, device=dev) * 2 - 1) * max_t
    ty = (torch.rand(n, device=dev) * 2 - 1) * max_t
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = scale; theta[:, 0, 2] = tx
    theta[:, 1, 1] = scale; theta[:, 1, 2] = ty
    grid = torch.nn.functional.affine_grid(theta, x.shape, align_corners=False)
    return torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def apply_image_aug(img: torch.Tensor, aug: dict) -> torch.Tensor:
    """Return img with the sampled augmentations applied, clamped to [0, 1].

    Effects are applied geometric-first, then photometric, then sensor noise:
    distortion, crop, brightness, contrast, gamma, saturation, hue, white
    balance, vignette, blur, cutout, shot noise, additive noise. Each is skipped
    when its key is absent from ``aug``.

    Args:
        img: Float image tensor ``(*B, C, H, W)`` in [0, 1].
        aug: Augmentation config keyed by effect name to its range or scale.
            Ranges (``(lo, hi)`` tuples): ``brightness``, ``contrast``,
            ``saturation``, ``gamma``. Scalar magnitudes: ``hue``, ``blur``,
            ``cutout``, ``noise``, ``distortion``, ``crop``, ``white_balance``,
            ``vignette``, ``shot_noise``.

    Returns:
        The augmented image, same shape as ``img``.
    """
    if not aug:
        return img
    lead = img.shape[:-3]
    c, h, w = img.shape[-3:]
    x = img.reshape(-1, c, h, w).clone()
    n, dev = x.shape[0], x.device

    if aug.get("distortion"):
        k = (torch.rand(n, device=dev) * 2 - 1) * aug["distortion"]
        x = _barrel_distort(x, k)
    if aug.get("crop"):
        x = _crop_resize(x, aug["crop"])
    if "brightness" in aug:
        x = x * _u(*aug["brightness"], n, dev)
    if "contrast" in aug:
        mean = x.mean(dim=(-3, -2, -1), keepdim=True)
        x = (x - mean) * _u(*aug["contrast"], n, dev) + mean
    if "gamma" in aug:
        x = x.clamp(min=0.0) ** _u(*aug["gamma"], n, dev)
    if "saturation" in aug:
        gray = x.mean(dim=-3, keepdim=True)
        x = gray + (x - gray) * _u(*aug["saturation"], n, dev)
    if aug.get("hue"):
        theta = (torch.rand(n, device=dev) * 2 - 1) * aug["hue"] * 2 * math.pi
        cos, sin = torch.cos(theta), torch.sin(theta)
        rot = torch.zeros(n, 3, 3, device=dev)
        rot[:, 0, 0] = 1.0
        rot[:, 1, 1] = cos; rot[:, 1, 2] = -sin
        rot[:, 2, 1] = sin; rot[:, 2, 2] = cos
        m = YIQ2RGB.to(dev) @ rot @ RGB2YIQ.to(dev)
        x = torch.einsum("nij,njhw->nihw", m, x)
    if aug.get("white_balance"):
        gains = 1.0 + (torch.rand(n, c, 1, 1, device=dev) * 2 - 1) * aug["white_balance"]
        x = x * gains
    if aug.get("vignette"):
        strength = torch.rand(n, 1, 1, 1, device=dev) * aug["vignette"]
        _, r2 = _radial_grid(h, w, dev)
        x = x * (1.0 - strength * r2[None, None])
    if aug.get("blur"):
        sigma = float(torch.rand(()).item()) * aug["blur"]
        if sigma > 0.05:
            k = _gaussian_kernel(sigma, dev)
            pad = k.shape[-1] // 2
            blurred = torch.nn.functional.conv2d(
                x, k.expand(c, 1, -1, -1), padding=pad, groups=c)
            mask = (torch.rand(n, 1, 1, 1, device=dev) < 0.5).float()
            x = mask * blurred + (1 - mask) * x
    if aug.get("cutout"):
        active = torch.rand(n, device=dev) < aug["cutout"]
        if active.any():
            x = _cutout(x, active)
    if aug.get("shot_noise"):
        x = x + torch.sqrt(x.clamp(min=0.0)) * torch.randn_like(x) * aug["shot_noise"]
    if aug.get("noise"):
        x = x + torch.randn_like(x) * aug["noise"]

    return x.clamp(0.0, 1.0).reshape(*lead, c, h, w)
