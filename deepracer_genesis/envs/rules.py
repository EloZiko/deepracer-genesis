"""The stateless "laws" of the DeepRacer task.

Small pure functions over tensors — orientation helpers and the off-track /
flip predicates — with no simulator, scene or env state, so they read on their
own and are unit-testable in isolation. The env and :mod:`mdp` call these.
"""

from __future__ import annotations

import math

import torch


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Heading (yaw) from a batch of wxyz quaternions ``(N, 4)`` → ``(N,)``."""
    w, x, y, z = q.unbind(dim=1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles into ``(-pi, pi]``."""
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def up_z_from_quat(q: torch.Tensor) -> torch.Tensor:
    """z-component of the body-frame up vector (1 = upright, < 0 = upside-down)."""
    w, x, y, z = q.unbind(dim=1)
    return 1 - 2 * (x * x + y * y)


def is_off_track(lateral: torch.Tensor, half_width: torch.Tensor,
                 margin: float) -> torch.Tensor:
    """True where |lateral offset| exceeds the road half-width + ``margin``."""
    return lateral.abs() > (half_width + margin)


def is_flipped(up_z: torch.Tensor, thresh: float = 0.3) -> torch.Tensor:
    """True where the car has tipped past ``thresh`` of upright."""
    return up_z < thresh
