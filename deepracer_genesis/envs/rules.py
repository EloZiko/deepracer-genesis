"""Define the stateless, batched "laws" of the DeepRacer task.

Pure tensor functions over ``N`` parallel environments, carrying no env state.
"""

from __future__ import annotations

import math

import torch


def yaw_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Extract the heading (yaw) angle from a batch of orientations.

    Args:
        q: Batch of ``(N, 4)`` orientation quaternions in wxyz order.

    Returns:
        The per-environment yaw angle in radians, shape ``(N,)``, measured
        about the world z-axis.
    """
    w, x, y, z = q.unbind(dim=1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles into the ``(-pi, pi]`` range.

    Args:
        a: Batch of angles in radians, any shape.

    Returns:
        The same tensor with each angle folded into ``(-pi, pi]``, preserving
        the input shape.
    """
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def up_z_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Compute the z-component of the body-frame up vector.

    Cheap uprightness measure: 1 is fully upright, below 0 is upside-down.

    Args:
        q: Batch of ``(N, 4)`` orientation quaternions in wxyz order.

    Returns:
        The per-environment up-vector z-component, shape ``(N,)``.
    """
    w, x, y, z = q.unbind(dim=1)
    return 1 - 2 * (x * x + y * y)


def is_off_track(lateral: torch.Tensor, half_width: torch.Tensor,
                 margin: float) -> torch.Tensor:
    """Test whether the car has left the drivable road surface.

    Args:
        lateral: Per-environment signed lateral offset from the track
            centerline, shape ``(N,)``.
        half_width: Per-environment road half-width at the car's location,
            shape ``(N,)``.
        margin: Extra tolerance added to the half-width before an environment
            counts as off-track.

    Returns:
        A boolean tensor, shape ``(N,)``, true where the absolute lateral
        offset exceeds ``half_width + margin``.
    """
    return lateral.abs() > (half_width + margin)


def is_flipped(up_z: torch.Tensor, thresh: float = 0.3) -> torch.Tensor:
    """Test whether the car has tipped over.

    Args:
        up_z: Per-environment up-vector z-component, shape ``(N,)``, as
            returned by :func:`up_z_from_quat`.
        thresh: Uprightness cutoff; an environment is considered flipped once
            ``up_z`` falls below this fraction of fully upright.

    Returns:
        A boolean tensor, shape ``(N,)``, true where ``up_z`` is below
        ``thresh``.
    """
    return up_z < thresh
