"""Scripted agents that drive the sim from privileged state (no learning)."""

from __future__ import annotations

import torch


class PrivilegedAgent:
    """Base scripted driver mapping sim state to (N, 2) [steering, throttle]."""

    def act(self, sim) -> torch.Tensor:
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        pass


class CenterlineFollower(PrivilegedAgent):
    """Deterministic P-controller on lateral offset and heading error."""

    def __init__(self, k_lateral: float = 1.1, k_heading: float = 0.9,
                 throttle: float = -0.3):
        """Configure the P-controller.

        Args:
            k_lateral: Gain on the width-normalized lateral offset.
            k_heading: Gain on sin(heading error).
            throttle: Constant throttle command in [-1, 1] (-1 = slowest).
        """
        self.k_lateral = k_lateral
        self.k_heading = k_heading
        self.throttle = throttle

    def act(self, sim) -> torch.Tensor:
        lat = sim.lateral * sim.dir_sign / sim.half_width.clamp(min=0.1)
        steer = -(self.k_lateral * lat
                  + self.k_heading * torch.sin(sim.heading_err))
        return torch.stack(
            [steer, torch.full_like(steer, self.throttle)], dim=1).clamp(-1, 1)


class NoisyExpert(CenterlineFollower):
    """CenterlineFollower with temporally-correlated Ornstein-Uhlenbeck noise."""

    def __init__(self, noise: float = 0.35, theta: float = 0.05, **kwargs):
        """Configure the OU noise on top of the follower gains.

        Args:
            noise: Stationary standard deviation of the OU process.
            theta: Mean-reversion rate (smaller = longer excursions).
            **kwargs: Forwarded to CenterlineFollower (k_lateral, k_heading,
                throttle).
        """
        super().__init__(**kwargs)
        self.noise = noise
        self.theta = theta
        self._ou: torch.Tensor | None = None

    def act(self, sim) -> torch.Tensor:
        base = super().act(sim)
        if self._ou is None or self._ou.shape[0] != base.shape[0]:
            self._ou = torch.zeros_like(base)
        self._ou.mul_(1.0 - self.theta).add_(
            torch.randn_like(self._ou),
            alpha=self.noise * (2 * self.theta) ** 0.5)
        return (base + self._ou).clamp(-1.0, 1.0)

    def reset(self, env_ids: torch.Tensor) -> None:
        if self._ou is not None:
            self._ou[env_ids] = 0.0
