"""Training algorithms: the Algorithm protocol and shipped implementations
(PPO, PPO-Lagrangian). Select one by passing the class to `Algo(cls=...)`."""

from .lagrangian import PIDLagrangian, PPOLagrangian
from .ppo import PPO
from .protocol import Algorithm

__all__ = ["Algorithm", "PPO", "PPOLagrangian", "PIDLagrangian"]
