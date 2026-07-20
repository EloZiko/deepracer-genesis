"""Training algorithms: the Algorithm protocol, its registry, and shipped
implementations (PPO, PPO-Lagrangian)."""

from .lagrangian import PIDLagrangian, PPOLagrangian
from .ppo import PPO
from .protocol import ALGORITHMS, Algorithm, make_algorithm, register_algorithm

__all__ = ["ALGORITHMS", "Algorithm", "register_algorithm", "make_algorithm",
           "PPO", "PPOLagrangian", "PIDLagrangian"]
