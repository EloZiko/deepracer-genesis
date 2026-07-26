"""Algorithm contract for plugging a custom training algorithm into the Trainer.

Satisfy the `Algorithm` protocol and pass the class directly to the DSL via
`Algo(cls=MyAlgorithm)`; the Trainer instantiates and `setup()`s it. Set the
class attribute `requires_cost = True` if the algorithm consumes a cost signal
(the spec then requires a cost-emitting env + budget, as PPO-Lagrangian does).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:  # only for annotations; keep import-time light
    from tensordict import TensorDictBase

    from .builder import Builder


@runtime_checkable
class Algorithm(Protocol):
    """What the Trainer drives; see the module docstring for the guide."""

    #: True if the algorithm consumes a cost signal (safe-RL). The spec then
    #: demands a cost-emitting env and a budget; the Trainer wires the cost
    #: channel. Plain reward-only algorithms leave this False.
    requires_cost: ClassVar[bool] = False

    def setup(self, builder: "Builder") -> None:
        """Build networks, losses and optimizers from the Builder.

        Args:
            builder: Gives you `actor()`, `critic(out_key=...)`, `gae(...)`,
                `optimizer(...)`, obs-key dims, the sim, and the spec.
        """

    @property
    def collect_policy(self):
        """Policy module the Collector runs (exploratory)."""

    @property
    def eval_actor(self):
        """Actor used for deterministic evaluation rollouts."""

    def train_on_batch(self, data: "TensorDictBase") -> dict[str, float]:
        """Consume one collector yield and take training steps.

        Args:
            data: One [N, T] collector batch (root obs/action + ("next",
                reward/done/terminated/truncated/obs)). Off-policy
                algorithms are free to stash it in their own replay buffer
                and take gradient steps at their own cadence; on-policy ones
                run their epochs/minibatches here.

        Returns:
            Dict of scalars to log.
        """

    def observe_env_logs(self, logs: dict[str, Any]) -> None:
        """Receive the sim's episode logs once per iteration (optional use).

        Args:
            logs: The sim's episode-log dict (PPO-Lagrangian reads
                "Episode/cost" here to drive the PID lambda).
        """

    def checkpoint(self) -> dict[str, Any]:
        """Extra state to persist in the run checkpoint.

        Returns:
            Mapping of name -> state_dict, saved alongside the Trainer
            payload.
        """


