"""Algorithm contract for plugging a custom training algorithm into the Trainer.

Satisfy the `Algorithm` protocol, register with `@register_algorithm`, and select via `Algo(kind=...)`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:  # only for annotations; keep import-time light
    from tensordict import TensorDictBase

    from .builder import Builder

ALGORITHMS: dict[str, type] = {}


def register_algorithm(kind: str):
    """Class decorator: make `kind` selectable from AlgorithmSpec.kind.

    Args:
        kind: Registry key, referenced from the DSL via
            `Algo(kind="my_kind", ...)`.

    Returns:
        The decorator; it registers and returns the class unchanged.

    Raises:
        ValueError: If `kind` is already registered (raised at decoration
            time).
    """
    def deco(cls: type) -> type:
        if kind in ALGORITHMS:
            raise ValueError(f"algorithm kind {kind!r} already registered")
        ALGORITHMS[kind] = cls
        return cls
    return deco


def make_algorithm(builder: "Builder") -> "Algorithm":
    """Resolve AlgorithmSpec.kind against the registry and set it up.

    Args:
        builder: The Builder whose spec selects the algorithm kind; passed
            through to the instance's setup().

    Returns:
        A ready (setup-complete) Algorithm instance.

    Raises:
        ValueError: If the kind is not registered; the message lists the
            registered names.
    """
    kind = builder.spec.algorithm.kind
    try:
        cls = ALGORITHMS[kind]
    except KeyError:
        raise ValueError(
            f"unknown algorithm kind {kind!r}; registered: {sorted(ALGORITHMS)} "
            "(custom algorithms register via "
            "deepracer_genesis.algorithms.register_algorithm)") from None
    algo = cls()
    algo.setup(builder)
    return algo


@runtime_checkable
class Algorithm(Protocol):
    """What the Trainer drives; see the module docstring for the guide."""

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


