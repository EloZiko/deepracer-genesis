"""GPU smoke: prove ``Algo(cls=...)`` drives a CUSTOM algorithm through rsl-rl.

Run on a machine with a CUDA GPU + Genesis::

    python scripts/verify_custom_algorithm.py

It trains a few iterations with a trivial custom algorithm that *is* PPO
(``EchoPPO``): the point is that the custom **class** routes through the SAME
``OnPolicyRunner`` PPO uses (via ``cfg["algorithm"]["class_name"]``), not the
learning rule. Once this passes, swap ``EchoPPO`` for your ``Reinforce(PPO)``
(overriding ``compute_returns``/``update``) to smoke-test the real algorithm —
no other wiring changes are needed.
"""

from __future__ import annotations

from rsl_rl.algorithms import PPO

from deepracer_genesis.experiment import Algo, FeatureEnvironment, VectorPolicy, run


class EchoPPO(PPO):
    """Trivial custom algorithm == PPO (proves the custom-class path).

    Replace with your ``Reinforce(PPO)`` — inherit ``construct_algorithm``,
    ``act``, ``get_policy``, ``save``/``load``; override ``compute_returns`` and
    ``update`` with the policy-gradient learning rule.
    """


def main() -> None:
    num_envs, horizon, iters = 64, 24, 3
    record = run(
        FeatureEnvironment(num_envs=num_envs) >> VectorPolicy() >> Algo(cls=EchoPPO),
        total_env_steps=num_envs * horizon * iters,
        root="runs/_smoke_custom_algo",
    )
    assert record is not None, "run() returned no EvalRecord"
    print("OK: custom algorithm class trained via rsl-rl OnPolicyRunner")
    print("    algorithm:", record.spec["algorithm"].get("cls"))
    print("    completion_rate:", record.metrics.get("completion_rate"))


if __name__ == "__main__":
    main()
