"""Big rsl-rl run (watch_live config): checks for reward collapse and crashes.

Feature + physics DR, num_envs=16, 3M steps on GPU via the rsl-rl backend.
"""

from deepracer_genesis.experiment import (
    DomainRandomizationPhysics,
    Evaluation,
    FeatureEnvironment,
    VectorPolicy,
    run,
)


def main():
    spec = (
        FeatureEnvironment(num_envs=16, backend="gpu", view="none")
        >> DomainRandomizationPhysics()
        >> VectorPolicy(keys=("state",))
        >> Evaluation(real_tracks=("reinvent_base", "Oval_track"),
                      eval_num_envs=32, charts=True)
    ).build(total_env_steps=3_000_000, eval_every_steps=500_000, seed=0)
    run(spec, root="runs/big_run")


if __name__ == "__main__":
    main()
