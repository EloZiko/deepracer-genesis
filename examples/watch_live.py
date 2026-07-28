"""Watch-it-live example: the interactive Genesis viewer on the GPU backend.

GPU feature env + viewer + physics DR + holdout eval + charts, on the rsl-rl
backend (which is immune to the TorchRL allocator crash; see the migration doc).
"""

from deepracer_genesis.experiment import (
    DomainRandomizationPhysics,
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)


class WatchLive(Experiment):
    """GPU feature env + interactive viewer + physics DR + holdout eval + charts.

    Attributes:
        num_envs: watchable in the viewer yet enough parallelism to learn and
            use the GPU (rsl-rl leaves plenty of headroom; see the perf review).
    """

    num_envs = 64
    seed = 0
    total_env_steps = 3_000_000
    eval_every_steps = 500_000
    ablation_group = "examples"
    variant = "watch_live"

    def pipeline(self):
        return (
            FeatureEnvironment(num_envs=self.num_envs, backend="gpu",
                               view="gui",              # GPU + viewer (Part M)
                               realtime_factor=0)       # 0 = uncapped: run as fast as the GPU allows
            >> DomainRandomizationPhysics()
            >> VectorPolicy(keys=("state",))
            >> Evaluation(real_tracks=("reinvent_base", "Oval_track"),
                          eval_num_envs=32, charts=True)             # Part N
        )


if __name__ == "__main__":
    WatchLive().run()
