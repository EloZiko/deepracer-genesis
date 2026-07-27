"""Watch-it-live example: the interactive Genesis viewer on the CPU backend.

Blends BOTH Part M features — ``backend="cpu"`` + ``view="gui"`` — plus physics
DR and first-class eval (Part N). Watching a run is a small, debug-scale task,
and the CPU backend is the right tool for it: it has no CUDA streams, so it is
immune to the Genesis<->torch GPU stream race that the interactive viewer +
physics DR can trigger on the GPU backend (use the GPU examples, without the
viewer, for DR *training* throughput). Class-authoring style.
"""

from deepracer_genesis.experiment import (
    DomainRandomizationPhysics,
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
)


class WatchLive(Experiment):
    """CPU feature env + interactive viewer + physics DR + holdout eval + charts.

    Attributes:
        num_envs: kept small — the viewer + CPU backend is a debug/watch path.
    """

    num_envs = 16
    seed = 0
    total_env_steps = 3_000_000
    eval_every_steps = 500_000
    ablation_group = "examples"
    variant = "watch_live"

    def pipeline(self):
        return (
            FeatureEnvironment(num_envs=self.num_envs,
                               backend="gpu", view="gui")   # GPU + viewer (Part M)
            >> DomainRandomizationPhysics()
            >> VectorPolicy(keys=("state",))
            >> Evaluation(real_tracks=("reinvent_base", "Oval_track"),
                          eval_num_envs=32, charts=True)             # Part N
        )


if __name__ == "__main__":
    WatchLive().run()
