"""Train a policy on the channels the CNN can predict, with its error simulated.

    python -m perception.train_policy_with_noise

No renderer here: the CNN's measured per-channel error is added to the exact
values instead. Cheap enough to sweep, which is what the ablation uses.
"""

import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import (
    Evaluation,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
    run,
)

from perception.dataset import HOLDOUT_TRACKS, TRAINING_TRACKS, track_names
from perception.noisy_features import NoisyPerceptionFeatures

# every track the dataset covers, minus the shared holdout. Ten training tracks
# left the policy unable to drive an unseen one: off-track rate tracked corner
# severity rank for rank (+1.00 over six held-out tracks), while the perception
# source barely moved it. Breadth is the lever, not a better CNN.
TRAIN_TRACKS = track_names(TRAINING_TRACKS)
TEST_TRACKS = track_names(HOLDOUT_TRACKS)


class NoisyPerceptionPolicy(Experiment):
    seed = 0
    total_env_steps = 50_000_000     # 5x the tracks, so more samples;
                                     # ~2 h at the 7 700 steps/s measured
    eval_every_steps = 5_000_000
    ablation_group = "cnn"
    variant = "noisy_perception"

    noise = 1.0            # 1 = the CNN's measured error, 0 = perfect perception
    noise_channels = None  # None = all; else ("lateral", "heading"), etc.
    num_envs = 2048
    backend = "gpu"    # "gpu" = Metal on Mac, "cpu" otherwise
    max_speed = 2.0    # action cap; None = the 4.0 m/s default
    tracks = TRAIN_TRACKS
    test_tracks = TEST_TRACKS

    def pipeline(self):
        return (
            FeatureEnvironment(
                feature_set=NoisyPerceptionFeatures,
                feature_params={"noise": self.noise,
                                "noise_channels": self.noise_channels},
                tracks=self.tracks,
                num_envs=self.num_envs,
                backend=self.backend,
                max_speed=self.max_speed,
            )
            >> VectorPolicy(keys=("state",))
            >> Evaluation(real_tracks=self.test_tracks, eval_num_envs=16)
        )


if __name__ == "__main__":
    record = run(NoisyPerceptionPolicy, root="runs")
    print("\nfinal result:", {k: round(v, 3) for k, v in record.metrics.items()
                              if isinstance(v, (int, float))})
    print("checkpoint:", record.metrics.get("checkpoint",
                                            record.__dict__.get("checkpoint")))
