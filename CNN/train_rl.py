"""Entraine une politique RL sur les seuls canaux que le CNN sait predire.

    python CNN/train_rl.py
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

from features_cnn import CNNFeatures

PISTES = ("reinvent_base", "Oval_track", "Bowtie_track", "Monaco", "Spain_track",
          "New_York_Track", "Austin", "Singapore", "China_track", "Canada_Training")
TEST = ("Vegas_track", "Mexico_track")


class PiloteCNN(Experiment):
    seed = 0
    total_env_steps = 20_000_000     # ~45 min a 7 700 pas/s
    eval_every_steps = 2_000_000
    ablation_group = "cnn"
    variant = "perception"

    noise = 1.0        # 1 = l'erreur mesuree du CNN, 0 = perception parfaite
    num_envs = 2048
    backend = "gpu"    # "gpu" = Metal sur Mac, "cpu" sinon
    max_speed = 2.0    # plafond d'action ; None = 4.0 m/s par defaut

    def pipeline(self):
        return (
            FeatureEnvironment(
                feature_set=CNNFeatures,
                feature_params={"noise": self.noise},
                tracks=PISTES,
                num_envs=self.num_envs,
                backend=self.backend,
                max_speed=self.max_speed,
            )
            >> VectorPolicy(keys=("state",))
            >> Evaluation(real_tracks=TEST, eval_num_envs=16)
        )


if __name__ == "__main__":
    record = run(PiloteCNN, root="runs")
    print("\nresultat final :", {k: round(v, 3) for k, v in record.metrics.items()
                                 if isinstance(v, (int, float))})
    print("checkpoint :", record.metrics.get("checkpoint", record.__dict__.get("checkpoint")))
