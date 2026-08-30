"""Politique entrainee a travers le vrai CNN de perception.

    caffeinate -i ../.venv/bin/python train_rl_reel.py
"""

import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import (
    CameraEnvironment,
    Evaluation,
    Experiment,
    VectorPolicy,
    run,
)

from features_reel import CNNPerception


class PiloteReel(Experiment):
    seed = 0
    total_env_steps = 9_000_000     # ~9 h a 274 pas/s
    eval_every_steps = 1_000_000
    ablation_group = "cnn"
    variant = "reel"

    # une piste longue : moins de voitures dans le champ des autres
    tracks = ("Monaco",)
    num_envs = 16          # le rasterizer CPU rend une camera par env, en boucle ;
                           # au-dela de 16 le debit s'effondre (32 -> 155 pas/s)
    max_speed = 2.0

    def pipeline(self):
        return (
            CameraEnvironment(
                feature_set=CNNPerception,
                resolution=(160, 120),
                frame_stack=4,
                tracks=self.tracks,
                num_envs=self.num_envs,
                backend="cpu",          # pas de Madrona sur Mac
                max_speed=self.max_speed,
            )
            >> VectorPolicy(keys=("state",))
            >> Evaluation(real_tracks=self.tracks, eval_num_envs=8)
        )


if __name__ == "__main__":
    record = run(PiloteReel, root="../runs")
    m = record.metrics
    print(f"\ncompletion {m['completion_rate']:.3f}  progres {m['mean_progress_m']:.1f} m"
          f"  hors piste {m['offtrack_rate']:.2f}")
