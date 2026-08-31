"""Politique entrainee a travers le vrai CNN de perception.

    caffeinate -i ../.venv/bin/python train_rl_reel.py
"""

import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import (
    CameraEnvironment,
    PPO,
    Evaluation,
    Experiment,
    VectorPolicy,
    run,
)

from deepracer_genesis.envs.features import PerceptionFeatures

from features_reel import CNNPerception


class PiloteReel(Experiment):
    seed = 0
    # on repart de la politique entrainee sur les valeurs exactes du simu :
    # elle sait deja conduire, il ne lui reste qu'a encaisser les erreurs du CNN
    resume = "politique_reference.pt"
    total_env_steps = 3_000_000     # ~2 h 40 a 320 pas/s sur 6 pistes
    eval_every_steps = 300_000
    ablation_group = "cnn"
    variant = "reel"

    # 6 pistes couvrant toute la gamme de severite des virages, du plus doux
    # au plus serre : c'est elle qui decide des sorties (correlation +0.76)
    tracks = ("2022_march_open",       # k_std 0.10  tres facile
              "arctic_open",           #       0.18  facile
              "jyllandsringen_open",   #       0.22  moyen
              "hamption_pro",          #       0.25  difficile
              "thunder_hill_pro",      #       0.27  tres difficile
              "Tokyo_Training_track")  #       0.42  extreme
    # 883 pas/s mesures a 16 envs (le 320 d'avant venait d'une machine chargee).
    # La boucle tient sur un seul coeur des dix et 1.4 Go des 16 : monter les
    # envs ne coute donc pas de temps de mur, et le lot PPO passe de 384 a 1536
    # echantillons par correction -- ce qui etait la cause suspectee n2.
    num_envs = 64
    max_speed = 2.0
    lr = 3e-5          # on affine une politique qui sait deja conduire
    # le schedule adaptatif retaille le lr d'apres la KL mesuree ; sur 96
    # echantillons par minibatch cette KL est du bruit et le lr derive vers
    # son plafond de 1e-2, ce qui detruit la politique reprise. On le fige.
    schedule = "fixed"
    feature_set = CNNPerception   # PerceptionFeatures = perception parfaite
    feature_params = None         # p.ex. {"cnn_device": "cpu"}

    def pipeline(self):
        return (
            CameraEnvironment(
                feature_set=self.feature_set,
                feature_params=self.feature_params,
                resolution=(160, 120),
                frame_stack=4,
                tracks=self.tracks,
                num_envs=self.num_envs,
                backend="cpu",          # pas de Madrona sur Mac
                max_speed=self.max_speed,
            )
            >> VectorPolicy(keys=("state",))
            >> PPO(lr=self.lr, schedule=self.schedule)
            >> Evaluation(real_tracks=self.tracks, eval_num_envs=8)
        )


if __name__ == "__main__":
    record = run(PiloteReel, root="../runs")
    m = record.metrics
    print(f"\ncompletion {m['completion_rate']:.3f}  progres {m['mean_progress_m']:.1f} m"
          f"  hors piste {m['offtrack_rate']:.2f}")
