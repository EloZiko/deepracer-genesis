"""Videos vues du dessus des politiques entrainees.

    python CNN/voir_pilote.py                  # les 2 bruits sur les pistes de test
    python CNN/voir_pilote.py Monaco           # sur une piste precise
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment.visualize import rollout_video

from train_rl import TEST, PiloteCNN

BRUITS = (0.0, 1.0)


def main():
    pistes = (sys.argv[1],) if len(sys.argv) > 1 else TEST
    for bruit in BRUITS:
        for piste in pistes:
            try:
                sortie = rollout_video(PiloteCNN, root="../runs", track=piste,
                                       steps=1500, num_envs=1,
                                       out=f"../runs/videos/noise{bruit}",
                                       noise=bruit)
                print(f"noise={bruit} {piste:16} -> {sortie}", flush=True)
            except FileNotFoundError as e:
                print(f"noise={bruit} {piste:16} -> pas de checkpoint ({e})")


if __name__ == "__main__":
    main()
