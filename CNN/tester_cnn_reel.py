"""Met une politique deja entrainee sur la piste, en lui donnant les valeurs
du vrai CNN au lieu de celles du simu.

Aucun entrainement : on mesure ce que coute la perception a une politique qui
sait deja conduire.

    python CNN/tester_cnn_reel.py <chemin/model.pt>
"""

import sys
import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.evaluator import (
    build_single_track_sim,
    evaluate_on_tracks,
)
from deepracer_genesis.experiment.visualize import _rsl_actor

from train_rl_reel import PiloteReel

NUM_ENVS = 8


def main():
    ckpt = sys.argv[1]
    spec = run(PiloteReel, build_only=True)
    for piste in PiloteReel.tracks:
        sim = build_single_track_sim(spec, piste, NUM_ENVS)
        actor = _rsl_actor(spec, ckpt, sim)
        m = evaluate_on_tracks(actor, (piste,),
                               sim_factory=lambda t, s=sim: s)[piste]
        print(f"\n{piste}  ({m['episodes']} episodes)")
        for k in ("completion_rate", "mean_progress_m", "mean_speed_mps",
                  "offtrack_rate", "mean_laps"):
            print(f"  {k:18} {m[k]:.3f}")


if __name__ == "__main__":
    main()
