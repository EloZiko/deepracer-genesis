"""Video vue du dessus montrant TOUTES les voitures a la fois.

La camera batchee dessine chaque environnement isolement, donc une image ne
contient qu'une voiture. On les recompose : le fond est la mediane des N images
(la piste vide, puisque les voitures sont a des endroits differents), puis on y
repose chaque voiture la ou son image s'ecarte du fond.

    python CNN/video_flotte.py <chemin/model.pt> <piste> [parfait|cnn]
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import imageio.v2 as imageio
import numpy as np
import torch

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.builder import Builder
from deepracer_genesis.experiment.visualize import _rsl_actor

from dataset import RACINE
from features_reel import CNNPerception
from train_rl_reel import PiloteReel

NUM_ENVS, PAS, RES = 16, 900, (900, 675)
SEUIL = 28          # ecart au fond, en niveaux, au-dela duquel c'est une voiture


def composer(frames):
    """Repose les voitures des N images sur un fond de piste vide."""
    fond = np.median(frames, axis=0).astype(np.uint8)
    sortie = fond.copy()
    ecart = np.abs(frames.astype(np.int16) - fond).sum(-1)     # (N, H, W)
    for i in range(len(frames)):
        m = ecart[i] > SEUIL
        sortie[m] = frames[i][m]
    return sortie


def main():
    ckpt, piste = sys.argv[1], sys.argv[2]
    cas = sys.argv[3] if len(sys.argv) > 3 else "cnn"
    fs = CNNPerception if cas == "cnn" else PerceptionFeatures

    spec = run(PiloteReel, build_only=True, feature_set=fs,
               tracks=(piste,), num_envs=NUM_ENVS)
    sim = Builder(spec).sim(extra_cfg={"vision": {"spectator": True,
                                                  "spectator_res": RES}})
    actor = _rsl_actor(spec, ckpt, sim)
    sim.reset_idx(torch.arange(NUM_ENVS, device=sim.device))
    sim._post_physics(torch.arange(NUM_ENVS, device=sim.device))

    dossier = RACINE / "runs" / "videos" / cas
    dossier.mkdir(parents=True, exist_ok=True)
    sortie = dossier / f"flotte_{piste}.mp4"

    sorties = 0
    with imageio.get_writer(sortie, fps=50) as video, torch.no_grad():
        for n in range(PAS):
            td = actor(sim.get_observations().clone())
            sim.step(td["action"])
            info = sim.step_info
            sorties += int((info["offtrack"] | info["flipped"]).sum())
            brut = np.asarray(sim.renderer.spec_cam.render(rgb=True)[0])
            video.append_data(composer(np.ascontiguousarray(brut)))
            if n % 150 == 0:
                print(f"  {100*n//PAS:3d} %", flush=True)

    print(f"{piste} / {cas} : {sorties} sorties de piste")
    print("ouvre :", sortie)


if __name__ == "__main__":
    main()
