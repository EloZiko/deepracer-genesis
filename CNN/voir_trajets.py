"""Trajectoires des 16 voitures, perception parfaite contre CNN, cote a cote.

    python CNN/voir_trajets.py <chemin/model.pt> <piste>
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.envs.track import ASSETS_DIR, TRACKS
from deepracer_genesis.experiment import run
from deepracer_genesis.experiment.evaluator import build_single_track_sim
from deepracer_genesis.experiment.visualize import _rsl_actor

from dataset import RACINE
from features_reel import CNNPerception
from train_rl_reel import PiloteReel

CAS = (("perception parfaite", PerceptionFeatures), ("via le CNN", CNNPerception))
NUM_ENVS, PAS = 16, 900


def rouler(spec, piste, ckpt):
    """Renvoie (trajectoires, sorties) : positions par pas et points de sortie."""
    sim = build_single_track_sim(spec, piste, NUM_ENVS)
    actor = _rsl_actor(spec, ckpt, sim)
    sim.reset_idx(torch.arange(NUM_ENVS, device=sim.device))
    xy, sorties = [], []
    for _ in range(PAS):
        td = actor(sim.get_observations().clone())
        sim.step(td["action"])
        pos = sim.base_pos[:, :2].cpu().numpy()
        xy.append(pos.copy())
        info = sim.step_info
        dehors = (info["offtrack"] | info["flipped"]).cpu().numpy()
        if dehors.any():
            sorties.extend(pos[dehors])
    return np.stack(xy), np.array(sorties).reshape(-1, 2)


def main():
    ckpt, piste = sys.argv[1], sys.argv[2]
    w = np.load(f"{ASSETS_DIR}/{TRACKS[piste][1]}").astype(np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, (nom, fs) in zip(axes, CAS):
        spec = run(PiloteReel, build_only=True, feature_set=fs, tracks=(piste,))
        xy, sorties = rouler(spec, piste, ckpt)
        for bord in (w[:, 2:4], w[:, 4:6]):
            ax.plot(*np.vstack([bord, bord[:1]]).T, color="0.35", lw=1.2)
        for i in range(NUM_ENVS):
            t = xy[:, i].copy()
            # une reapparition apres sortie fait un saut : on coupe le trait
            saut = np.linalg.norm(np.diff(t, axis=0), axis=1) > 0.5
            t[1:][saut] = np.nan
            ax.plot(t[:, 0], t[:, 1], lw=0.9, alpha=0.75)
        if len(sorties):
            ax.plot(*sorties.T, "x", color="crimson", ms=6, mew=1.6,
                    label=f"{len(sorties)} sorties")
            ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{piste} — {nom}", fontsize=11)
        ax.set_aspect("equal"); ax.axis("off")

    fig.tight_layout()
    sortie = RACINE / "data" / f"trajets_{piste}.png"
    fig.savefig(sortie, dpi=120)
    print("ouvre :", sortie)


if __name__ == "__main__":
    main()
