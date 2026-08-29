"""Planche contact de toutes les pistes du jeu de donnees, vue de dessus."""

import math
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepracer_genesis.envs.track import TRACKS, ASSETS_DIR
from deepracer_genesis.tools.track_builder import plot_track, track_metrics

from dataset import PISTES, RACINE

VAL = ("reInvent2019_track_v2", "2022_reinvent_champ_v2",
       "Vegas_track_v2", "Mexico_track_v2")
COLONNES = 6


def route(nom):
    return np.load(f"{ASSETS_DIR}/{TRACKS[nom][1]}").astype(np.float32)


def main():
    noms = [p[:-3] for p in PISTES]
    lignes = math.ceil(len(noms) / COLONNES)
    fig, axes = plt.subplots(lignes, COLONNES,
                             figsize=(2.4 * COLONNES, 2.6 * lignes))
    for ax, nom in zip(axes.flat, noms):
        r = route(nom)
        plot_track(r, ax=ax, dash_len=0.6, dash_gap=0.7)
        m = track_metrics(r)
        val = f"{nom}_v2" in VAL
        ax.set_title(f"{nom}\n{m['length_m']:.0f} m  |  {m['width_m']:.2f} m",
                     fontsize=6.5, pad=3, color="crimson" if val else "black")
    for ax in axes.flat[len(noms):]:
        ax.axis("off")

    fig.suptitle(f"{len(noms)} pistes  —  en rouge : validation", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    sortie = RACINE / "data" / "pistes.png"
    fig.savefig(sortie, dpi=110)
    print(sortie)


if __name__ == "__main__":
    main()
