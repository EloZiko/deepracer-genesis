"""Vue de dessus d'une piste + trajectoires : python CNN/voir_piste.py Monaco_dr"""
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

RACINE = Path(__file__).resolve().parent.parent
ASSETS = RACINE / "deepracer_genesis" / "assets"
dossier = sys.argv[1]
nom = dossier.replace("_dr", "").replace("_v2", "")


def route(nom):
    direct = ASSETS / "routes" / f"{nom}.npy"
    if direct.exists():
        return np.load(direct)[:, :2]
    return np.load(ASSETS / "tracks" / "generated" / nom / "route.npy")[:, :2]


centre = route(nom)
table = pq.read_table(str(sorted(glob.glob(str(RACINE / "data" / dossier / "*.parquet")))[0]),
                      columns=["env", "pose", "done"])
pose = np.array(table["pose"].to_pylist(), dtype=np.float32)
env = np.array(table["env"].to_pylist())
fin = np.array(table["done"].to_pylist())

plt.figure(figsize=(9, 9))
plt.plot(centre[:, 0], centre[:, 1], "k-", lw=3, alpha=.35, label="centre de la piste")
for v in sorted(set(env))[:8]:
    m = env == v
    plt.plot(pose[m, 0], pose[m, 1], lw=1, alpha=.8)
plt.scatter(pose[fin, 0], pose[fin, 1], c="red", s=60, zorder=5,
            label=f"sorties de piste ({fin.sum()})")
plt.axis("equal")
plt.legend()
plt.title(f"{dossier} — {len(set(env))} voitures, {table.num_rows} images")
sortie = RACINE / "data" / f"piste_{dossier}.png"
plt.savefig(sortie, dpi=90, facecolor="white", bbox_inches="tight")
print("ouvre :", sortie)
