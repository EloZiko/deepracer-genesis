"""Evalue la politique nourrie par le CNN sur beaucoup de pistes.

Genesis ne reconstruit pas une scene camera dans un meme processus, donc chaque
piste tourne dans son propre sous-processus.

    python CNN/balayage.py <chemin/model.pt> [nombre de pistes]
    python CNN/balayage.py <chemin/model.pt> --une <piste>     # usage interne
"""

import json
import math
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from dataset import PISTES, RACINE

NUM_ENVS = 16


def evaluer(ckpt, piste):
    from deepracer_genesis.experiment import run
    from deepracer_genesis.experiment.evaluator import (build_single_track_sim,
                                                        evaluate_on_tracks)
    from deepracer_genesis.experiment.visualize import _rsl_actor
    from features_reel import CNNPerception
    from train_rl_reel import PiloteReel

    spec = run(PiloteReel, build_only=True, feature_set=CNNPerception,
               tracks=(piste,), num_envs=NUM_ENVS)
    sim = build_single_track_sim(spec, piste, NUM_ENVS)
    actor = _rsl_actor(spec, ckpt, sim)
    return evaluate_on_tracks(actor, (piste,), sim_factory=lambda t, s=sim: s)[piste]


def geometrie(piste):
    """Longueur, largeur, ecartement des waypoints et amplitude de courbure."""
    from deepracer_genesis.envs.track import ASSETS_DIR, TRACKS
    w = np.load(f"{ASSETS_DIR}/{TRACKS[piste][1]}").astype(np.float64)
    c = w[:, :2]
    if np.allclose(c[0], c[-1], atol=1e-6):
        w, c = w[:-1], c[:-1]
    seg = np.linalg.norm(np.roll(c, -1, 0) - c, axis=1)
    yaw = np.arctan2(*(np.roll(c, -1, 0) - c).T[::-1])
    dyaw = (np.roll(yaw, -1) - yaw + math.pi) % (2 * math.pi) - math.pi
    k = dyaw / np.maximum(seg, 1e-6) / 2.5
    return {"longueur": seg.sum(),
            "largeur": float(np.linalg.norm(w[:, 4:6] - w[:, 2:4], axis=1).mean()),
            "pas_wp": float(seg[seg > 1e-3].mean()),
            "k_std": float(k[np.abs(k) < 1].std())}


def main():
    ckpt = sys.argv[1]
    if "--une" in sys.argv:
        piste = sys.argv[sys.argv.index("--une") + 1]
        print("RESULTAT " + json.dumps(evaluer(ckpt, piste)))
        return

    from train_rl import PISTES as VUES
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    toutes = [p[:-3] for p in PISTES]
    pistes = [toutes[i] for i in np.linspace(0, len(toutes) - 1, n).astype(int)]

    lignes = []
    for i, piste in enumerate(pistes, 1):
        r = subprocess.run([sys.executable, __file__, ckpt, "--une", piste],
                           capture_output=True, text=True, cwd=str(RACINE / "CNN"))
        ligne = next((l for l in r.stdout.splitlines() if l.startswith("RESULTAT ")), None)
        if ligne is None:
            print(f"  {i:2}/{n}  {piste:28} ECHEC", flush=True)
            continue
        m = json.loads(ligne[9:])
        m.update(geometrie(piste), piste=piste, vue=piste in VUES)
        lignes.append(m)
        print(f"  {i:2}/{n}  {piste:28} hors piste {m['offtrack_rate']:.2f}", flush=True)

    lignes.sort(key=lambda m: m["offtrack_rate"])
    print(f"\n{'piste':28} {'vue':>4} {'long':>6} {'larg':>5} {'pas wp':>7} "
          f"{'k std':>6} {'hors piste':>11} {'progres':>8} {'tours':>6}")
    for m in lignes:
        print(f"{m['piste']:28} {'oui' if m['vue'] else '-':>4} {m['longueur']:5.0f}m "
              f"{m['largeur']:5.2f} {m['pas_wp']:7.3f} {m['k_std']:6.3f} "
              f"{m['offtrack_rate']:11.2f} {m['mean_progress_m']:7.1f}m {m['mean_laps']:6.2f}")

    (RACINE / "data" / "balayage.json").write_text(json.dumps(lignes, indent=1))
    d = [m for m in lignes if not m["vue"]]
    print(f"\n{len(lignes)} pistes, dont {len(d)} jamais vues par la politique")
    print(f"hors piste moyen : {np.mean([m['offtrack_rate'] for m in lignes]):.2f}  "
          f"(pistes inconnues seules : {np.mean([m['offtrack_rate'] for m in d]):.2f})")


if __name__ == "__main__":
    main()
