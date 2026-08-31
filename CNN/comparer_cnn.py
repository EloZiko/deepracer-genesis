"""Compare la meme politique nourrie par le simu puis par le CNN.

Meme piste, meme camera, meme checkpoint : seule la source des 7 valeurs change.
C'est la mesure qui dit si la perception apprise coute quelque chose a la conduite.

Chaque scene camera tourne dans son propre sous-processus. Le contexte OpenGL de
pyrender est global au processus : quand le ramasse-miettes detruit une scene
precedente, il emporte le contexte de la scene vivante, et le rendu suivant
echoue sur `glBindFramebuffer: invalid operation`. Une boucle en processus
unique ne tient donc pas au-dela d'une scene.

    python CNN/comparer_cnn.py <chemin/model.pt> [piste ...] [--video]
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MESURES = ("episodes", "mean_return", "mean_progress_m", "mean_episode_s",
           "completion_rate", "mean_laps", "offtrack_rate", "lap_time_s",
           "mean_speed_mps")
CAS = ("parfait", "cnn")
NUM_ENVS = 16
ICI = Path(__file__).resolve().parent


def _feature_set(cas):
    from deepracer_genesis.envs.features import PerceptionFeatures
    from features_reel import CNNPerception
    return PerceptionFeatures if cas == "parfait" else CNNPerception


# ---------------------------------------------------------------- enfant
def une_eval(ckpt, piste, cas):
    """Une piste, une source de perception, une scene. Ecrit RESULTAT <json>."""
    from deepracer_genesis.experiment import run
    from deepracer_genesis.experiment.evaluator import (build_single_track_sim,
                                                        evaluate_on_tracks)
    from deepracer_genesis.experiment.visualize import _rsl_actor
    from train_rl_reel import PiloteReel

    spec = run(PiloteReel, build_only=True, feature_set=_feature_set(cas),
               tracks=(piste,), num_envs=NUM_ENVS)
    sim = build_single_track_sim(spec, piste, NUM_ENVS)
    actor = _rsl_actor(spec, ckpt, sim)
    m = evaluate_on_tracks(actor, (piste,), sim_factory=lambda t, s=sim: s)[piste]
    print("RESULTAT " + json.dumps(m))


def une_video(ckpt, piste, cas):
    from deepracer_genesis.experiment.visualize import rollout_video
    from train_rl_reel import PiloteReel

    rollout_video(PiloteReel, root="../runs", ckpt=ckpt, track=piste, steps=1500,
                  num_envs=1, out=f"../runs/videos/{cas}",
                  feature_set=_feature_set(cas), tracks=(piste,))
    print("RESULTAT {}")


# ---------------------------------------------------------------- parent
def lancer(ckpt, mode, piste, cas):
    p = subprocess.run([sys.executable, __file__, ckpt, mode, piste, cas],
                       cwd=ICI, capture_output=True, text=True)
    ligne = next((l for l in p.stdout.splitlines() if l.startswith("RESULTAT ")), None)
    if ligne is None:
        print("  echec :\n" + "\n".join((p.stdout + p.stderr).splitlines()[-6:]))
        return None
    return json.loads(ligne[9:])


def main():
    args = sys.argv[1:]
    ckpt = args.pop(0)

    if args and args[0] in ("--une", "--video-une"):
        mode, piste, cas = args[0], args[1], args[2]
        (une_eval if mode == "--une" else une_video)(ckpt, piste, cas)
        return

    video = "--video" in args
    pistes = [a for a in args if not a.startswith("--")]
    if not pistes:
        from train_rl_reel import PiloteReel
        pistes = list(PiloteReel.tracks)

    resume = []
    for piste in pistes:
        res = {}
        for cas in CAS:
            print(f"  {piste:24} {cas:8} ...", flush=True)
            m = lancer(ckpt, "--une", piste, cas)
            if m is None:
                break
            res[cas] = m
            if video:
                lancer(ckpt, "--video-une", piste, cas)
        if len(res) != 2:
            print(f"\n=== {piste} === incomplet, piste ignoree\n", flush=True)
            continue

        resume.append((piste, res))
        print(f"\n=== {piste} ===")
        print(f"{'mesure':18} {'parfait':>10} {'cnn':>10} {'ecart':>9}")
        for k in MESURES:
            a, b = res["parfait"][k], res["cnn"][k]
            ecart = f"{100*(b-a)/a:+7.0f} %" if a else "       —"
            print(f"{k:18} {a:10.3f} {b:10.3f} {ecart}")
        if video:
            print(f"videos : runs/videos/parfait|cnn/spectator_{piste}.mp4")
        print(flush=True)

    if len(resume) > 1:
        print(f"\n{'':24} {'hors piste':>21} {'progres (m)':>21}")
        print(f"{'piste':24} {'parfait':>10} {'cnn':>10} {'parfait':>10} {'cnn':>10}")
        for piste, r in resume:
            print(f"{piste:24} {r['parfait']['offtrack_rate']:10.2f} "
                  f"{r['cnn']['offtrack_rate']:10.2f} "
                  f"{r['parfait']['mean_progress_m']:10.1f} "
                  f"{r['cnn']['mean_progress_m']:10.1f}")
        n = len(resume)
        moy = lambda c, k: sum(r[c][k] for _, r in resume) / n
        print(f"{'MOYENNE':24} {moy('parfait','offtrack_rate'):10.2f} "
              f"{moy('cnn','offtrack_rate'):10.2f} "
              f"{moy('parfait','mean_progress_m'):10.1f} "
              f"{moy('cnn','mean_progress_m'):10.1f}")


if __name__ == "__main__":
    main()
