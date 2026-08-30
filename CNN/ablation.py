"""Serie d'entrainements pour situer d'ou vient la perte de performance.

    caffeinate -i ../.venv/bin/python ablation.py
"""

import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import run

from train_rl import PiloteCNN

ESSAIS = (
    ("reference",        dict(noise=0.0)),
    ("cnn complet",      dict(noise=1.0)),
    ("moitie de l'erreur", dict(noise=0.5)),
    ("ou suis-je",       dict(noise=1.0, noise_channels=("lateral", "heading"))),
    ("quoi devant",      dict(noise=1.0, noise_channels=("curv@1m", "curv@3m"))),
    ("oval seul",        dict(noise=1.0, tracks=("Oval_track",),
                              test_tracks=("Oval_track",))),
)


def main():
    for nom, reglages in ESSAIS:
        record = run(PiloteCNN, root="../runs", **reglages)
        m = record.metrics
        print(f"\n>>> {nom:20} completion {m['completion_rate']:.3f}"
              f"  progres {m['mean_progress_m']:.1f} m"
              f"  hors piste {m['offtrack_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
