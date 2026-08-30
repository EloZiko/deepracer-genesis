"""Deux entrainements : perception parfaite, puis celle du CNN.

    caffeinate -i ../.venv/bin/python comparer_bruit.py
"""

import warnings

warnings.filterwarnings("ignore")

from deepracer_genesis.experiment import run

from train_rl import PiloteCNN


def main():
    for bruit in (0.0, 1.0):
        record = run(PiloteCNN, root="../runs", noise=bruit)
        m = record.metrics
        print(f"\nnoise={bruit}  completion {m['completion_rate']:.2f}"
              f"  progres {m['mean_progress_m']:.1f} m"
              f"  vitesse {m['mean_speed_mps']:.2f} m/s"
              f"  hors piste {m['offtrack_rate']:.2f}", flush=True)


if __name__ == "__main__":
    main()
