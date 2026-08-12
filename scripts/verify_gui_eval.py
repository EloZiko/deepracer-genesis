"""See the out-of-loop holdout eval in the GUI viewer (Evaluation(gui=True)).

Trains a tiny feature-vector policy for a few iterations, then runs the
out-of-loop per-track holdout eval with ``gui=True`` — a Genesis window opens
and you watch the policy drive each real track (paced to real time). This is the
fast way to check the viewer path; it is renderer-agnostic, so a camera/Nyx env
behaves the same (the window shows the cars while the policy uses its own obs).

Run on a machine with a display (needs an X/Wayland session)::

    python scripts/verify_gui_eval.py
    python scripts/verify_gui_eval.py --track Oval_track --eval_envs 2

Note: a fresh 3-iteration policy is bad on purpose — you are checking that the
*viewer* opens and the rollout is watchable, not that it laps cleanly.
"""

from __future__ import annotations

import argparse

from deepracer_genesis.experiment import (
    Evaluation,
    FeatureEnvironment,
    VectorPolicy,
    run,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="donut_track", help="track to train + holdout-eval on")
    ap.add_argument("--num_envs", type=int, default=64, help="training envs")
    ap.add_argument("--eval_envs", type=int, default=4, help="holdout eval envs (keep small for a legible window)")
    ap.add_argument("--iters", type=int, default=3, help="training iterations before the eval")
    args = ap.parse_args()

    pipe = (
        FeatureEnvironment(num_envs=args.num_envs, tracks=(args.track,))
        >> VectorPolicy(keys=("state",))
        # real_tracks (non-empty) triggers the out-of-loop holdout stage;
        # gui=True opens the viewer for it.
        >> Evaluation(real_tracks=(args.track,), eval_num_envs=args.eval_envs,
                      gui=True, charts=False)
    )
    print(f"training {args.iters} iters on {args.track!r}, then the GUI holdout eval opens...")
    run(pipe,
        total_env_steps=args.num_envs * 32 * args.iters,   # ~args.iters iterations
        eval_every_steps=0,                                # skip periodic eval
        root="runs/_gui_eval_demo")
    print("done — the holdout eval ran in a GUI window.")


if __name__ == "__main__":
    main()
