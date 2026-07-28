"""Multi-seed long-run regression check for the reward-exploit collapse.

Before the `rules.lap_progress` clamp, a car hitting the reinvent_base track
pinch teleported its localized arclength (localize nearest-waypoint jump) into
~2.7 m of spurious "progress" in one step — a huge advantage outlier that
collapsed the shared PPO policy (rew_progress spikes, then goes permanently
NEGATIVE; "all cars spinning"). num_envs is kept small (16) on purpose: one
exploiting env dominates the batch, so the collapse is far more observable than
at 1024 envs where it averages out.

This launches ONE seed (fresh process, own CUDA context, own run dir). Different
seeds -> different spawn/DR draws -> reproducible if a collapse ever recurs.

    python scripts/seed_sweep_progress.py --seed 0 --steps 3000000

The trainer prints `rew_progress` every 10 iters; grep the captured log for
`rew_progress -` to flag any negative (i.e. a collapse).
"""

import argparse

from deepracer_genesis.experiment import (
    DomainRandomizationPhysics,
    Experiment,
    FeatureEnvironment,
    VectorPolicy,
    run,
)


class ProgressRegression(Experiment):
    """Feature env + build-time physics DR on reinvent_base (the pinch track).

    The config that historically collapsed; run long + multi-seed to confirm the
    lap_progress clamp holds. Build-time DR is the stable timing (per-episode DR
    still sporadically faults on genesis 1.2.3 — see DR_FIX_ADOPTION_REPORT.md).
    """

    num_envs = 16
    total_env_steps = 3_000_000
    eval_every_steps = 0          # no mid-run eval: keep the loop tight + fast
    ablation_group = "progress_regression"
    variant = "progress_regression"

    dr = True   # set False to drop physics DR (disables batch_dofs/links_info)

    def pipeline(self):
        env = FeatureEnvironment(num_envs=self.num_envs)   # default track reinvent_base (the pinch track)
        chain = (env >> DomainRandomizationPhysics()) if self.dr else env
        return chain >> VectorPolicy(keys=("state",))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--root", default="runs/progress_regression")
    ap.add_argument("--no-dr", action="store_true",
                    help="drop physics DR (isolates the batched-DR alloc path)")
    args = ap.parse_args()
    run(ProgressRegression, seed=args.seed, dr=not args.no_dr,
        total_env_steps=args.steps, root=args.root)
