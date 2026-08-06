"""Run ONE example experiment end-to-end with a ~2-iteration budget.

Genesis builds one scene per process, so run this once per class (separate
processes). Used to verify the examples actually TRAIN (not just build a spec):

    python scripts/_run_example.py FeatureCpu
"""

from __future__ import annotations

import sys

import examples
from deepracer_genesis.experiment import run

name = sys.argv[1]
cls = getattr(examples, name)

# build once to read num_envs, then shrink to ~2 learning iterations
spec = run(cls, build_only=True)
n = spec.env.num_envs
cls.total_env_steps = 2 * n * 32       # ~2 iters regardless of num_envs
cls.eval_every_steps = 0               # skip periodic eval; final eval still runs

rec = run(cls, root="runs/_ex")
print(f"[EXAMPLE PASS] {name}: modality={spec.env.modality} render={spec.env.render} "
      f"backend={spec.env.backend} completion={rec.metrics.get('completion_rate')}")
