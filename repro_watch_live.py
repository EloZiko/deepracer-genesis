"""Full-trainer reproduction of the DR-reset CUDA illegal-memory-access.

Runs the EXACT crashing configuration — watch_live's GPU + interactive viewer +
physics DR through the whole TorchRL trainer — with two adjustments that
recreate the traceback-era (commit 9ec9dd8) code from the current branch:

1. The post-crash SpecError guard (gui + gpu + physics DR) is bypassed.
2. ``qd.sync`` is neutralized, so ``randomize_physics``/``step`` fence with
   ``torch.cuda.synchronize()`` only — exactly the fencing in the traceback.

total_env_steps/variant differ from watch_live so this writes its own run dir
instead of touching the real one.
"""

import quadrants as qd

import deepracer_genesis.experiment.spec as spec_mod
from examples.watch_live import WatchLive

_orig_validate = spec_mod.ExperimentSpec._validate_environment


def _validate_without_guard(self):
    try:
        _orig_validate(self)
    except spec_mod.SpecError as e:
        if "currently crashes" in str(e):
            print("[repro] spec guard bypassed (gui+gpu+DR allowed)", flush=True)
            return
        raise


spec_mod.ExperimentSpec._validate_environment = _validate_without_guard
qd.sync = lambda *a, **k: None
print("[repro] qd.sync neutralized -> torch-sync-only fencing (9ec9dd8 semantics)",
      flush=True)


class ReproWatchLive(WatchLive):
    """watch_live's crash config under a separate run dir."""

    total_env_steps = 3_000_001   # changes the spec hash -> fresh run dir
    variant = "repro_crash"


if __name__ == "__main__":
    ReproWatchLive().run()
