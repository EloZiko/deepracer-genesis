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

import os

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


# Restore the ORIGINAL unclamped mass draw (the branch now scales the shift per
# link to avoid negative masses; the crashing code did not — keep the repro
# faithful, and keep the clamp available as a separate A/B lever).
def _randomize_physics_unclamped(env, env_ids):
    import torch as _t

    from deepracer_genesis.randomization.physics import _u

    cfg = env.cfg["rand"]
    n = len(env_ids)
    car = env.car
    dev = env.device
    cuda = dev.type == "cuda"
    links_idx = _t.arange(car.n_links, device=dev)
    car_cfg = env.cfg["car"]

    def write(fn, tensor, dofs_idx):
        if cuda:
            _t.cuda.synchronize()   # 9ec9dd8: torch-sync-only, no qd.sync
        fn(tensor, dofs_idx, envs_idx=env_ids)

    lo, hi = cfg["friction_range"]
    write(car.set_friction_ratio, _u(lo, hi, (n, car.n_links), dev), links_idx)
    m = cfg.get("mass_shift_kg", 0.0)
    if m > 0:
        write(car.set_mass_shift, _u(-m, m, (n, car.n_links), dev), links_idx)
    c = cfg.get("com_shift_m", 0.0)
    if c > 0:
        write(car.set_COM_shift, _u(-c, c, (n, car.n_links, 3), dev), links_idx)
    lo, hi = cfg["steer_kp_scale"]
    write(car.set_dofs_kp, car_cfg["steer_kp"] * _u(lo, hi, (n, 2), dev), env.steer_dofs)
    write(car.set_dofs_kv, car_cfg["steer_kv"] * _u(lo, hi, (n, 2), dev), env.steer_dofs)
    lo, hi = cfg["wheel_kv_scale"]
    write(car.set_dofs_kv, car_cfg["wheel_kv"] * _u(lo, hi, (n, 4), dev), env.wheel_dofs)
    lo, hi = cfg.get("armature_range", (0.0, 0.0))
    if hi > 0:
        write(car.set_dofs_armature, _u(lo, hi, (n, 6), dev),
              env.wheel_dofs + env.steer_dofs)


if os.environ.get("REPRO_UNCLAMPED") == "1":
    import deepracer_genesis.envs.base_env as _base_env_mod
    import deepracer_genesis.randomization.physics as _phys_mod

    _base_env_mod.randomize_physics = _randomize_physics_unclamped
    _phys_mod.randomize_physics = _randomize_physics_unclamped
    print("[repro] unclamped mass draw restored (faithful to 9ec9dd8)", flush=True)
else:
    print("[repro] branch physics active: per-link mass clamp + NaN guards "
          "(REPRO_UNCLAMPED=1 restores the collapsing draw)", flush=True)


class ReproWatchLive(WatchLive):
    """watch_live's crash config under a separate run dir."""

    total_env_steps = 3_000_001   # changes the spec hash -> fresh run dir
    variant = "repro_crash"


if __name__ == "__main__":
    ReproWatchLive().run()
