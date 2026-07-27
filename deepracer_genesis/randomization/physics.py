"""Per-env physics domain randomization: friction, mass, COM, gains, armature.

Both functions are applied ONCE per run (static per-env bodies), from the env's
``__init__``, not per reset: ``randomize_physics`` sets friction / mass / COM /
controller gains, ``randomize_armature`` sets the motor armature (a separate
call because ``set_dofs_armature`` hides a full mass-matrix recompute). Per-reset
physics setters still sporadically fault even on genesis 1.2.3 (see below), so
DR bodies are fixed for the run and only spawn/state randomizes per episode.
The definition lives here (Part L).

Requires genesis>=1.2.3 (quadrants 1.1.1): quadrants 1.0.2's GPU allocator had
memory-safety bugs (chunk-tail overrun, use-after-erase, DLPack offset
truncation) that corrupted state under the dense alloc/free churn of DR at
reset — the cause of the sporadic CUDA_ERROR_ILLEGAL_ADDRESS and NaN'd physics.
"""

import torch


def _u(lo, hi, shape, device):
    """Sample a uniform tensor of `shape` in `[lo, hi)` on `device`."""
    return lo + (hi - lo) * torch.rand(shape, device=device)


def randomize_physics(env, env_ids):
    """Draw fresh per-env friction / mass / COM / controller gains for `env_ids`."""
    cfg = env.cfg["rand"]
    n = len(env_ids)
    car = env.car
    dev = env.device
    links_idx = torch.arange(car.n_links, device=dev)
    car_cfg = env.cfg["car"]

    # ---- links: friction, mass, center of mass ----
    lo, hi = cfg["friction_range"]
    car.set_friction_ratio(_u(lo, hi, (n, car.n_links), dev), links_idx, envs_idx=env_ids)

    m = cfg.get("mass_shift_kg", 0.0)
    if m > 0:
        # genesis adds the shift to each link's rest mass UNCLAMPED, and several
        # links weigh far less than the configured range (0.1 kg steering hinges,
        # <=1e-3 kg camera links vs the 0.2 kg default) — an unscaled draw makes
        # their effective mass negative nearly every reset (negative mass
        # accelerates against force -> runaway velocities -> NaN). Scale each
        # link's span to at most 90% of its rest mass, symmetrically.
        rest = getattr(env, "_link_rest_mass", None)
        if rest is None:
            # batched dofs/links info returns (n_envs, n_links); unbatched
            # (n_links,). Rest masses are identical across envs -> row 0.
            rest = (car.get_links_inertial_mass().to(dev)
                    .reshape(-1, car.n_links)[:1])
            env._link_rest_mass = rest
        span = torch.clamp(0.9 * rest, max=m)
        car.set_mass_shift(_u(-1.0, 1.0, (n, car.n_links), dev) * span,
                           links_idx, envs_idx=env_ids)

    c = cfg.get("com_shift_m", 0.0)
    if c > 0:
        car.set_COM_shift(_u(-c, c, (n, car.n_links, 3), dev), links_idx, envs_idx=env_ids)

    # ---- dofs: controller gains (per env, batched) ----
    lo, hi = cfg["steer_kp_scale"]
    car.set_dofs_kp(car_cfg["steer_kp"] * _u(lo, hi, (n, 2), dev), env.steer_dofs, envs_idx=env_ids)
    car.set_dofs_kv(car_cfg["steer_kv"] * _u(lo, hi, (n, 2), dev), env.steer_dofs, envs_idx=env_ids)
    lo, hi = cfg["wheel_kv_scale"]
    car.set_dofs_kv(car_cfg["wheel_kv"] * _u(lo, hi, (n, 4), dev), env.wheel_dofs, envs_idx=env_ids)


def randomize_armature(env, env_ids):
    """Per-RUN motor-armature DR (applied once at build, not every reset).

    ``set_dofs_armature`` triggers a full mass-matrix recompute per call, so it
    is applied once for the whole run rather than at every episode reset.
    """
    lo, hi = env.cfg["rand"].get("armature_range", (0.0, 0.0))
    if hi > 0:
        n = len(env_ids)
        env.car.set_dofs_armature(
            _u(lo, hi, (n, 6), env.device),
            env.wheel_dofs + env.steer_dofs, envs_idx=env_ids)
