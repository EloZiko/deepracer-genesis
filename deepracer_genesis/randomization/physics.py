"""Per-env physics domain randomization: friction, mass, COM, gains, armature.

Applied at episode reset via `envs_idx`; requires batched dofs/links info.
The definition lives here (Part L); the env's ``reset_idx`` is the application
site that calls :func:`randomize_physics`.
"""

import torch


def _u(lo, hi, shape, device):
    """Sample a uniform tensor of `shape` in `[lo, hi)` on `device`."""
    return lo + (hi - lo) * torch.rand(shape, device=device)


def randomize_physics(env, env_ids):
    """Draw fresh per-env physics for `env_ids` at reset.

    Each genesis write consumes its value tensor — AND genesis-internal index
    masks it allocates on torch's stream — ASYNCHRONOUSLY on genesis's own CUDA
    stream. torch's caching allocator is unaware of genesis's stream, so without
    a fence the next write's allocation recycles a buffer a genesis kernel is
    still reading -> sporadic ``CUDA_ERROR_ILLEGAL_ADDRESS`` (surfaces under the
    TorchRL collector's allocations or the interactive viewer; hidden under
    ``CUDA_LAUNCH_BLOCKING=1``). We therefore ``synchronize`` right after each
    write, before any torch allocation can recycle memory still in flight.
    """
    cfg = env.cfg["rand"]
    n = len(env_ids)
    car = env.car
    dev = env.device
    cuda = dev.type == "cuda"
    links_idx = torch.arange(car.n_links, device=dev)
    car_cfg = env.cfg["car"]

    def write(fn, tensor, dofs_idx):
        """Fence the torch fill, THEN apply one genesis DR write.

        The draw is filled by a ``torch.rand`` kernel on torch's stream; genesis
        reads ``tensor`` on its OWN stream without waiting for that fill, so a
        device sync must land between the fill and the genesis read (it also
        waits out the previous write's kernel + its internal index masks).
        """
        if cuda:
            torch.cuda.synchronize()
        fn(tensor, dofs_idx, envs_idx=env_ids)

    # ---- links: friction, mass, center of mass ----
    lo, hi = cfg["friction_range"]
    write(car.set_friction_ratio, _u(lo, hi, (n, car.n_links), dev), links_idx)
    m = cfg.get("mass_shift_kg", 0.0)
    if m > 0:
        write(car.set_mass_shift, _u(-m, m, (n, car.n_links), dev), links_idx)
    c = cfg.get("com_shift_m", 0.0)
    if c > 0:
        write(car.set_COM_shift, _u(-c, c, (n, car.n_links, 3), dev), links_idx)

    # ---- dofs: controller gains + motor armature (per env, batched) ----
    lo, hi = cfg["steer_kp_scale"]
    write(car.set_dofs_kp, car_cfg["steer_kp"] * _u(lo, hi, (n, 2), dev), env.steer_dofs)
    write(car.set_dofs_kv, car_cfg["steer_kv"] * _u(lo, hi, (n, 2), dev), env.steer_dofs)
    lo, hi = cfg["wheel_kv_scale"]
    write(car.set_dofs_kv, car_cfg["wheel_kv"] * _u(lo, hi, (n, 4), dev), env.wheel_dofs)
    lo, hi = cfg.get("armature_range", (0.0, 0.0))
    if hi > 0:
        write(car.set_dofs_armature, _u(lo, hi, (n, 6), dev),
              env.wheel_dofs + env.steer_dofs)
