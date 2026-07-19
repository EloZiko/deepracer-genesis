"""Per-env domain randomization, following the Genesis DR best-practices guide
(user_guide/.../domain_randomization.html): friction ratio, mass shift, COM
shift, controller gains (kp/kv) and motor armature, all batched per env.

Requires `batch_dofs_info=True, batch_links_info=True` in RigidOptions (the
env enables both when `randomize` is on). Applied at episode reset via
`envs_idx` so each episode draws fresh physics. Visual DR lives with the
renderer strategy (`envs/renderers.py`): camera-mount jitter on the Madrona
renderer, world-color remap on the vision renderers. Per-env lighting is not
supported by the current BatchRenderer; global lighting is fixed at build.
"""

import torch


def _u(lo, hi, shape, device):
    return lo + (hi - lo) * torch.rand(shape, device=device)


def randomize_physics(env, env_ids):
    cfg = env.cfg["rand"]
    n = len(env_ids)
    car = env.car
    links_idx = torch.arange(car.n_links, device=env.device)

    # ---- links: friction, mass, center of mass ----
    lo, hi = cfg["friction_range"]
    car.set_friction_ratio(
        _u(lo, hi, (n, car.n_links), env.device), links_idx, envs_idx=env_ids)

    m = cfg.get("mass_shift_kg", 0.0)
    if m > 0:
        car.set_mass_shift(
            _u(-m, m, (n, car.n_links), env.device), links_idx, envs_idx=env_ids)

    c = cfg.get("com_shift_m", 0.0)
    if c > 0:
        car.set_COM_shift(
            _u(-c, c, (n, car.n_links, 3), env.device), links_idx, envs_idx=env_ids)

    # ---- dofs: controller gains + motor armature (per env, batched) ----
    lo, hi = cfg["steer_kp_scale"]
    car.set_dofs_kp(env.cfg["steer_kp"] * _u(lo, hi, (n, 2), env.device),
                    env.steer_dofs, envs_idx=env_ids)
    car.set_dofs_kv(env.cfg["steer_kv"] * _u(lo, hi, (n, 2), env.device),
                    env.steer_dofs, envs_idx=env_ids)

    lo, hi = cfg["wheel_kv_scale"]
    car.set_dofs_kv(env.cfg["wheel_kv"] * _u(lo, hi, (n, 4), env.device),
                    env.wheel_dofs, envs_idx=env_ids)

    lo, hi = cfg.get("armature_range", (0.0, 0.0))
    if hi > 0:
        car.set_dofs_armature(
            _u(lo, hi, (n, 6), env.device),
            env.wheel_dofs + env.steer_dofs, envs_idx=env_ids)
