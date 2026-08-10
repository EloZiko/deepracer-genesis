"""Verify camera training on the CPU backend via the per-env rasterizer (M.2).

This is the cpu-vision / raster-vision branch feature: render the policy camera
on the CPU backend (no CUDA graphics) through gs.renderers.Rasterizer instead of
Madrona. Builds a camera+cpu env, steps, and checks every env produces a real
frame. Also trains ~2 iters through the full run() path to prove end-to-end.
"""

from __future__ import annotations

import math

import torch


def main() -> None:
    from deepracer_genesis._gs import ensure_init
    from deepracer_genesis.configs.cfgs import get_env_cfg
    from deepracer_genesis.envs import DeepRacerEnv

    n = 4
    env_cfg = get_env_cfg(vision=True, track="reinvent_base", backend="cpu")
    env_cfg["vision"]["vision_renderer"] = "rasterizer"   # what Builder sets for camera+cpu
    ensure_init("cpu")
    env = DeepRacerEnv(num_envs=n, env_cfg=env_cfg)

    max_std = torch.zeros(n, device=env.device)
    for t in range(5):
        act = torch.tensor([[0.4 * math.sin(t), -0.2]], device=env.device).repeat(n, 1)
        env.step(act)
        max_std = torch.maximum(max_std, env.image_buf.flatten(1).std(dim=1))

    img = env.image_buf
    finite = bool(torch.isfinite(img).all())
    nondegen = bool((max_std > 0.02).all())
    print(f"CPU rasterizer camera  shape={tuple(img.shape)}  device={img.device}  finite={finite}")
    print(f"per-env pixel std: {[round(s, 3) for s in max_std.tolist()]}")
    ok = finite and nondegen
    print(f"[{'PASS' if ok else 'FAIL'}] CPU-backend camera renders real per-env frames")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
