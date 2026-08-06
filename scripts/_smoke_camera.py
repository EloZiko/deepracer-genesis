"""Minimal camera-render GPU smoke (renderer-agnostic).

Builds a small camera env, steps a few times, and asserts every env produces a
non-degenerate frame. Used to check the vision path still works after a
dependency bump. Renderer via --renderer (batch=Madrona default, nyx).

    python scripts/_smoke_camera.py                 # Madrona
    python scripts/_smoke_camera.py --renderer nyx  # Nyx
"""

from __future__ import annotations

import argparse
import math

import torch

from deepracer_genesis._gs import ensure_init
from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs import DeepRacerEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default="batch", choices=["batch", "nyx"])
    ap.add_argument("--num_envs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    n = args.num_envs
    env_cfg = get_env_cfg(vision=True, track="reinvent_base")
    env_cfg["vision"]["vision_renderer"] = args.renderer
    ensure_init(env_cfg["sim"]["backend"])
    env = DeepRacerEnv(num_envs=n, env_cfg=env_cfg)

    max_std = torch.zeros(n, device=env.device)
    for t in range(args.steps):
        steer = 0.6 * math.sin(2 * math.pi * t / 30)
        act = torch.tensor([[steer, -0.2]], device=env.device).repeat(n, 1)
        env.step(act)
        max_std = torch.maximum(max_std, env.image_buf.flatten(1).std(dim=1))

    img = env.image_buf
    finite = bool(torch.isfinite(img).all())
    nondegen = bool((max_std > 0.02).all())
    print(f"renderer={args.renderer}  shape={tuple(img.shape)}  finite={finite}")
    print(f"per-env pixel std: {[round(s, 3) for s in max_std.tolist()]}")
    ok = finite and nondegen
    print(f"[{'PASS' if ok else 'FAIL'}] camera renders non-degenerate frames on all envs")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
