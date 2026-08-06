"""GPU verification of per-env Nyx environment-map DR (Part P.1).

With spawn randomization off and no other DR, every env holds the SAME pose, so
the onboard frames can differ only by their per-env sky (tint + exposure). A
nonzero cross-env frame difference therefore proves ``env_map`` DR is applied per
env; a control run with the knob off should give ~identical frames.

Run on a GPU + gs_nyx machine::

    python scripts/verify_nyx_env_map.py            # env_map DR ON  -> frames differ
    python scripts/verify_nyx_env_map.py --off      # control        -> frames ~identical
"""

from __future__ import annotations

import argparse

import torch

from deepracer_genesis._gs import ensure_init
from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs import DeepRacerEnv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", action="store_true", help="control: env_map DR disabled")
    ap.add_argument("--num_envs", type=int, default=4)
    args = ap.parse_args()

    n = args.num_envs
    env_cfg = get_env_cfg(vision=True, track="reinvent_base")
    env_cfg["vision"]["vision_renderer"] = "nyx"
    if not args.off:
        # wide ranges so the per-env lighting difference is unmistakable
        env_cfg["vision"]["env_map"] = {"tint": (0.2, 0.9), "multiplier": (0.5, 2.0)}
    env_cfg["spawn"]["random_start"] = False       # identical pose across envs
    env_cfg["spawn"]["random_direction"] = False
    ensure_init(env_cfg["sim"]["backend"])
    env = DeepRacerEnv(num_envs=n, env_cfg=env_cfg)

    act = torch.tensor([[0.0, -1.0]], device=env.device).repeat(n, 1)
    for _ in range(6):                             # settle physics + Nyx temporal
        env.step(act)

    img = env.image_buf                            # (N, 3, H, W) in [0, 1]
    means = [[round(c) for c in (img[i].mean(dim=(1, 2)) * 255).tolist()] for i in range(n)]
    pair = [(img[i] - img[j]).abs().mean().item()
            for i in range(n) for j in range(i + 1, n)]
    print(f"env_map DR: {'OFF (control)' if args.off else 'ON'}")
    print(f"per-env mean RGB: {means}")
    print(f"cross-env pair diffs: {[round(p, 4) for p in pair]}")

    threshold = 0.01
    differ = max(pair) > threshold
    ok = (not differ) if args.off else differ
    verdict = "per-env skies differ (env_map DR active)" if not args.off else \
              "frames ~identical (no env_map DR, as expected)"
    print(f"[{'PASS' if ok else 'FAIL'}] {verdict}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
