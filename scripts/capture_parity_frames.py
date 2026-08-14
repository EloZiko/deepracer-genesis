"""Capture sim frames + the torch policy's actions for the on-car parity gate.

Runs IN the genesis process (never alongside onnxruntime). Rolls the trained
policy in a DR-STRIPPED rebuild of its own spec — with all domain
randomization off, the renderer's float obs is exactly uint8/255, so the
captured uint8 frames reproduce the policy's input losslessly and the parity
test can demand tight agreement instead of hand-waving past DR noise.

Output .npz:
    frames  (N, H, W, 3) uint8 RGB — exactly what a perfect camera would feed
    actions (N, 2) float32 — the torch policy's RAW (unclipped) outputs
    plus resolution/spec_id metadata.

Usage:
    uv run python scripts/capture_parity_frames.py examples.camera:CameraMadronaDr \
        --ckpt runs/examples/camera_madrona_dr-0-<id>/model.pt --out parity.npz
"""

from __future__ import annotations

import argparse
import importlib
from dataclasses import replace

import numpy as np


def resolve_target(path: str):
    import os
    import sys
    # scripts run with sys.path[0]=scripts/; targets like examples.camera
    # live under the project root (= cwd, per the usage line above).
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    mod_name, cls_name = path.split(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="experiment as module:ClassName")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="parity.npz")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=8)
    args = parser.parse_args()

    import torch
    from rsl_rl.runners import OnPolicyRunner

    from deepracer_genesis.experiment.builder import Builder
    from deepracer_genesis.experiment.rsl_backend import spec_to_train_cfg
    from deepracer_genesis.experiment.run import build
    from deepracer_genesis.experiment.spec import ActionDRSpec, ObsDRSpec

    spec = build(resolve_target(args.target))
    # Strip ALL randomization: the deployment contract is the clean pipeline,
    # and world-color remap would break the lossless uint8 round-trip.
    spec = replace(spec,
                   obs_dr=ObsDRSpec(),
                   action_dr=ActionDRSpec(),
                   env=replace(spec.env, num_envs=args.num_envs, view="none"))
    spec.validate()

    sim = Builder(spec).sim()
    device = str(sim.device)
    runner = OnPolicyRunner(sim, spec_to_train_cfg(spec), None, device=device)
    runner.load(args.ckpt)
    policy = runner.get_inference_policy(device=device)

    # Sequential capture: frames keep (T, N, ...) layout and episode
    # boundaries are recorded, because with frame_stack > 1 the node's
    # stacker must be primed exactly where the env primed (episode starts) —
    # parity is over SEQUENCES now, not independent samples.
    frames, actions, dones = [], [], []
    with torch.inference_mode():
        obs = sim.get_observations()
        for _ in range(args.steps):
            act = policy(obs)                       # raw mean, UNCLIPPED
            # image_buf is float in [0,1] == uint8/255 exactly (no DR), NCHW;
            # with stacking it is still the SINGLE newest frame — the test
            # rebuilds the stack the way the node would.
            rgb = (sim.image_buf * 255.0).round().to(torch.uint8)
            frames.append(rgb.permute(0, 2, 3, 1).cpu().numpy())
            actions.append(act.detach().cpu().numpy())
            obs, _, done, _ = sim.step(act)
            dones.append(done.detach().cpu().numpy().astype(bool))

    frames_np = np.stack(frames, axis=0)                       # (T, N, H, W, 3)
    actions_np = np.stack(actions, axis=0).astype(np.float32)  # (T, N, 2)
    dones_np = np.stack(dones, axis=0)                         # (T, N)

    w, h = spec.env.resolution
    np.savez_compressed(args.out, frames=frames_np, actions=actions_np,
                        dones=dones_np,
                        frame_stack=np.array(getattr(spec.env, "frame_stack", 1)),
                        resolution=np.array([w, h]),
                        spec_id=np.array(spec.id()),
                        ckpt=np.array(args.ckpt))
    print(f"[capture] {frames_np.shape[0]}x{frames_np.shape[1]} steps x envs "
          f"({frames_np.shape[2]}x{frames_np.shape[3]}), "
          f"frame_stack={getattr(spec.env, 'frame_stack', 1)} -> {args.out}")


if __name__ == "__main__":
    main()
