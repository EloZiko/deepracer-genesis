"""Part O render verification: prove camera spatial tiling has no tile-bleed.

Camera multi-track loads ALL K track meshes into every env, each translated to
its own world tile (``MultiTrack.variant_offset``), spaced beyond camera reach so
each env's camera sees only its home track. The definitive no-bleed check: a
tiled env's camera frame must be bit-identical to a SINGLE-track reference render
of its home track — if any foreign tile entered frame, the frames would differ.

Needs a working Madrona batch renderer (GPU). On a CUDA-13 toolkit, first run
``scripts/fix_madrona_cuda13.sh`` (see the README CUDA-13 note).

    python scripts/verify_track_tiling.py
"""

import torch

from deepracer_genesis._gs import ensure_init
from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs.base_env import DeepRacerEnv


def _build(tracks, n_envs=6):
    cfg = get_env_cfg(vision=True, track=tracks, backend="gpu")
    # deterministic spawn (waypoint 0, no noise) so tiled env 0 and the
    # single-track reference share an identical world pose
    cfg["spawn"]["random_start"] = False
    cfg["spawn"]["spawn_lateral_noise"] = 0.0
    cfg["spawn"]["spawn_yaw_noise"] = 0.0
    env = DeepRacerEnv(num_envs=n_envs, env_cfg=cfg)
    env.reset_idx(torch.arange(n_envs, device=env.device))
    env.step(torch.zeros(n_envs, 2, device=env.device))   # refresh image_buf
    return env


def main():
    ensure_init("gpu")
    tracks = ["reinvent_base", "reInvent2019_track"]

    tiled = _build(tracks)
    print("grid_spacing:", tiled.track.grid_spacing,
          "offsets:", tiled.track.variant_offset.tolist())
    print("variant_idx:", tiled.track.variant_idx.tolist())
    imgs = tiled.image_buf.float()
    assert torch.isfinite(imgs).all(), "non-finite camera frames"

    # each variant group must render a DIFFERENT track (not superimposed)
    va = imgs[tiled.track.variant_idx == 0][0]
    vb = imgs[tiled.track.variant_idx == 1][0]
    group_diff = (va - vb).abs().mean().item()
    print("variant-0 vs variant-1 mean|diff|:", round(group_diff, 4))
    assert group_diff > 1e-3, "variant groups render identically — tiling not applied"

    # NO-BLEED: tiled variant-0 frame must equal the single-track-A reference
    ref = _build([tracks[0]])
    refimg = ref.image_buf.float()[0]
    diff = (va - refimg).abs()
    print("tiled-A vs single-A  max|diff|:", diff.max().item(),
          " mean|diff|:", diff.mean().item())
    assert diff.max().item() < 2.0, "FOREIGN TILE BLEED: frame differs from single-track ref"
    print("\nNO-BLEED PASS: tiled camera frame is identical to the single-track render.")


if __name__ == "__main__":
    main()
