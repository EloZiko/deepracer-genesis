"""Part O spatial tiling: variant offsets are applied to absolute fields only,
so localize / lateral / progress stay offset-invariant (the whole reason tiling
is transparent to the rulebook and to car-relative features)."""

import torch

from deepracer_genesis.envs.track import MultiTrack, grid_offsets


def test_grid_offsets_layout_and_off_by_default():
    dev = torch.device("cpu")
    # no tiling: all zero
    assert torch.equal(grid_offsets(3, 0.0, dev), torch.zeros(3, 2))
    assert torch.equal(grid_offsets(1, 100.0, dev), torch.zeros(1, 2))
    # 4 variants on a 2-col grid at spacing 100
    off = grid_offsets(4, 100.0, dev)
    assert off.tolist() == [[0, 0], [100, 0], [0, 100], [100, 100]]


def test_single_track_never_offset():
    mt = MultiTrack(["reinvent_base"], num_envs=4, device=torch.device("cpu"),
                    grid_spacing=100.0)
    assert torch.equal(mt.variant_offset, torch.zeros(1, 2))


def test_localize_is_offset_invariant():
    dev = torch.device("cpu")
    names = ["reinvent_base", "reInvent2019_track"]
    plain = MultiTrack(names, num_envs=2, device=dev, grid_spacing=0.0)
    tiled = MultiTrack(names, num_envs=2, device=dev, grid_spacing=100.0)

    # variant 1 lands on the second tile; its waypoints shift by that offset
    off = tiled.variant_offset[1]
    assert not torch.allclose(off, torch.zeros(2))

    # a point sitting on variant-1's centerline, in each world frame
    wp = 5
    p_plain = plain.center[1, wp].clone()
    p_tiled = tiled.center[1, wp].clone()
    assert torch.allclose(p_tiled, p_plain + off, atol=1e-4)

    # localize env index 1 (variant 1 under balanced mapping over 2 envs)
    lp = plain.localize(p_plain[None], envs_idx=torch.tensor([1]))
    lt = tiled.localize(p_tiled[None], envs_idx=torch.tensor([1]))
    for key in ("lateral", "half_width", "progress_m", "track_yaw"):
        assert torch.allclose(lp[key], lt[key], atol=1e-4), key
