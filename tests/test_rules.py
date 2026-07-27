"""Unit tests for the stateless world-fact predicates (Part B.4 / Finding F.5).

These are the subtle off-track / flip / angle-wrap laws that used to live inline
in the env and had no coverage. Pure batched-torch functions, so they test on
tiny synthetic tensors with no sim.
"""

import math

import torch

from deepracer_genesis.envs.rules import (
    is_flipped,
    is_off_track,
    up_z_from_quat,
    wrap,
    yaw_from_quat,
)


def _quat_about(axis: str, angle: float) -> torch.Tensor:
    """wxyz quaternion for a rotation of `angle` about a principal axis."""
    h = angle / 2
    w, s = math.cos(h), math.sin(h)
    v = {"x": (s, 0, 0), "y": (0, s, 0), "z": (0, 0, s)}[axis]
    return torch.tensor([[w, *v]], dtype=torch.float32)


# ------------------------------------------------------------------------ wrap
def test_wrap_folds_into_pi_interval():
    # interior values (avoid the +/-pi seam, which is float-fragile)
    a = torch.tensor([0.0, 0.5, -0.5, math.pi + 0.5, -math.pi - 0.5,
                      2 * math.pi + 0.3, -2 * math.pi - 0.3])
    w = wrap(a)
    assert torch.allclose(w, torch.tensor(
        [0.0, 0.5, -0.5, -math.pi + 0.5, math.pi - 0.5, 0.3, -0.3]), atol=1e-5)
    assert (w.abs() <= math.pi + 1e-5).all()


# --------------------------------------------------------------- yaw_from_quat
def test_yaw_identity_is_zero():
    assert torch.allclose(yaw_from_quat(_quat_about("z", 0.0)),
                          torch.zeros(1), atol=1e-6)


def test_yaw_recovers_z_rotation():
    for angle in (math.pi / 2, -math.pi / 2, 2.0):
        got = yaw_from_quat(_quat_about("z", angle))
        assert torch.allclose(got, torch.tensor([angle]), atol=1e-5)


def test_yaw_ignores_roll_about_x():
    # a pure roll about x has zero heading
    assert torch.allclose(yaw_from_quat(_quat_about("x", 0.9)),
                          torch.zeros(1), atol=1e-6)


# -------------------------------------------------------------- up_z_from_quat
def test_up_z_upright_flat_flipped():
    assert torch.allclose(up_z_from_quat(_quat_about("z", 1.0)),
                          torch.ones(1), atol=1e-6)          # yaw keeps it upright
    assert torch.allclose(up_z_from_quat(_quat_about("x", math.pi / 2)),
                          torch.zeros(1), atol=1e-6)          # on its side
    assert torch.allclose(up_z_from_quat(_quat_about("x", math.pi)),
                          -torch.ones(1), atol=1e-6)          # upside down


# ---------------------------------------------------------------- is_off_track
def test_off_track_threshold_and_margin():
    lateral = torch.tensor([0.0, 0.25, -0.25, 0.4, -0.4])
    half_width = torch.full((5,), 0.3)
    off = is_off_track(lateral, half_width, margin=0.05)
    # |lateral| > 0.35 => off; sign-symmetric
    assert off.tolist() == [False, False, False, True, True]


def test_off_track_is_strict_inequality():
    lateral = torch.tensor([0.35])
    half_width = torch.tensor([0.3])
    assert not is_off_track(lateral, half_width, margin=0.05).item()  # exactly on edge


def test_off_track_per_env_half_width():
    lateral = torch.tensor([0.4, 0.4])
    half_width = torch.tensor([0.3, 0.5])   # wider road in env 1
    off = is_off_track(lateral, half_width, margin=0.0)
    assert off.tolist() == [True, False]


# ------------------------------------------------------------------ is_flipped
def test_flipped_threshold():
    up_z = torch.tensor([1.0, 0.31, 0.30, 0.29, -1.0])
    flipped = is_flipped(up_z, thresh=0.3)
    # strict <: exactly 0.30 is NOT flipped
    assert flipped.tolist() == [False, False, False, True, True]


def test_flipped_default_threshold_matches_env():
    # the env historically used up_z < 0.3
    assert is_flipped(torch.tensor([0.29])).item() is True
    assert is_flipped(torch.tensor([0.31])).item() is False


# ------------------------------------------------------------- lap_progress
def test_lap_progress_normal_step():
    from deepracer_genesis.envs.rules import lap_progress
    L = torch.tensor([20.0, 20.0])
    dp, crossed = lap_progress(torch.tensor([5.0, 5.0]), torch.tensor([5.05, 4.95]),
                               L, torch.tensor([1.0, 1.0]), max_step=0.12)
    assert torch.allclose(dp, torch.tensor([0.05, -0.05]), atol=1e-6)
    assert crossed.tolist() == [False, False]


def test_lap_progress_wraps_finish_line():
    from deepracer_genesis.envs.rules import lap_progress
    L = torch.tensor([20.0])
    # crossed finish: 19.98 -> 0.02 is a +0.04 real step, not a -19.96 jump
    dp, crossed = lap_progress(torch.tensor([19.98]), torch.tensor([0.02]),
                               L, torch.tensor([1.0]), max_step=0.12)
    assert torch.allclose(dp, torch.tensor([0.04]), atol=1e-6)
    assert crossed.item() is True


def test_lap_progress_clamps_spurious_pinch_jump():
    """A multi-metre localize jump (below 0.5*L) is clamped to the bound."""
    from deepracer_genesis.envs.rules import lap_progress
    L = torch.tensor([20.0])
    dp, crossed = lap_progress(torch.tensor([5.0]), torch.tensor([7.7]),  # +2.7 m jump
                               L, torch.tensor([1.0]), max_step=0.12)
    assert abs(dp.item() - 0.12) < 1e-6    # clamped to the bound, not 2.7
    assert crossed.item() is False         # 2.7 < 0.5*L, so not a lap


def test_lap_progress_reversed_direction():
    from deepracer_genesis.envs.rules import lap_progress
    L = torch.tensor([20.0])
    # reversed car: arclength DEcreases, but that is forward progress for it
    dp, _ = lap_progress(torch.tensor([5.0]), torch.tensor([4.95]),
                         L, torch.tensor([-1.0]), max_step=0.12)
    assert torch.allclose(dp, torch.tensor([0.05]), atol=1e-6)   # positive progress
