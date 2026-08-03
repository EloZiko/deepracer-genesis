"""Pure image-space DR math (randomization.image_aug.apply_image_aug)."""

import torch

from deepracer_genesis.randomization.image_aug import apply_image_aug


def test_brightness_is_multiplicative():
    out = apply_image_aug(torch.full((4, 3, 8, 8), 0.8), {"brightness": (0.5, 0.5)})
    assert torch.allclose(out, torch.full_like(out, 0.4), atol=1e-6)


def test_output_clamped_to_unit_range():
    out = apply_image_aug(torch.rand(2, 3, 8, 8), {"brightness": (3.0, 3.0), "noise": 0.5})
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_cutout_zeroes_a_patch():
    out = apply_image_aug(torch.ones(4, 3, 32, 32), {"cutout": 1.0})
    assert (out == 0).any() and (out == 1).any()


def test_hue_preserves_luma_direction():
    img = torch.rand(3, 3, 16, 16)
    out = apply_image_aug(img, {"hue": 0.2})
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0


# ---- Part P.2 photometric / geometric sensor block ----

def test_gamma_identity_at_one():
    img = torch.rand(4, 3, 8, 8)
    out = apply_image_aug(img, {"gamma": (1.0, 1.0)})
    assert torch.allclose(out, img, atol=1e-6)


def test_gamma_above_one_darkens_midtones():
    out = apply_image_aug(torch.full((4, 3, 8, 8), 0.5), {"gamma": (2.0, 2.0)})
    assert torch.allclose(out, torch.full_like(out, 0.25), atol=1e-6)


def test_white_balance_scales_channels_and_clamps():
    img = torch.full((4, 3, 8, 8), 0.5)
    out = apply_image_aug(img, {"white_balance": 0.4})
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0
    # a 40% per-channel gain magnitude cannot leave the gray image unchanged
    assert not torch.allclose(out, img)


def test_vignette_darkens_corners_not_center():
    out = apply_image_aug(torch.ones(1, 3, 16, 16), {"vignette": 0.4})[0, 0]
    assert out[0, 0] <= out[8, 8] and out.min() >= 0.0


def test_distortion_preserves_shape_and_range():
    img = torch.rand(4, 3, 16, 16)
    out = apply_image_aug(img, {"distortion": 0.2})
    assert out.shape == img.shape and torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_shot_noise_perturbs_and_clamps():
    img = torch.full((4, 3, 16, 16), 0.5)
    out = apply_image_aug(img, {"shot_noise": 0.05})
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.allclose(out, img)


def test_full_photometric_stack_is_valid():
    img = torch.rand(2, 4, 3, 24, 32)  # extra leading batch dim
    out = apply_image_aug(img, {
        "distortion": 0.15, "brightness": (0.8, 1.2), "gamma": (0.7, 1.4),
        "white_balance": 0.1, "vignette": 0.3, "shot_noise": 0.03,
    })
    assert out.shape == img.shape and torch.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
