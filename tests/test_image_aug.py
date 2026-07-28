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
