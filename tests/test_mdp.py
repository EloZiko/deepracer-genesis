"""MDP transforms that need no live sim (mdp.map_action)."""

import math

import torch

from deepracer_genesis.envs import mdp


class _StubEnv:
    """Minimal stand-in exposing only what map_action reads."""

    def __init__(self, actions, caps):
        self.actions = actions
        self.cfg = {"action": caps}


CAPS = {"max_steering_deg": 30.0, "min_speed": 0.1, "max_speed": 4.0}


def test_map_action_matches_inline_formula():
    actions = torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.3, -0.4], [0.0, 0.0]])
    steer, speed = mdp.map_action(_StubEnv(actions, CAPS))
    exp_steer = actions[:, 0:1] * math.radians(CAPS["max_steering_deg"])
    exp_speed = CAPS["min_speed"] + (actions[:, 1:2] + 1) * 0.5 * (
        CAPS["max_speed"] - CAPS["min_speed"])
    assert torch.equal(steer, exp_steer) and torch.equal(speed, exp_speed)


def test_map_action_hits_speed_caps_at_extremes():
    steer, speed = mdp.map_action(
        _StubEnv(torch.tensor([[0.0, -1.0], [0.0, 1.0]]), CAPS))
    assert torch.allclose(speed, torch.tensor([[0.1], [4.0]]))


def test_map_action_respects_overridden_caps():
    caps = {"max_steering_deg": 20.0, "min_speed": 0.5, "max_speed": 2.0}
    steer, speed = mdp.map_action(_StubEnv(torch.tensor([[1.0, 1.0]]), caps))
    assert torch.allclose(steer, torch.tensor([[math.radians(20.0)]]))
    assert torch.allclose(speed, torch.tensor([[2.0]]))
