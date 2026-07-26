"""Signal bus: lazy per-step cache + metadata (Part K.1).

No sim — a tiny fake env carries the attributes signals read, so we can verify
the bus is lazy (a heavy signal nobody reads never computes), caches within a
step, and recomputes after invalidate().
"""

import torch

from deepracer_genesis.envs.signals import Signal, SignalBus


class _FakeEnv:
    """Minimal stand-in exposing the attributes the signal defs read."""

    def __init__(self):
        n = 4
        self.v_forward = torch.arange(n, dtype=torch.float32)
        self.lateral = torch.tensor([0.0, 0.2, -0.4, 0.5])
        self.half_width = torch.full((n,), 0.3)
        self.heading_err = torch.zeros(n)
        self.actions = torch.zeros(n, 2)
        self.last_actions = torch.zeros(n, 2)
        self.cfg = {"termination": {"wheel_margin": 0.0}}


def _counting_registry(counter):
    def compute(env):
        counter["n"] += 1
        return env.v_forward * 2

    return {"double_v": Signal("double_v", compute, pixel_observable=False)}


def test_signal_read_returns_value():
    bus = SignalBus(_FakeEnv())
    assert torch.equal(bus["v_forward"], torch.arange(4, dtype=torch.float32))


def test_lazy_compute_is_cached_within_step():
    counter = {"n": 0}
    bus = SignalBus(_FakeEnv(), _counting_registry(counter))
    bus["double_v"]; bus["double_v"]; bus.get("double_v")
    assert counter["n"] == 1, "signal recomputed instead of cached"


def test_invalidate_forces_recompute():
    counter = {"n": 0}
    bus = SignalBus(_FakeEnv(), _counting_registry(counter))
    bus["double_v"]
    bus.invalidate()
    bus["double_v"]
    assert counter["n"] == 2


def test_unread_signal_never_computes():
    """A registry with two signals: reading one must not compute the other."""
    counter = {"n": 0}
    reg = _counting_registry(counter)
    reg["v_forward"] = Signal("v_forward", lambda e: e.v_forward, pixel_observable=False)
    bus = SignalBus(_FakeEnv(), reg)
    bus["v_forward"]
    assert counter["n"] == 0, "an unread signal was eagerly computed"


def test_off_track_and_metadata():
    bus = SignalBus(_FakeEnv())
    off = bus["off_track"]                       # |lateral| > 0.3 => envs 2,3
    assert off.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert bus.meta("d_progress").pixel_observable is False
    assert bus.meta("lateral").pixel_observable is True
    assert bus.meta("curvature_ahead").cost == "heavy"


def test_unknown_signal_raises():
    bus = SignalBus(_FakeEnv())
    try:
        bus["nope"]
    except KeyError as e:
        assert "unknown signal" in str(e)
    else:
        raise AssertionError("expected KeyError for unknown signal")


# ------------------------------------------------- K.5 learnability check
def test_learnability_flags_signal_missing_from_critic():
    from deepracer_genesis.envs.signals import check_learnability
    probs = check_learnability(reward_reads={"off_track", "d_progress"},
                               cost_reads=set(),
                               actor_signals={"lateral"},
                               critic_signals={"off_track"})   # d_progress hidden
    assert any("unlearnable" in p and "d_progress" in p for p in probs)


def test_learnability_ok_when_critic_sees_all():
    from deepracer_genesis.envs.signals import check_learnability
    probs = check_learnability(reward_reads={"d_progress", "off_track"},
                               cost_reads=set(),
                               actor_signals={"*"}, critic_signals={"*"})
    assert probs == []


def test_learnability_warns_actor_cannot_act_on_privileged_signal():
    from deepracer_genesis.envs.signals import check_learnability
    # d_progress is not pixel_observable; critic sees it, actor doesn't
    probs = check_learnability(reward_reads={"d_progress"}, cost_reads=set(),
                               actor_signals={"lateral"}, critic_signals={"*"})
    assert any("cannot act on it" in p for p in probs)
    assert not any("unlearnable" in p for p in probs)   # critic sees it -> learnable
