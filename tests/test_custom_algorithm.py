"""Custom-algorithm plug-in via Algo(cls=...) — the rsl-rl interface + wiring.

rsl-rl's OnPolicyRunner resolves cfg["algorithm"]["class_name"] and drives it, so
a custom Algorithm plugs into the SAME runner PPO uses. These CPU tests pin the
contract (no Genesis sim): a conforming class validates and its class object is
passed through to the runner cfg; a non-conforming class is rejected up front.
The end-to-end training is a GPU smoke (scripts/verify_custom_algorithm.py).
"""

import pytest

from deepracer_genesis.algorithms.protocol import (
    ALGO_INTERFACE,
    RslAlgorithm,
    missing_algorithm_methods,
)
from deepracer_genesis.experiment import Algo, FeatureEnvironment, VectorPolicy
from deepracer_genesis.experiment.rsl_backend import rsl_supported, spec_to_train_cfg
from deepracer_genesis.experiment.spec import SpecError


class ConformingAlgo:
    """A structurally-conforming, no-op algorithm (plumbing test only)."""

    learning_rate = 0.001

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        return ConformingAlgo()

    def act(self, obs): ...
    def process_env_step(self, obs, rewards, dones, extras): ...
    def compute_returns(self, obs): ...
    def update(self): return {}
    def train_mode(self): ...
    def eval_mode(self): ...
    def broadcast_parameters(self): ...
    def get_policy(self): ...
    def save(self): return {}
    def load(self, loaded_dict, load_cfg, strict): return True


def _pipe(cls):
    return FeatureEnvironment(num_envs=8) >> VectorPolicy() >> Algo(cls=cls)


# ---------------------------------------------------------------- conformance
def test_conforming_algo_has_no_missing_methods():
    assert missing_algorithm_methods(ConformingAlgo) == []


def test_incomplete_algo_reports_missing_methods():
    class Bad:
        def act(self, obs): ...
    missing = missing_algorithm_methods(Bad)
    assert "update" in missing and "construct_algorithm" in missing
    assert "act" not in missing


def test_conforming_algo_satisfies_runtime_protocol():
    assert isinstance(ConformingAlgo(), RslAlgorithm)


def test_interface_covers_the_runner_calls():
    # the methods OnPolicyRunner drives on self.alg (guards against silent drift)
    for m in ("construct_algorithm", "act", "process_env_step",
              "compute_returns", "update", "get_policy"):
        assert m in ALGO_INTERFACE


# ------------------------------------------------------------ validate + wiring
def test_validate_rejects_nonconforming_custom_algo():
    class Bad:
        pass
    with pytest.raises(SpecError, match="rsl-rl interface"):
        _pipe(Bad).build()


def test_conforming_custom_algo_builds_and_passes_class_object():
    spec = _pipe(ConformingAlgo).build()
    assert spec.algorithm.cls is ConformingAlgo
    cfg = spec_to_train_cfg(spec)
    # the CLASS OBJECT is passed through (resolve_callable takes it directly) —
    # not stringified, so a notebook-defined class works too.
    assert cfg["algorithm"]["class_name"] is ConformingAlgo


def test_default_algo_keeps_ppo_class_name():
    spec = (FeatureEnvironment(num_envs=8) >> VectorPolicy()).build()
    assert spec_to_train_cfg(spec)["algorithm"]["class_name"] == "PPO"


def test_custom_continuous_algo_is_rsl_supported():
    assert rsl_supported(_pipe(ConformingAlgo).build()) is True
