"""Unit tests for override() + coupled-field sync."""

import pytest

import experiments  # noqa: F401
from deepracer_genesis.experiment import SpecError, run
from deepracer_genesis.experiment.ablation import override


def test_budget_sync_env_to_algorithm_and_back():
    base = run("safe_feature", build_only=True)
    s = override(base, "env.cost_budget", 10.0)
    assert s.algorithm.lagrangian["budget"] == 10.0
    s2 = override(base, "algorithm.lagrangian.budget", 40.0)
    assert s2.env.cost_budget == 40.0


def test_divergent_budgets_rejected():
    base = run("safe_feature", build_only=True)
    s = override(base, "algorithm.lagrangian", dict(base.algorithm.lagrangian,
                                                    budget=99.0))
    with pytest.raises(SpecError, match="conflicting budgets"):
        s.validate()


def test_override_unknown_path():
    base = run("feature_baseline", build_only=True)
    with pytest.raises((SpecError, TypeError)):
        override(base, "env.does_not_exist", 1)
