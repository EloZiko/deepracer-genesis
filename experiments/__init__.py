"""Authored experiments — this package IS the config surface (pure Python).

Registration is EXPLICIT: the ``EXPERIMENTS`` map below lists every runnable
name and its handle (a spec factory or an ``Experiment`` subclass). Importing
this package calls :func:`deepracer_genesis.experiment.register` on that map,
so `run("cam_baseline")` and the CLI resolve names after ``import experiments``.
No decorator or subclass magic — to add an experiment, write it and add a line
here.
"""

from deepracer_genesis.experiment import register

from .camera import cam_baseline, cam_full_dr, cam_nyx, cam_plain
from .feature import (
    feature_baseline,
    feature_dr,
    feature_kinematic,
    feature_minimal,
    feature_nodr,
    feature_perception,
)
from .safe import SafeTransfer, SafeTransferNyx, SafeTransferTight, safe_feature
from .template import MyExperiment, MyExperimentNoDelay

EXPERIMENTS = {
    # camera (Env 1)
    "cam_baseline": cam_baseline,
    "cam_nyx": cam_nyx,
    "cam_full_dr": cam_full_dr,
    "cam_plain": cam_plain,
    # feature-vector baselines (each chooses its feature blocks)
    "feature_baseline": feature_baseline,
    "feature_minimal": feature_minimal,
    "feature_kinematic": feature_kinematic,
    "feature_perception": feature_perception,
    "feature_dr": feature_dr,
    "feature_nodr": feature_nodr,
    # safe-RL (Env 2)
    "safe_feature": safe_feature,
    "SafeTransfer": SafeTransfer,
    "SafeTransferNyx": SafeTransferNyx,
    "SafeTransferTight": SafeTransferTight,
    # template (copy-me starting points)
    "MyExperiment": MyExperiment,
    "MyExperimentNoDelay": MyExperimentNoDelay,
}

register(EXPERIMENTS)

__all__ = ["EXPERIMENTS"]
