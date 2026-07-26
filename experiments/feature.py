"""Feature-vector experiments — each one chooses which features it uses.

New API notes:
- Registration is explicit — these functions are listed in
  ``experiments/__init__.py``'s ``EXPERIMENTS`` map (no ``@experiment``).
- A feature vector is an ordered selection of named blocks, defined once in
  ``base_feature_vector.py`` and picked per experiment via ``feature_env(...)``
  (Part K.2). ``feature_perception`` instead passes the whole sim2real
  ``PerceptionFeatures`` class — the same C.2 parameter-passing API.
"""

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.experiment import (
    DomainRandomizationPhysics,
    FeatureEnvironment,
    VectorPolicy,
)

from .base_feature_vector import FULL, KINEMATIC, MINIMAL, feature_env


def feature_baseline():
    """Phase-1 baseline: the FULL feature vector (== classic), no DR, no vision."""
    return (
        feature_env(FULL)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="feature_vectors", variant="full")


def feature_minimal():
    """Feature ablation: pose-only (v_forward, lateral offset, heading)."""
    return (
        feature_env(MINIMAL)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="feature_vectors", variant="minimal")


def feature_kinematic():
    """Feature ablation: full kinematics but no track look-ahead."""
    return (
        feature_env(KINEMATIC)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="feature_vectors", variant="kinematic")


def feature_perception():
    """FeatureSet ablation: the sim2real PerceptionFeatures vector.

    Passes the ``FeatureSet`` class directly — the C.2 parameter-passing API
    that replaced the old string registry.
    """
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024,
                           feature_set=PerceptionFeatures)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="feature_vectors", variant="perception")


def feature_dr():
    """dr_effect_feature pairing: physics-DR treatment on the FULL feature env."""
    return (
        feature_env(FULL)
        >> DomainRandomizationPhysics()
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="dr_effect_feature", variant="physics_dr")


def feature_nodr():
    """dr_effect_feature pairing: the no-DR baseline (FULL feature vector)."""
    return (
        feature_env(FULL)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="dr_effect_feature", variant="no_dr")
