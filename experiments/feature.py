"""Feature-vector experiments — the baseline everything else is measured against.

New API notes:
- Registration is explicit — these functions are listed in
  ``experiments/__init__.py``'s ``EXPERIMENTS`` map (no ``@experiment``).
- The state vector is chosen by passing a ``FeatureSet`` *class* to
  ``FeatureEnvironment(feature_set=...)`` (``None`` -> ``ClassicFeatures``);
  ``feature_perception`` below uses the sim2real ``PerceptionFeatures`` vector.
"""

from deepracer_genesis.envs.features import PerceptionFeatures
from deepracer_genesis.experiment import (
    DomainRandomizationPhysics,
    FeatureEnvironment,
    VectorPolicy,
)


def feature_baseline():
    """Phase-1 baseline: state-vector PPO, no DR, no vision (ClassicFeatures)."""
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="baselines", variant="feature")


def feature_perception():
    """FeatureSet ablation: the sim2real PerceptionFeatures vector.

    Passes the ``FeatureSet`` class directly — the C.2 parameter-passing API
    that replaced the old string registry — so the env assembles the perception
    observation instead of the classic vector.
    """
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024,
                           feature_set=PerceptionFeatures)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="baselines", variant="perception")


def feature_dr():
    """dr_effect_feature pairing: physics-DR treatment on the feature env."""
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024)
        >> DomainRandomizationPhysics()
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="dr_effect_feature", variant="physics_dr")


def feature_nodr():
    """dr_effect_feature pairing: the no-DR baseline (same config as
    feature_baseline; shares its content id, tagged into this group)."""
    return (
        FeatureEnvironment(lookahead_k=10, num_envs=1024)
        >> VectorPolicy(keys=("state",))
    ).build(seed=0, ablation_group="dr_effect_feature", variant="no_dr")
