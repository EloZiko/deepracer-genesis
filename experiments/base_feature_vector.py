"""Named feature vectors — each experiment chooses which features it uses.

A feature vector is an ordered *selection* of named blocks from
``deepracer_genesis.envs.features.FEATURE_BLOCKS`` (Part K.2), assembled by
``SelectFeatures``. Define a selection here once, then any feature experiment
picks one via :func:`feature_env`. Selecting every block (``FULL``) reproduces
the classic vector exactly.

Available blocks: ``v_forward``, ``v_lateral``, ``yaw_rate``, ``lateral``,
``heading`` (sin/cos), ``last_action``, ``lookahead_xy`` (2·lookahead_k).
"""

from deepracer_genesis.envs.features import SelectFeatures
from deepracer_genesis.experiment import FeatureEnvironment

# ---- named feature vectors (order matters: it is the channel order) ----
#: pose-only: where am I on the road and which way am I pointed
MINIMAL = ("v_forward", "lateral", "heading")
#: full kinematics, no track look-ahead
KINEMATIC = ("v_forward", "v_lateral", "yaw_rate", "lateral", "heading", "last_action")
#: kinematics + body-frame look-ahead waypoints (== ClassicFeatures)
FULL = KINEMATIC + ("lookahead_xy",)

#: registry so the CLI / reports can refer to a vector by name
FEATURE_VECTORS = {"minimal": MINIMAL, "kinematic": KINEMATIC, "full": FULL}


def feature_env(features, *, num_envs=1024, lookahead_k=10,
                tracks=("reinvent_base",)):
    """A feature-vector env that uses exactly ``features``.

    Args:
        features: Ordered tuple of block names (see :data:`FEATURE_VECTORS`
            for presets, or any subset of ``FEATURE_BLOCKS``).
        num_envs: Parallel sim instances.
        lookahead_k: Waypoints exposed to ``lookahead_xy``.
        tracks: Track name(s) to train on.

    Returns:
        A ``FeatureEnvironment`` stage wired to ``SelectFeatures`` with the
        chosen blocks.
    """
    return FeatureEnvironment(
        num_envs=num_envs, lookahead_k=lookahead_k, tracks=tracks,
        feature_set=SelectFeatures,
        feature_params={"features": tuple(features)})
