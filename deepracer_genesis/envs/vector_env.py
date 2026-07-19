"""State-vector (feature) DeepRacer env — no camera.

The observation is the ``state`` group only; the base's :class:`NullRenderer`
produces no image (and only the optional spectator debug view). Kept as an
explicit class so the no-vision path is a distinct, typed contract rather than
an ``if vision`` branch.
"""

from __future__ import annotations

from .base_env import DeepRacerEnv


class VectorDeepRacerEnv(DeepRacerEnv):
    pass
