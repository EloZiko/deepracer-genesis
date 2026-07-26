"""Back-compat shim: physics DR moved to ``randomization.physics`` (Part L)."""

from .physics import _u, randomize_physics  # noqa: F401

__all__ = ["randomize_physics"]
