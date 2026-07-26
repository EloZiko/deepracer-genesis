"""Domain-randomization definitions and the shared HPO/DR search-space types.

- ``spaces``: declarative range/choice *types* (``FloatRange``/``IntRange``/
  ``SymRange``/``Choice``) shared by DR and HPO (Part H).
- ``domain_rand``: per-env physics randomization applied at reset.
- ``appearance``: offline texture bake (not wired into the train path).
"""

from .spaces import Choice, FloatRange, IntRange, Space, SymRange

__all__ = ["Space", "FloatRange", "IntRange", "SymRange", "Choice"]
