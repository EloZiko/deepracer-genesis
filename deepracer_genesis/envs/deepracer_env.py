"""Public entry point for the DeepRacer env.

``DeepRacerEnv(num_envs, env_cfg, ...)`` constructs the right env: it dispatches
by ``env_cfg["vision"]`` to :class:`VisionDeepRacerEnv` (camera obs) or
:class:`VectorDeepRacerEnv` (state obs). ``DeepRacerEnv`` is also the common base
class, so ``isinstance(env, DeepRacerEnv)`` and ``sim: DeepRacerEnv`` annotations
hold for both. The implementation lives in :mod:`deepracer_genesis.envs.base_env`
(shared loop), with rendering behind
:mod:`deepracer_genesis.envs.renderers` and the car behind
:mod:`deepracer_genesis.envs.entities`.
"""

from .base_env import DeepRacerEnv
from .vector_env import VectorDeepRacerEnv
from .vision_env import VisionDeepRacerEnv

__all__ = ["DeepRacerEnv", "VisionDeepRacerEnv", "VectorDeepRacerEnv"]
