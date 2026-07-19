"""Camera-observation DeepRacer env.

Adds the ``camera`` observation group on top of the shared base: preallocates
the image buffer, refreshes it each step from the renderer, and includes it in
the observation TensorDict. Which vision renderer (Madrona / Nyx) is used is
decided by the config (see :func:`~deepracer_genesis.envs.renderers.make_renderer`).
"""

from __future__ import annotations

import torch

from .base_env import DeepRacerEnv


class VisionDeepRacerEnv(DeepRacerEnv):
    def _init_obs_buffers(self, env_cfg: dict) -> None:
        w, h = env_cfg["camera_res"]
        self.image_buf = torch.zeros(self.num_envs, 3, h, w, device=self.device)
        # policy may train below render resolution (demo videos); render() sets both
        self.obs_image_buf = self.image_buf

    def _observe_camera(self) -> None:
        self.image_buf, self.obs_image_buf = self.renderer.render(self)

    def _obs_groups(self) -> dict:
        groups = super()._obs_groups()
        groups["camera"] = self.obs_image_buf
        return groups
