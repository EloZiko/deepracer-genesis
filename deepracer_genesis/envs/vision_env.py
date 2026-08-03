"""Provide the camera-observation DeepRacer environment.

Add a ``camera`` observation group backed by an image buffer refreshed each step.
"""

from __future__ import annotations

import torch

from .base_env import DeepRacerEnv


class VisionDeepRacerEnv(DeepRacerEnv):
    """DeepRacer environment that exposes a rendered camera observation.

    Owns the rendered image buffer and publishes it as an extra observation group.

    Attributes:
        image_buf: Full-resolution rendered camera frames for all parallel envs.
        obs_image_buf: Policy-resolution camera frames published as observations.
    """

    def _init_obs_buffers(self, env_cfg: dict) -> None:
        """Preallocate the camera image buffers for all parallel envs.

        Args:
            env_cfg: Environment configuration; ``camera_res`` gives the render
                resolution as a ``(width, height)`` pair.
        """
        w, h = env_cfg["vision"]["camera_res"]
        self.image_buf = torch.zeros(self.num_envs, 3, h, w, device=self.device)
        # policy may train below render resolution (demo videos); render() sets both
        self.obs_image_buf = self.image_buf
        # stateful temporal DR (camera latency / frame drop); None when disabled
        lat = int(self.image_aug.get("latency_steps", 0))
        drop = float(self.image_aug.get("frame_drop", 0.0))
        if lat or drop:
            from ..randomization.latency import FrameLatency
            self._frame_latency = FrameLatency(self.num_envs, lat, drop, self.device)
        else:
            self._frame_latency = None

    def _observe_camera(self) -> None:
        """Refresh the camera buffers from the renderer for the current state.

        Applies image-space DR to the observed frame; image_buf stays un-augmented.
        """
        self.image_buf, self.obs_image_buf = self.renderer.render(self)
        if self.image_aug:
            from ..randomization.image_aug import apply_image_aug
            self.obs_image_buf = apply_image_aug(self.obs_image_buf, self.image_aug)

    def _finalize_obs(self) -> None:
        """Advance the camera-latency buffer once for the step's final frame.

        Runs after any auto-reset re-render, so the delayed frame the policy
        sees reflects the buffer state exactly one step forward.
        """
        if self._frame_latency is not None:
            self.obs_image_buf = self._frame_latency.advance(self.obs_image_buf)

    def _reset_obs_dr(self, env_ids: torch.Tensor) -> None:
        """Drop camera-latency history for respawned envs (no cross-episode bleed)."""
        if self._frame_latency is not None:
            self._frame_latency.reset(env_ids)

    def _obs_groups(self) -> dict:
        """Assemble the observation groups, adding the ``camera`` frame.

        Returns:
            The base observation groups augmented with a ``camera`` entry
            holding the current policy-resolution image buffer.
        """
        groups = super()._obs_groups()
        groups["camera"] = self.obs_image_buf
        return groups
