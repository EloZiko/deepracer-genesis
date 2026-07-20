"""TorchRL EnvBase wrapper over the GPU-batched DeepRacerEnv sim.

Uses the sim's native autoreset; terminated is crash/offtrack, truncated is timeout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase

from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

if TYPE_CHECKING:
    from .deepracer_env import DeepRacerEnv


class TorchRLDeepRacerEnv(EnvBase):
    """TorchRL EnvBase adapter for the batched DeepRacerEnv sim.

    Attributes:
        sim: The wrapped GPU-batched DeepRacer simulation.
        emit_cost: Whether a cost signal is emitted alongside the reward.
        observation_spec: Spec for state and optional camera observations.
        action_spec: Spec for discrete or continuous actions.
        full_reward_spec: Spec for reward and optional cost leaves.
        full_done_spec: Spec for done, terminated, and truncated flags.
    """

    def __init__(self, sim: DeepRacerEnv, emit_cost: bool = False) -> None:
        """Build specs from the sim and enable native autoreset."""
        n = sim.num_envs
        device = sim.device
        super().__init__(device=device, batch_size=[n])
        self.sim: DeepRacerEnv = sim
        self.emit_cost = emit_cost

        obs = {"state": Unbounded(shape=(n, sim.num_state_obs),
                                  dtype=torch.float32, device=device)}
        if sim.vision:
            w, h = sim.cfg["camera_res"]
            obs["camera"] = Unbounded(shape=(n, 3, h, w),
                                      dtype=torch.float32, device=device)
        self.observation_spec = Composite(**obs, shape=(n,), device=device)
        if sim.action_table is not None:
            self.action_spec = Categorical(
                n=sim.action_table.shape[0], shape=(n,),
                device=sim.device, dtype=torch.long)
        else:
            self.action_spec = Bounded(-1.0, 1.0, shape=(n, 2),
                                   dtype=torch.float32, device=device)
        reward = {"reward": Unbounded(shape=(n, 1), dtype=torch.float32, device=device)}
        if emit_cost:
            # cost rides in the reward spec: reward-like keys are not NaN-filled
            # by the autoreset machinery, so the cost-GAE sees clean streams
            reward["cost"] = Unbounded(shape=(n, 1), dtype=torch.float32, device=device)
        self.full_reward_spec = Composite(**reward, shape=(n,), device=device)
        self.full_done_spec = Composite(
            done=Categorical(2, shape=(n, 1), dtype=torch.bool, device=device),
            terminated=Categorical(2, shape=(n, 1), dtype=torch.bool, device=device),
            truncated=Categorical(2, shape=(n, 1), dtype=torch.bool, device=device),
            shape=(n,), device=device)

        self._torchrl_native_autoreset = True

    # ------------------------------------------------------------------
    def _obs_leaves(self, obs_td: TensorDictBase) -> dict[str, torch.Tensor]:
        """Return the observation leaf tensors keyed by spec name."""
        leaves = {"state": obs_td["state"]}
        if self.sim.vision:
            leaves["camera"] = obs_td["camera"]
        return leaves

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        """Advance the sim one step and pack obs, reward, and done flags."""
        obs_td, rew, dones, _extras = self.sim.step(tensordict["action"])
        info = self.sim.step_info
        n1 = (*self.batch_size, 1)
        terminated = (info["offtrack"] | info["flipped"]).reshape(n1)
        truncated = (info["time_out"] & ~terminated.reshape(-1)).reshape(n1)
        out = {
            **self._obs_leaves(obs_td),            # post-reset obs on done rows
            "reward": rew.reshape(n1),
            "terminated": terminated,
            "truncated": truncated,
            "done": dones.reshape(n1),
        }
        if self.emit_cost:
            out["cost"] = self.sim.cost_buf.reshape(n1)
        return TensorDict(out, batch_size=self.batch_size, device=self.device)

    def _reset(self, tensordict: TensorDictBase | None, **kwargs) -> TensorDict:
        """Reset the masked sub-envs and return fresh observations."""
        mask = tensordict.get("_reset", None) if tensordict is not None else None
        if mask is None:
            ids = torch.arange(self.sim.num_envs, device=self.device)
        else:
            ids = mask.reshape(-1).nonzero(as_tuple=True)[0]
        if len(ids):
            self.sim.reset_idx(ids)
            self.sim._post_physics(ids)
        obs_td = self.sim.get_observations()
        z = torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
        return TensorDict(
            {**self._obs_leaves(obs_td),
             "done": z, "terminated": z.clone(), "truncated": z.clone()},
            batch_size=self.batch_size, device=self.device)

    def _set_seed(self, seed: int | None) -> None:
        """Seed the global torch RNG used for sim spawn noise."""
        if seed is not None:
            torch.manual_seed(seed)   # sim spawn noise uses the global RNG
