"""PerceptionFeatures with the CNN's measured error injected on the camera channels.

Models deployment without paying for the renderer: the first 7 channels come
from the CNN, so they are noisy; the next 22 (past actions, command deltas) are
computed onboard and stay exact.
"""

import torch

from deepracer_genesis.envs.features import PerceptionFeatures

CHANNEL_NAMES = ("lateral", "heading", "speed", "yaw_rate", "beta",
                 "curv@1m", "curv@3m")

# the CNN's typical error on each channel: root of the validation MSE
SIGMA = (0.125, 0.064, 0.060, 0.065, 0.083, 0.122, 0.224)


class NoisyPerceptionFeatures(PerceptionFeatures):
    """Params:

    noise: scales SIGMA (0 = perfect perception, 1 = the CNN's).
    noise_channels: names of the channels to corrupt; None = all of them.
    """

    def _sigma(self, device, dtype) -> torch.Tensor:
        chosen = self.params.get("noise_channels")
        s = [v if chosen is None or n in chosen else 0.0
             for n, v in zip(CHANNEL_NAMES, SIGMA)]
        return torch.tensor(s, device=device, dtype=dtype)

    def compute(self) -> torch.Tensor:
        x = super().compute()
        strength = float(self.params.get("noise", 0.0))
        if strength:
            lo, hi = self.cnn_target_slice
            sigma = self._sigma(x.device, x.dtype)[:hi - lo]
            x[:, lo:hi] += torch.randn_like(x[:, lo:hi]) * sigma * strength
        return x
