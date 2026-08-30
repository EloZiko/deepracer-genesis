"""PerceptionFeatures avec l'erreur du CNN injectee sur les canaux camera.

Modelise le deploiement : les 7 premiers canaux viennent du CNN, donc bruites ;
les 22 suivants (actions passees, ecarts de commande) sont calcules a bord et
restent exacts.
"""

import torch

from deepracer_genesis.envs.features import PerceptionFeatures

NOMS = ("lateral", "heading", "speed", "yaw_rate", "beta", "curv@1m", "curv@3m")

# erreur type du CNN sur chaque canal : racine de la MSE de validation
SIGMA = (0.125, 0.064, 0.060, 0.065, 0.083, 0.122, 0.224)


class CNNFeatures(PerceptionFeatures):
    """Params:

    noise: multiplie SIGMA (0 = perception parfaite, 1 = celle du CNN).
    noise_channels: noms des canaux bruites; None = tous.
    """

    def _sigma(self, device, dtype) -> torch.Tensor:
        choisis = self.params.get("noise_channels")
        s = [v if choisis is None or n in choisis else 0.0
             for n, v in zip(NOMS, SIGMA)]
        return torch.tensor(s, device=device, dtype=dtype)

    def compute(self) -> torch.Tensor:
        x = super().compute()
        force = float(self.params.get("noise", 0.0))
        if force:
            lo, hi = self.cnn_target_slice
            sigma = self._sigma(x.device, x.dtype)[:hi - lo]
            x[:, lo:hi] += torch.randn_like(x[:, lo:hi]) * sigma * force
        return x
