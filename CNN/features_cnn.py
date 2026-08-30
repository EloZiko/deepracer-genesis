"""PerceptionFeatures avec l'erreur du CNN injectee sur les canaux camera.

Modelise le deploiement : les 7 premiers canaux viennent du CNN, donc bruites ;
les 22 suivants (actions passees, ecarts de commande) sont calcules a bord et
restent exacts.
"""

import torch

from deepracer_genesis.envs.features import PerceptionFeatures

# erreur type du CNN sur chaque canal : racine de la MSE de validation
SIGMA = (0.125, 0.064, 0.060, 0.065, 0.083, 0.122, 0.224)


class CNNFeatures(PerceptionFeatures):
    """Params: ``noise`` multiplie SIGMA (0 = perception parfaite, 1 = le CNN)."""

    def compute(self) -> torch.Tensor:
        x = super().compute()
        force = float(self.params.get("noise", 0.0))
        if force:
            lo, hi = self.cnn_target_slice
            sigma = torch.tensor(SIGMA[:hi - lo], device=x.device, dtype=x.dtype)
            x[:, lo:hi] += torch.randn_like(x[:, lo:hi]) * sigma * force
        return x
