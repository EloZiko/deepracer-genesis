"""Les 7 canaux camera produits par le vrai CNN, au lieu du simu.

L'env rend l'image, le CNN gele la lit, la politique recoit ses estimations.
Les 22 canaux restants (actions passees, ecarts de commande) restent calcules
a bord, comme sur la voiture.
"""

from pathlib import Path

import torch

from deepracer_genesis.envs.features import PerceptionFeatures

from model import PerceptionCNN

DEFAUT = Path(__file__).resolve().parent / "perception.pt"


class CNNPerception(PerceptionFeatures):
    """Params: ``checkpoint`` (chemin du .pt), sinon CNN/perception.pt."""

    def __init__(self, env, params: dict):
        super().__init__(env, params)
        chemin = params.get("checkpoint") or DEFAUT
        # l'env camera tourne sur CPU (pas de Madrona sur Mac) mais le CNN, lui,
        # gagne beaucoup a passer sur MPS : c'est lui qui domine le cout du pas.
        self.dev = ("mps" if params.get("cnn_device", "mps") == "mps"
                    and torch.backends.mps.is_available() else env.device)
        self.net = PerceptionCNN().to(self.dev).eval()
        self.net.load_state_dict(torch.load(chemin, map_location=self.dev))
        for p in self.net.parameters():
            p.requires_grad_(False)

    def compute(self) -> torch.Tensor:
        x = super().compute()
        pile = self.env._stack_buf          # (N, 12, H, W) dans [0, 1], plus ancien d'abord
        if pile is not None:
            lo, hi = self.cnn_target_slice
            with torch.inference_mode():
                y = self.net(pile.to(self.dev))
            x[:, lo:hi] = y.to(x.device)
        return x
