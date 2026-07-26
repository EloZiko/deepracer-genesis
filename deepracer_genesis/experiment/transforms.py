"""TorchRL transforms for the training pipeline.

The DR transform *definitions* (``ImageAug``, ``ActionNoiseDelay``) moved to
``deepracer_genesis.randomization.actuation`` so the randomization folder is the
single home for DR (Part L); they are re-exported here so the builder / dataset
pipeline imports keep working. ``FrozenEncoder`` (an encoder stage, not DR)
stays here.
"""

from __future__ import annotations

import torch

from torchrl.envs.transforms import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance

# Re-export the DR transforms from their new home (Part L).
from ..randomization.actuation import ActionNoiseDelay, ImageAug  # noqa: F401


class FrozenEncoder(Transform):
    """Run a frozen module over an obs key, write a new key, drop the raw one.

    The raw camera key is dropped from tensordict and spec so downstream never carries pixels.

    Attributes:
        encoder: frozen module applied to each image to produce embeddings.
        embed_dim: length of the embedding vector written per element.
        del_keys: whether the raw in_keys are removed after encoding.
    """

    def __init__(self, encoder, embed_dim: int, in_keys=("camera",),
                 out_keys=("encoded",), del_keys: bool = True):
        """Freeze the encoder and record the output embedding dim."""
        super().__init__(in_keys=list(in_keys), out_keys=list(out_keys))
        encoder.eval()
        encoder.requires_grad_(False)
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.del_keys = del_keys

    @torch.no_grad()
    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode the obs batch into an embed_dim vector per element."""
        lead = obs.shape[:-3]
        out = self.encoder(obs.reshape(-1, *obs.shape[-3:]))
        return out.reshape(*lead, self.embed_dim)

    def _call(self, next_tensordict):
        """Encode then drop the raw in_keys when del_keys is set."""
        next_tensordict = super()._call(next_tensordict)
        if self.del_keys:
            next_tensordict = next_tensordict.exclude(*self.in_keys)
        return next_tensordict

    forward = _call

    def _reset(self, tensordict, tensordict_reset):
        """Populate out_keys on the reset tensordict."""
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    _reset_on_native_autoreset = _reset

    def transform_observation_spec(self, observation_spec):
        """Replace the raw camera spec with the embedding spec."""
        from torchrl.data import Unbounded
        observation_spec = observation_spec.clone()
        ref = observation_spec[self.in_keys[0]]
        lead = ref.shape[:-3]
        if self.del_keys:
            for k in self.in_keys:
                del observation_spec[k]
        for k in self.out_keys:
            observation_spec[k] = Unbounded(shape=(*lead, self.embed_dim),
                                            device=ref.device)
        return observation_spec
