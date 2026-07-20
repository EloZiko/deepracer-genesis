"""Export trained policies to ONNX plus a JSON model card, rebuilt on CPU.

Runs without genesis (it shares clashing LLVM symbols with onnxruntime).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

import torch
from torch import nn
from torchrl.modules import MLP, ConvNet

from ..experiment.run import build
from ..experiment.spec import ExperimentSpec
from ..physics.limits import MAX_SPEED, MAX_STEERING_DEG, MIN_SPEED

_ACT = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}

#: physical meaning of the normalized action channels (the env's mapping)
ACTION_PHYSICAL = {
    "steering": {"low": -MAX_STEERING_DEG, "high": MAX_STEERING_DEG, "unit": "deg"},
    "speed": {"low": MIN_SPEED, "high": MAX_SPEED, "unit": "m/s"},
}

def state_dim(spec: ExperimentSpec) -> int:
    """Width of the state vector (delegates to the spec's feature set)."""
    from ..envs.features import feature_dim
    return feature_dim(spec.env.feature_set, lookahead_k=spec.env.lookahead_k,
                       params=spec.env.feature_params)


def state_layout(spec: ExperimentSpec) -> str:
    """Channel-by-channel description of the state vector."""
    from ..envs.features import feature_layout
    return feature_layout(spec.env.feature_set,
                          lookahead_k=spec.env.lookahead_k,
                          params=spec.env.feature_params)


class _ExportActor(nn.Module):
    """Actor rebuilt as a plain nn.Module with a deterministic head.

    Attributes:
        keys: input observation keys in forward order.
        cnn: optional camera feature extractor.
        mlp: policy head mapping features to outputs.
        discrete: whether the head emits logits versus a continuous action.
    """

    def __init__(self, spec: ExperimentSpec, cnn: Optional[ConvNet], mlp: MLP):
        super().__init__()
        self.keys = list(spec.policy.actor_keys)
        self.cnn = cnn
        self.mlp = mlp
        self.discrete = spec.policy.actions is not None

    def forward(self, *obs):
        feats = []
        for key, x in zip(self.keys, obs):
            feats.append(self.cnn(x) if key == "camera" else x)
        out = self.mlp(torch.cat(feats, dim=-1))
        if self.discrete:
            return out                              # logits
        return torch.tanh(out[..., : out.shape[-1] // 2])   # loc -> action


def _sub_state_dict(sd: dict, prefix: str) -> dict:
    out = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not out:
        raise KeyError(f"no checkpoint keys under {prefix!r}; "
                       f"have {sorted(set(k.rsplit('.', 2)[0] for k in sd))[:6]}...")
    return out


def _rebuild_actor(spec: ExperimentSpec, actor_sd: dict) -> _ExportActor:
    """Mirror Builder.actor()'s module shapes on CPU and load the weights."""
    p = spec.policy
    discrete = p.actions is not None
    out_features = len(p.actions) if discrete else 4      # loc+scale

    cnn = None
    in_dim = 0
    if p.cnn is not None and "camera" in p.actor_keys:
        c = p.cnn
        cnn = ConvNet(in_features=3, num_cells=list(c["channels"]),
                      kernel_sizes=list(c["kernels"]), strides=list(c["strides"]),
                      activation_class=_ACT[c.get("activation", "relu")],
                      device="cpu")
        w, h = spec.env.resolution
        with torch.no_grad():
            in_dim += cnn(torch.zeros(1, 3, h, w, device="cpu")).shape[-1]
    for k in p.actor_keys:
        if k == "state":
            in_dim += state_dim(spec)
        elif k != "camera":
            in_dim += spec.encoder.output_dim

    m = p.mlp
    mlp = MLP(in_features=in_dim, out_features=out_features,
              num_cells=list(m.get("hidden", (256, 128, 64))),
              activation_class=_ACT[m.get("activation", "elu")], device="cpu")

    mlp_stage = 1 if cnn is not None else 0
    if cnn is not None:
        cnn.load_state_dict(_sub_state_dict(actor_sd, "module.0.module.0.module."))
    mlp.load_state_dict(_sub_state_dict(actor_sd, f"module.0.module.{mlp_stage}.module."))
    return _ExportActor(spec, cnn, mlp).eval()


def export_policy(target, *, root: str = "runs", ckpt: Optional[str] = None,
                  out: Optional[str] = None, opset: int = 17,
                  **overrides) -> str:
    """Export `target`'s trained actor to ONNX + model_card.json.

    Returns the export directory.
    """
    import sys
    if "genesis" in sys.modules:
        raise RuntimeError(
            "export_policy must run in a process that has NOT imported "
            "genesis: genesis and onnxruntime bundle clashing LLVM symbols "
            "and crash together. Run the export as its own step/script.")
    spec: ExperimentSpec = build(target, **overrides)
    run_dir = spec.run_dir(root)
    ckpt = ckpt or os.path.join(run_dir, "best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint at {ckpt} — train first")
    out = out or os.path.join(run_dir, "export")
    os.makedirs(out, exist_ok=True)

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    actor = _rebuild_actor(spec, payload["actor"])

    keys = list(spec.policy.actor_keys)
    discrete = spec.policy.actions is not None
    dummies = []
    for k in keys:
        if k == "camera":
            w, h = spec.env.resolution
            dummies.append(torch.zeros(1, 3, h, w, device="cpu"))
        else:
            dummies.append(torch.zeros(
                1, state_dim(spec) if k == "state" else spec.encoder.output_dim,
                device="cpu"))

    onnx_path = os.path.join(out, "policy.onnx")
    torch.onnx.export(
        actor, tuple(dummies), onnx_path, opset_version=opset,
        input_names=keys,
        output_names=["logits" if discrete else "action"],
        dynamic_axes={name: {0: "batch"} for name in
                      keys + (["logits"] if discrete else ["action"])},
        dynamo=False,
    )

    # verify the graph loads and matches torch outputs (on random inputs)
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        rand = [torch.rand_like(d) for d in dummies]
        feeds = {k: r.numpy() for k, r in zip(keys, rand)}
        (onnx_out,) = sess.run(None, feeds)
        with torch.no_grad():
            torch_out = actor(*rand).numpy()
        assert np.allclose(onnx_out, torch_out, atol=1e-4), "ONNX/torch mismatch"
        verified = True
    except ImportError:
        verified = False

    card = {
        "policy": {
            "file": "policy.onnx",
            "sha256": hashlib.sha256(open(onnx_path, "rb").read()).hexdigest(),
            "opset": opset,
            "verified_against_torch": verified,
        },
        "action_space": (
            {"type": "discrete",
             "size": len(spec.policy.actions),
             "output": "logits (argmax -> index into `table`)",
             "table": [{"steer": a[0], "speed": a[1]} for a in spec.policy.actions],
             "normalized_to_physical": ACTION_PHYSICAL}
            if discrete else
            {"type": "continuous",
             "output": "action (B, 2) = [steer, speed] in [-1, 1]",
             "normalized_to_physical": ACTION_PHYSICAL}
        ),
        "observations": {
            k: ({"shape": [3, spec.env.resolution[1], spec.env.resolution[0]],
                 "dtype": "float32 in [0, 1]", "fov_deg": spec.env.fov,
                 "camera": "front RGB"}
                if k == "camera" else
                {"shape": [state_dim(spec) if k == "state"
                           else spec.encoder.output_dim],
                 "dtype": "float32",
                 "layout": (state_layout(spec) if k == "state"
                            else "frozen encoder output")})
            for k in keys
        },
        "training": {
            "spec": spec.to_dict(),
            "spec_id": spec.id(),
            "checkpoint": os.path.abspath(ckpt),
            "metrics": _load_metrics(os.path.dirname(ckpt)) or _load_metrics(run_dir),
        },
    }
    with open(os.path.join(out, "model_card.json"), "w") as f:
        json.dump(card, f, indent=2)
    print(f"[export] policy.onnx + model_card.json -> {out}"
          + ("" if verified else " (onnxruntime not installed; skip verify)"))
    return out


def _load_metrics(run_dir: str) -> dict:
    p = os.path.join(run_dir, "eval_record.json")
    if os.path.exists(p):
        return json.load(open(p)).get("metrics", {})
    return {}
