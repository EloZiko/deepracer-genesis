"""Export trained policies to ONNX plus a JSON model card, rebuilt on CPU.

Runs without genesis (it shares clashing LLVM symbols with onnxruntime).
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Optional

from ..experiment.spec import ExperimentSpec
from ..physics.limits import MAX_SPEED, MAX_STEERING_DEG, MIN_SPEED

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


def export_policy(target, *, root: str = "runs", ckpt: Optional[str] = None,
                  out: Optional[str] = None, opset: int = 17,
                  **overrides) -> str:
    """Export a trained rsl-rl policy to ONNX (not yet ported).

    Args:
        target: Any experiment handle.
        root: Runs directory the run dir resolves under.
        ckpt: Checkpoint path; defaults to model.pt in the run dir.
        out: Output directory; defaults to <run_dir>/export.
        opset: ONNX opset version.
        **overrides: Keyword overrides forwarded to build(target).

    Raises:
        NotImplementedError: Offline ONNX export is not yet on the rsl-rl backend.
    """
    raise NotImplementedError(
        "offline ONNX export is not yet ported to the rsl-rl backend; use "
        "OnPolicyRunner.export_policy_to_onnx() in-process, or port this "
        "rebuild to the rsl-rl model format (model card helpers below still apply)")



def _load_metrics(run_dir: str) -> dict:
    p = os.path.join(run_dir, "eval_record.json")
    if os.path.exists(p):
        return json.load(open(p)).get("metrics", {})
    return {}
