"""rsl-rl execution backend for the experiment API (migration off TorchRL).

Drives ``DeepRacerEnv`` (already a VecEnv) through rsl-rl's ``OnPolicyRunner``,
whose preallocated rollout storage has no per-step tensordict churn — so it does
not provoke the quadrants async-alloc CUDA crash that the TorchRL collector does
(see CUDA_ASYNC_CRASH_ROOTCAUSE.md: rsl_rl-contract 0/78 vs TorchRL 2/6).

``OnPolicyRunner.learn`` is a closed loop, so we run it in **chunks** and do the
experiment framework's periodic eval / ``on_eval`` (HPO prune) / EvalRecord
around the chunks — preserving the Trainer's outer-loop features.

Phase 0 scope: symmetric, continuous-action, feature-vector PPO (physics DR ok).
Camera / asymmetric / cost / encoder / action-DR route to the TorchRL Trainer
until their phases land (see MIGRATION_TORCHRL_TO_RSLRL.md). ``rsl_supported``
gates the dispatch.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import torch

from .evaluator import EvalRecord, evaluate_policy

if TYPE_CHECKING:
    from .spec import ExperimentSpec

# our PPO-stage hyperparameter names -> rsl-rl algorithm cfg names
_PPO_KEY_MAP = {
    "clip": "clip_param",
    "epochs": "num_learning_epochs",
    "minibatches": "num_mini_batches",
    "gamma": "gamma",
    "gae_lambda": "lam",
    "lr": "learning_rate",
    "entropy_coef": "entropy_coef",
    "max_grad_norm": "max_grad_norm",
}


def rsl_supported(spec: "ExperimentSpec") -> bool:
    """True when this spec is in the migrated (rsl-rl backend) scope.

    Migrated: feature + camera PPO, symmetric or asymmetric (obs_groups is native
    to rsl-rl), continuous actions, physics DR (applied in the env, backend-
    agnostic). Still on the TorchRL Trainer until their phases: cost/Lagrangian,
    frozen-CNN encoder, action/image DR, discrete actions.
    """
    e, p = spec.env, spec.policy
    return (
        e.modality in ("feature", "camera")
        and not e.emits_cost
        and spec.encoder.kind == "none"
        and not spec.obs_dr.image_aug
        and not (spec.action_dr.delay_steps or spec.action_dr.steer_noise
                 or spec.action_dr.speed_noise)
        and p.actions is None                       # continuous only
    )


def spec_to_train_cfg(spec: "ExperimentSpec") -> dict:
    """Translate an ExperimentSpec into an rsl-rl ``OnPolicyRunner`` train cfg."""
    from ..configs.cfgs import get_train_cfg

    vision = spec.env.modality == "camera"
    cfg = get_train_cfg(vision=vision)
    cfg["obs_groups"] = {"actor": list(spec.policy.actor_keys),
                         "critic": list(spec.policy.critic_keys)}
    hidden = spec.policy.mlp.get("hidden")
    if hidden:
        cfg["actor"]["hidden_dims"] = list(hidden)
        cfg["critic"]["hidden_dims"] = list(hidden)
    if vision and spec.policy.cnn:                    # map the spec's CNN trunk
        c = spec.policy.cnn
        cnn_cfg = {"output_channels": list(c["channels"]),
                   "kernel_size": list(c["kernels"]),
                   "stride": list(c["strides"]),
                   "activation": c.get("activation", "relu"), "flatten": True}
        cfg["actor"]["cnn_cfg"] = cnn_cfg
        if "cnn_cfg" in cfg["critic"]:
            cfg["critic"]["cnn_cfg"] = dict(cnn_cfg)
    ppo = spec.algorithm.ppo
    for ours, theirs in _PPO_KEY_MAP.items():
        if ours in ppo:
            cfg["algorithm"][theirs] = ppo[ours]
    if "horizon" in ppo:
        cfg["num_steps_per_env"] = ppo["horizon"]
    return cfg


class _RslActor:
    """Adapt an rsl-rl inference policy to the evaluator's ``actor(td)`` call."""

    def __init__(self, policy):
        self._policy = policy

    def __call__(self, td):
        td.set("action", self._policy(td))
        return td


def _eval(sim, policy):
    # rsl-rl collection runs under torch.inference_mode(), which taints the sim's
    # mutable buffers as inference tensors; the eval rollout must therefore also
    # run inside inference_mode to be allowed to update them in place.
    with torch.inference_mode():
        return evaluate_policy(sim, _RslActor(policy))


def run_rsl(spec: "ExperimentSpec", root: str = "runs", on_eval=None) -> EvalRecord:
    """Train ``spec`` via rsl-rl's OnPolicyRunner, returning an EvalRecord.

    Runs ``learn`` in eval-cadence chunks so periodic eval + ``on_eval`` fire
    around rsl-rl's closed loop (parity with the TorchRL Trainer's outer loop).
    """
    from rsl_rl.runners import OnPolicyRunner

    from .builder import Builder

    assert spec.env is not None and spec.algorithm is not None
    sim = Builder(spec).sim()                 # reuses all sim_cfg logic (incl. DR)
    device = str(sim.device)
    run_dir = spec.run_dir(root)
    os.makedirs(run_dir, exist_ok=True)

    train_cfg = spec_to_train_cfg(spec)
    horizon = train_cfg["num_steps_per_env"]
    per_iter = spec.env.num_envs * horizon
    total_iters = max(1, spec.total_env_steps // per_iter)
    eval_iters = (max(1, spec.eval_every_steps // per_iter)
                  if spec.eval_every_steps else 0)

    runner = OnPolicyRunner(sim, train_cfg, run_dir, device=device)

    eval_history: list[dict] = []
    t0 = time.perf_counter()
    done_iters = 0
    first = True
    while done_iters < total_iters:
        chunk = min(eval_iters or total_iters, total_iters - done_iters)
        runner.learn(num_learning_iterations=chunk, init_at_random_ep_len=first)
        first = False
        done_iters += chunk
        frames = done_iters * per_iter
        if eval_iters:
            policy = runner.get_inference_policy(device=device)
            metrics = _eval(sim, policy)
            eval_history.append({"frames": frames, **metrics})
            if on_eval is not None:
                on_eval(frames, metrics)     # may raise to prune (HPO)

    wall = time.perf_counter() - t0
    policy = runner.get_inference_policy(device=device)
    metrics = _eval(sim, policy)
    ckpt = os.path.join(run_dir, "model.pt")
    try:
        runner.save(ckpt)
    except Exception:            # noqa: BLE001 - never lose the run over a save
        ckpt = ""

    record = EvalRecord(
        spec_id=spec.id(), spec=spec.to_dict(), seed=spec.seed,
        ablation_group=spec.ablation_group, variant=spec.variant,
        metrics=metrics, eval_history=eval_history, holdout={},
        train={"wall_clock_s": round(wall, 1),
               "total_env_steps": done_iters * per_iter,
               "steps_per_s": round(done_iters * per_iter / max(wall, 1e-9), 1),
               "checkpoint": ckpt, "backend": "rsl_rl"},
    )
    record.save(run_dir)
    return record
