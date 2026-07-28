"""Validate the theory: ANY algorithm following the rsl_rl VecEnv contract
(preallocated storage, in-place buffers, no per-step tensordict allocation)
avoids the quadrants async-alloc CUDA crash — not just PPO.

Drives our DeepRacerEnv (already a VecEnv: num_envs/num_actions/get_observations/
step per the rsl_rl contract) with a THIRD-PARTY off-policy SAC
(leggedrobotics/rsl_rl_sac, OffPolicyRunner + preallocated ReplayBuffer). SAC is
a stronger test than PPO: off-policy, replay buffer, gradient updates interleaved
with stepping = more torch churn, more tightly overlapped with genesis kernels.

If SAC stays crash-free under the same concurrency trigger that crashes the
TorchRL front-end (~33%/proc), the invariant is the CONTRACT, not the algorithm.

Needs rsl_rl_sac installed over rsl-rl-lib (see the run harness). Run one seed:
    python scripts/sac_validate.py --seed 0 --iters 520 --randomize
"""

import argparse
import tempfile

import genesis as gs

from deepracer_genesis._gs import ensure_init
from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.envs import DeepRacerEnv


class _SacEnvAdapter:
    """Presents our DeepRacerEnv to rsl_rl_sac unchanged, except env.scene["robot"]
    yields identity ([-1,1]) action scaling — our actions are already normalized,
    whereas SAC's _compute_action_scaling expects an Isaac-Lab joint-limit robot.
    Everything else (step/get_observations/num_envs/attribute writes) forwards to
    the real env, so the allocation behavior under test is exactly the real env's.
    """

    def __init__(self, env):
        import torch
        n, dev = env.num_actions, env.device
        lim = torch.zeros(1, n, 2, device=dev); lim[..., 0] = -1.0; lim[..., 1] = 1.0
        robot = type("R", (), {"data": type("D", (), {
            "soft_joint_pos_limits": lim,
            "default_joint_pos": torch.zeros(1, n, device=dev)})()})()
        scene = type("S", (), {"__getitem__": lambda s, k: robot,
                               "keys": lambda s: ["robot"]})()
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_scene", scene)

    def __getattr__(self, name):
        if name == "scene":
            return object.__getattribute__(self, "_scene")
        return getattr(object.__getattribute__(self, "_env"), name)

    def __setattr__(self, name, value):        # forward writes (e.g. episode_length_buf)
        setattr(object.__getattribute__(self, "_env"), name, value)


def build_train_cfg():
    # Minimal valid SAC config for rsl_rl_sac. All net/hyperparams that we omit
    # fall back to the library defaults. Small nets + modest replay buffer keep
    # 6 concurrent procs within GPU memory.
    return {
        "num_steps_per_env": 24,
        "save_interval": 10_000_000,   # effectively never checkpoint (test run)
        "obs_groups": {"actor": ["state"], "critic": ["state"]},
        "actor": {"class_name": "SACActorModel", "hidden_dims": [128, 128],
                  "activation": "elu"},
        "critic": {"class_name": "SACCriticModel", "hidden_dims": [128, 128],
                   "activation": "elu"},
        "algorithm": {
            "class_name": "SAC",
            "replay_buffer_size": 50_000,   # per-env transitions (preallocated)
            "gamma": 0.99,
            "mini_batch_size": 256,
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=520)   # ~200k steps @ 16 envs x 24
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--randomize", action="store_true")
    args = ap.parse_args()

    import torch
    torch.manual_seed(args.seed)

    env_cfg = get_env_cfg(vision=False, track="reinvent_base",
                          randomize=args.randomize, backend="gpu")
    ensure_init("gpu")
    env = DeepRacerEnv(num_envs=args.num_envs, env_cfg=env_cfg)

    from rsl_rl.runners.off_policy_runner import OffPolicyRunner
    runner = OffPolicyRunner(_SacEnvAdapter(env), build_train_cfg(),
                             log_dir=tempfile.mkdtemp(prefix="sac_"),
                             device=str(gs.device))
    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
    print("SAC_RUN_COMPLETED")


if __name__ == "__main__":
    main()
