"""Headless GPU stress reproducer for the DR-reset CUDA illegal-memory-access.

Reproduces the sporadic ``CUDA_ERROR_ILLEGAL_ADDRESS`` that occurs when physics
domain randomization (genesis rigid-body setters fed torch-generated tensors)
runs on the GPU backend. No viewer, no TorchRL — just the env and a dense
DR-write loop, so a crash implicates the genesis<->torch seam and nothing else.

Modes:
    repo:   drive ``env.step`` + forced ``reset_idx`` — the real failing path,
            including whatever fences ``randomization/physics.py`` carries.
    direct: hammer the raw genesis setters back-to-back with interleaved torch
            allocations and NO fences — the tightest possible race window.

Differential levers (combine with either mode):
    --stream-fix            run torch AND quadrants on one shared CUDA stream
                            (torch moves to a dedicated stream; quadrants'
                            current stream is pinned to the same handle). If
                            the crash vanishes only here, the cross-stream
                            race is confirmed AND an env-level fix exists.
    CUDA_LAUNCH_BLOCKING=1  global kernel serialization (reported to hide it).
    PYTORCH_NO_CUDA_MEMORY_CACHING=1
                            disable torch's caching allocator; if the crash
                            vanishes, allocator block-recycling is the vector.

Exit status: 0 = survived all iterations, 1 = CUDA crash (details on stdout).
"""

import argparse
import sys
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=16,
                   help="parallel envs (16 mirrors the crashing watch_live run)")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--mode", choices=["repo", "direct"], default="direct")
    p.add_argument("--stream-fix", action="store_true",
                   help="pin quadrants' current CUDA stream to torch's stream")
    p.add_argument("--reset-frac", type=float, default=0.5,
                   help="fraction of envs force-reset (repo) / rewritten (direct) per iter")
    p.add_argument("--view", choices=["none", "gui"], default="none",
                   help="'gui' opens the interactive viewer (the crashing combo)")
    p.add_argument("--no-fence", action="store_true",
                   help="neutralize torch.cuda.synchronize + qd.sync fences "
                        "(recreates the pre-fix code that produced the crash)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def apply_stream_fix():
    """Put torch and quadrants on ONE CUDA stream.

    torch's default current stream is the legacy NULL stream (handle 0), and
    quadrants treats handle 0 as "use my internal default" — so sharing must go
    through a dedicated non-default stream: torch switches to it (thread-local,
    process-wide for this single-threaded repro) and quadrants' program-level
    current stream is pinned to the same raw handle.
    """
    from quadrants.lang import impl
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    handle = torch.cuda.current_stream().cuda_stream
    assert handle != 0, "expected a non-default torch stream"
    impl.get_runtime().prog.set_current_cuda_stream(handle)
    print(f"[repro] stream-fix: torch + quadrants pinned to stream {handle:#x}",
          flush=True)
    return stream  # keep alive


def build_env(num_envs, view="none"):
    from deepracer_genesis.configs.cfgs import get_env_cfg
    from deepracer_genesis.envs.base_env import DeepRacerEnv

    cfg = get_env_cfg(vision=False, track="reinvent_base", randomize=True,
                      backend="gpu", view=view)
    return DeepRacerEnv(num_envs=num_envs, env_cfg=cfg)


def loop_repo(env, args, pick_ids):
    """The real failing path: step + forced resets through reset_idx (fenced)."""
    acts = torch.empty(args.num_envs, 2, device=env.device)
    for i in range(args.iters):
        acts.uniform_(-1.0, 1.0)
        env.step(acts)
        ids = pick_ids()
        if len(ids):
            env.reset_idx(ids)
        yield i


def loop_direct(env, args, pick_ids):
    """Raw genesis setters, no fences, dense torch churn: tightest race window."""
    car = env.car
    dev = env.device
    car_cfg = env.cfg["car"]
    links_idx = torch.arange(car.n_links, device=dev)
    arm_dofs = env.wheel_dofs + env.steer_dofs  # python lists -> concatenation

    def u(lo, hi, shape):
        return lo + (hi - lo) * torch.rand(shape, device=dev)

    for i in range(args.iters):
        ids = pick_ids()
        n = len(ids)
        if n:
            car.set_friction_ratio(u(0.6, 1.4, (n, car.n_links)), links_idx, envs_idx=ids)
            car.set_mass_shift(u(-0.2, 0.2, (n, car.n_links)), links_idx, envs_idx=ids)
            car.set_COM_shift(u(-0.01, 0.01, (n, car.n_links, 3)), links_idx, envs_idx=ids)
            car.set_dofs_kp(car_cfg["steer_kp"] * u(0.8, 1.2, (n, 2)), env.steer_dofs, envs_idx=ids)
            car.set_dofs_kv(car_cfg["steer_kv"] * u(0.8, 1.2, (n, 2)), env.steer_dofs, envs_idx=ids)
            car.set_dofs_kv(car_cfg["wheel_kv"] * u(0.8, 1.2, (n, 4)), env.wheel_dofs, envs_idx=ids)
            car.set_dofs_armature(u(0.0, 0.01, (n, 6)), arm_dofs, envs_idx=ids)
        # torch-side allocation churn: freshly freed genesis temps are the
        # caching allocator's FIRST reuse candidates (LIFO) — grab them fast
        for sz in (64, 256, 1024, 4096):
            torch.rand(sz, device=dev)
        env.scene.step()  # concurrent quadrants physics work widens the window
        yield i


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    from deepracer_genesis._gs import ensure_init
    ensure_init("gpu")

    keep_alive = apply_stream_fix() if args.stream_fix else None  # noqa: F841

    real_sync = torch.cuda.synchronize
    if args.no_fence:
        import quadrants as qd
        torch.cuda.synchronize = lambda *a, **k: None
        qd.sync = lambda *a, **k: None  # covers _qd_sync in physics/base_env
        print("[repro] fences DISABLED (pre-fix condition)", flush=True)

    env = build_env(args.num_envs, view=args.view)
    ids_all = torch.arange(args.num_envs, device=env.device)

    def pick_ids():
        return ids_all[torch.rand(args.num_envs, device=env.device) < args.reset_frac]

    print(f"[repro] mode={args.mode} num_envs={args.num_envs} iters={args.iters} "
          f"stream_fix={args.stream_fix} reset_frac={args.reset_frac}", flush=True)

    loop = loop_repo if args.mode == "repo" else loop_direct
    t0 = time.time()
    last = -1
    try:
        for i in loop(env, args, pick_ids):
            last = i
            if i % args.log_every == 0:
                real_sync()  # surface pending async errors NOW
                print(f"[repro] iter {i} ok ({time.time() - t0:.1f}s)", flush=True)
        real_sync()
    except Exception as e:  # CUDA errors arrive as torch.AcceleratorError etc.
        print(f"[repro] RESULT: CRASHED at iter {last + 1} "
              f"after {time.time() - t0:.1f}s: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
    print(f"[repro] RESULT: SURVIVED {args.iters} iters "
          f"({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
