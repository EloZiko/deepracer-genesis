"""CLI for running an experiment class by its ``module:ClassName`` path.

There is no name registry — an experiment is a Python class, referenced
directly. Example::

    python -m deepracer_genesis.experiment examples.camera:CameraMadronaDR
    python -m deepracer_genesis.experiment my_pkg.runs:MyExperiment --seed 3

You can also just run an experiment file directly if it has a ``__main__``
(e.g. ``python examples/camera.py``).
"""

from __future__ import annotations

import argparse
import ast
import importlib


def _parse_set(pairs: list[str]) -> dict:
    """--set key=value pairs; values parsed as Python literals when possible."""
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _:
            raise SystemExit(f"--set expects key=value, got {pair!r}")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def _resolve(ref: str):
    """Import ``module:ClassName`` (or ``module.ClassName``) and return the object."""
    module_path, sep, attr = ref.partition(":")
    if not sep:
        module_path, _, attr = ref.rpartition(".")
    if not module_path or not attr:
        raise SystemExit(
            f"experiment ref must be 'module:ClassName' (got {ref!r})")
    try:
        module = importlib.import_module(module_path)
        return getattr(module, attr)
    except (ImportError, AttributeError) as e:
        raise SystemExit(f"could not resolve {ref!r}: {e}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepracer_genesis.experiment",
                                     description=__doc__)
    parser.add_argument("ref", nargs="?",
                        help="experiment as 'module:ClassName' (e.g. examples.camera:CameraNyx)")
    parser.add_argument("--report", action="store_true",
                        help="regenerate runs/report.md from stored records")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None,
                        help="override total_env_steps")
    parser.add_argument("--eval-every", type=int, default=None,
                        help="override eval_every_steps")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                        help="extra overrides (Experiment attrs or spec fields)")
    parser.add_argument("--root", default="runs")
    parser.add_argument("--video", action="store_true",
                        help="after training, record a spectator rollout video")
    parser.add_argument("--track", default=None,
                        help="with --video: evaluate on this track instead")
    args = parser.parse_args(argv)

    if args.report:
        from .report import build_report
        build_report(args.root)
        print(f"wrote {args.root}/report.md and {args.root}/report.csv")
        return 0
    if not args.ref:
        parser.error("experiment ref required (module:ClassName) — or --report")

    target = _resolve(args.ref)

    overrides = _parse_set(args.set)
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.steps is not None:
        overrides["total_env_steps"] = args.steps
    if args.eval_every is not None:
        overrides["eval_every_steps"] = args.eval_every

    from .run import run
    run(target, root=args.root, **overrides)

    if args.video:
        from .visualize import rollout_video
        path = rollout_video(target, root=args.root, track=args.track, **overrides)
        print(f"video: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
