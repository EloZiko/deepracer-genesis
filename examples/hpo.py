"""HPO example: an Optuna study over PPO hyperparameters (the last mile).

Trials run in-process: sample a config from a declarative ``{name: Space}`` map
(the SAME Space types domain randomization uses — Part H), ``run()`` it, and
report the periodic deterministic evals to the pruner so Hyperband can kill bad
trials mid-training. Not a single registered experiment (it is a study), so it
runs as a script:

    uv run examples/hpo.py
"""

from __future__ import annotations

import os

import optuna

from deepracer_genesis.experiment import PPO, FeatureEnvironment, VectorPolicy, run
from deepracer_genesis.randomization.spaces import FloatRange, IntRange

STEPS = int(os.environ.get("HPO_STEPS", 5_000_000))
EVAL_EVERY = int(os.environ.get("HPO_EVAL_EVERY", 500_000))
N_TRIALS = int(os.environ.get("HPO_TRIALS", 20))
METRIC = "completion_rate"

# declare the search space once; each site chooses its verb (suggest vs sample)
SEARCH_SPACE = {
    "lr": FloatRange(1e-4, 1e-2, log=True),
    "entropy_coef": FloatRange(1e-3, 3e-2, log=True),
    "epochs": IntRange(3, 8),
    "clip": FloatRange(0.1, 0.3),
}


def objective(trial: optuna.Trial) -> float:
    p = {name: space.suggest(trial, name) for name, space in SEARCH_SPACE.items()}
    spec = (
        FeatureEnvironment(num_envs=1024)
        >> VectorPolicy(keys=("state",))
        >> PPO(lr=p["lr"], entropy_coef=p["entropy_coef"],
               epochs=p["epochs"], clip=p["clip"])
    ).build(seed=0, total_env_steps=STEPS, eval_every_steps=EVAL_EVERY,
            ablation_group="hpo")

    def report(frames: int, metrics: dict) -> None:
        trial.report(metrics[METRIC], frames)
        if trial.should_prune():
            raise optuna.TrialPruned()

    record = run(spec, root="runs/hpo", on_eval=report)
    return float(record.metrics[METRIC])


if __name__ == "__main__":
    os.makedirs("runs/hpo", exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=EVAL_EVERY, max_resource=STEPS, reduction_factor=3),
        study_name="feature_ppo",
        storage="sqlite:///runs/hpo/study.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=N_TRIALS)
    print("best:", study.best_value, study.best_params)
