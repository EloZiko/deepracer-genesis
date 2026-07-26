"""Safe-RL example: representation transfer + a constrained (CMDP) objective.

Blends the most stages: a camera env that ALSO emits a cost signal, DR, a
frozen-CNN encoder mapping pixels to a feature vector, a downstream vector
policy, and PPO-Lagrangian (inferred automatically from the cost signal). Shown
in the class-authoring style so per-run knobs are overridable attributes.
"""

from deepracer_genesis.experiment import (
    DomainRandomizationActions,
    DomainRandomizationCamera,
    Experiment,
    FrozenCNNToFeatureVector,
    SafeRLCameraEnvironment,
    VectorPolicy,
)


class SafeTransfer(Experiment):
    """Safe-RL camera env -> frozen-CNN features -> vector policy, PPO-Lagrangian.

    Attributes:
        render: camera renderer ("madrona" or "nyx").
        budget: per-episode cost budget for the safety constraint.
        ckpt: a pretrained camera checkpoint the frozen CNN loads from.
    """

    render = "madrona"
    budget = 25.0
    ckpt = "runs/examples/camera_madrona_dr-0/best.pt"   # a camera checkpoint
    seed = 0
    total_env_steps = 10_000_000
    eval_every_steps = 2_000_000
    ablation_group = "examples"

    def spec(self):
        return (
            SafeRLCameraEnvironment(render=self.render,
                                    cost="offtrack_or_overspeed", budget=self.budget)
            >> DomainRandomizationCamera(brightness=(0.7, 1.3))
            >> FrozenCNNToFeatureVector(checkpoint=self.ckpt, output_dim=256)
            >> VectorPolicy(keys=("encoded", "state"))
            >> DomainRandomizationActions(steer_noise=0.02)
        ).build(seed=self.seed, total_env_steps=self.total_env_steps,
                eval_every_steps=self.eval_every_steps,
                ablation_group=self.ablation_group,
                variant=f"safe_transfer_{self.render}")


class SafeTransferTight(SafeTransfer):
    """Same transfer setup under a tighter cost budget."""
    budget = 10.0


if __name__ == "__main__":
    SafeTransfer().run()
