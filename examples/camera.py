"""Camera (vision) examples: Madrona batch renderer and the Nyx path tracer.

Same structure as the feature examples, but the env renders an onboard RGB
camera and the policy is asymmetric — the actor sees pixels, the critic also
sees privileged state (the vision-mode privileged critic).

Run one::

    python examples/camera.py                              # runs CameraMadronaDr
    python -m deepracer_genesis.experiment examples.camera:CameraNyx
"""

from deepracer_genesis.experiment import (
    PPO,
    AsymmetricCameraPolicy,
    CameraEnvironment,
    DomainRandomizationActions,
    DomainRandomizationCamera,
    DomainRandomizationPhysics,
    DomainRandomizationTrackAppearance,
    Evaluation,
    Experiment,
)


class CameraMadronaDr(Experiment):
    """End-to-end vision on the Madrona batch renderer with the FULL DR stack.

    Blends: GPU + camera(render="madrona") + appearance/camera/physics/action
    DR + an asymmetric camera policy (actor=pixels, critic=pixels+state).
    """

    total_env_steps = 10_000_000
    eval_every_steps = 2_000_000
    ablation_group = "examples"
    variant = "camera_madrona_dr"

    def pipeline(self):
        return (
            # 64 envs + 8 minibatches: the default 4-frame stack quadruples
            # rollout-storage obs (24 x N x 12 x 120 x 160 f32) and each PPO
            # minibatch indexes a dense copy — 128 envs / 4 minibatches OOMs
            # an 8 GB GPU next to Genesis. Also set
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (see __main__).
            CameraEnvironment(render="madrona", resolution=(160, 120), num_envs=64)
            >> DomainRandomizationTrackAppearance(strength=0.6)     # world-color remap
            >> DomainRandomizationCamera(brightness=(0.7, 1.3), hue=0.05, blur=0.3,
                                         camera_jitter=True)         # image + mount DR
            >> DomainRandomizationPhysics()                         # dynamics DR
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> DomainRandomizationActions(steer_noise=0.02, speed_noise=0.05,
                                          delay_steps=1)             # actuation DR
            >> PPO(minibatches=8)
        )


class CameraNyx(Experiment):
    """End-to-end vision on the Nyx path tracer (photorealistic, slower).

    Blends: GPU + camera(render="nyx", single track) + light camera DR + charts.
    Nyx is single-track per process, so no heterogeneous track list here.
    """

    total_env_steps = 5_000_000
    eval_every_steps = 1_000_000
    ablation_group = "examples"
    variant = "camera_nyx"

    def pipeline(self):
        return (
            CameraEnvironment(render="nyx", resolution=(160, 120), num_envs=64)
            >> DomainRandomizationCamera(brightness=(0.8, 1.2), hue=0.03)
            >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                      critic_keys=("camera", "state"))
            >> Evaluation(charts=True)                              # Part N charts
        )


if __name__ == "__main__":
    import os
    # Must be set before CUDA init: the stacked-obs copies fragment the
    # allocator on 8 GB GPUs without expandable segments.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    CameraMadronaDr().run()
