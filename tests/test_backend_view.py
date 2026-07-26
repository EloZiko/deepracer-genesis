"""CPU backend + view config plumbing (Part M) — no sim needed."""

import warnings

import pytest

from deepracer_genesis.configs.cfgs import get_env_cfg
from deepracer_genesis.experiment import (
    AsymmetricCameraPolicy,
    CameraEnvironment,
    FeatureEnvironment,
    VectorPolicy,
)
from deepracer_genesis.experiment.spec import SpecError


def test_get_env_cfg_carries_backend_and_view():
    c = get_env_cfg(backend="cpu", view="gui")
    assert c["sim"]["backend"] == "cpu"
    assert c["sim"]["view"] == "gui"
    d = get_env_cfg()                       # defaults
    assert d["sim"]["backend"] == "gpu" and d["sim"]["view"] == "none"


def test_env_stage_routes_backend_view_to_spec():
    s = (FeatureEnvironment(num_envs=8, backend="cpu", view="gui")
         >> VectorPolicy()).build()
    assert s.env.backend == "cpu" and s.env.view == "gui"


def test_ensure_init_rejects_bad_backend():
    from deepracer_genesis._gs import ensure_init
    with pytest.raises(ValueError, match="gpu.*cpu"):
        ensure_init("tpu")


def test_camera_on_cpu_is_rejected_clearly():
    with pytest.raises(SpecError, match="rasterizer ObsRenderer"):
        (CameraEnvironment(backend="cpu")
         >> AsymmetricCameraPolicy(actor_keys=("camera",),
                                   critic_keys=("camera", "state"))).build()


def test_gui_large_batch_warns_but_builds():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        (FeatureEnvironment(num_envs=1024, view="gui") >> VectorPolicy()).build()
        assert any("interactive window" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        (FeatureEnvironment(num_envs=16, view="gui") >> VectorPolicy()).build()
        assert not any("interactive window" in str(x.message) for x in w)


def test_gui_plus_gpu_dr_warns_cpu_immune():
    """view='gui' + physics DR on GPU warns (stream race); CPU does not."""
    from deepracer_genesis.experiment import DomainRandomizationPhysics

    def dr_gui(backend):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            (FeatureEnvironment(num_envs=16, view="gui", backend=backend)
             >> DomainRandomizationPhysics()
             >> VectorPolicy(keys=("state",))).build()
            return any("CUDA stream race" in str(x.message) for x in w)

    assert dr_gui("gpu") is True
    assert dr_gui("cpu") is False
