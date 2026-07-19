"""Genesis scene assembly for the DeepRacer env.

``build_scene(env, ...)`` creates the ``gs.Scene`` (renderer chosen by the
env's :class:`~deepracer_genesis.envs.renderers.Renderer` strategy), adds the
ground/field plane, the :class:`~deepracer_genesis.envs.entities.Car`, and the
track morph(s), lets the renderer add its cameras/lights/sensors, then calls
``scene.build()``. Post-build steps (car control config, camera attach) happen
back in the env's ``__init__`` — see :mod:`deepracer_genesis.envs.base_env`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs

from .entities import Car
from .renderers import NyxRenderer, make_renderer

if TYPE_CHECKING:
    from .base_env import DeepRacerEnv


def build_scene(env: "DeepRacerEnv", env_cfg: dict, show_viewer: bool) -> None:
    """Create the scene, add the plane/car/track, and build it. Populates
    ``env.renderer``, ``env.scene``, ``env.plane``, ``env.car``,
    ``env.track_entity``."""
    env.renderer = make_renderer(env_cfg)

    env.scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=env_cfg["dt"], substeps=1),
        rigid_options=gs.options.RigidOptions(
            dt=env_cfg["dt"],
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            # per-env dofs/links properties (kp/kv/armature/mass/COM DR) need
            # batched physics info, per the Genesis DR guide
            batch_dofs_info=bool(env_cfg.get("randomize", False)),
            batch_links_info=bool(env_cfg.get("randomize", False)),
        ),
        vis_options=gs.options.VisOptions(
            shadow=False,
            ambient_light=(0.35, 0.35, 0.35),
            background_color=tuple(env_cfg.get("background_color", (0.55, 0.72, 0.9))),
        ),
        renderer=env.renderer.scene_renderer(),
        show_viewer=show_viewer,
    )

    # green ground doubles as the field: some DAE ground materials render
    # transparent under Madrona, and this is what shows through. Must be a
    # surface color — Madrona does not sample ImageTexture on primitives.
    fc = env_cfg.get("field_color", (0.30, 0.48, 0.32))
    env.plane = env.scene.add_entity(
        gs.morphs.Plane(pos=(0, 0, -0.001)),
        surface=gs.surfaces.Rough(color=(*fc, 1.0)),
    )

    env.car = Car(env.scene, merge_fixed_links=env.renderer.merge_fixed_links)

    # Nyx cannot read DAE; use the OBJ conversions (same geometry/textures)
    nyx = isinstance(env.renderer, NyxRenderer)
    mesh_paths = env.track.obj_paths if nyx else env.track.mesh_paths
    if nyx and len(mesh_paths) > 1:
        raise NotImplementedError("heterogeneous tracks are not supported with the Nyx renderer")
    if env.renderer.has_camera and len(mesh_paths) > 1:
        raise NotImplementedError(
            "heterogeneous multi-track CAMERA training is unsound under the "
            "batch renderer: genesis 1.2.1 never feeds vgeom.active_envs_mask "
            "to it, so every env renders ALL track variants superimposed "
            "(z-fighting). Feature-mode multi-track (no rendering) is fine.")
    track_morphs = [gs.morphs.Mesh(file=p, fixed=True, collision=False)
                    for p in mesh_paths]
    # a list of morphs makes the entity heterogeneous: each parallel env
    # simulates (and renders) one geometry variant
    env.track_entity = env.scene.add_entity(
        track_morphs if len(track_morphs) > 1 else track_morphs[0])

    # renderer adds its cameras / lights / sensors, then we build the scene
    env.renderer.build(env, env_cfg)
    env.scene.build(n_envs=env.num_envs)
