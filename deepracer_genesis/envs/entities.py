"""The DeepRacer car as a domain object.

``Car`` wraps the Genesis URDF entity and owns every controller interaction —
resolving the steering/drive DOFs, setting the PD gains + torque limits,
measuring the wheel radius, and applying per-step control and resets. Unknown
attribute access forwards to the underlying entity (``__getattr__``), so callers
that still need raw Genesis entity methods (domain randomization, validation
scenes) keep working while the env talks to a ``Car``.
"""

from __future__ import annotations

import os

import torch

import genesis as gs

from ..physics.limits import ackermann_angles

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_URDF = os.path.join(_ASSETS, "urdf", "deepracer", "deepracer_processed.urdf")
_WHEEL_STL = os.path.join(_ASSETS, "meshes", "deepracer", "left_rear_wheel.STL")

WHEEL_DOFS = ["left_rear_wheel_joint", "right_rear_wheel_joint",
              "left_front_wheel_joint", "right_front_wheel_joint"]
STEER_DOFS = ["left_steering_hinge_joint", "right_steering_hinge_joint"]


class Car:
    """The car URDF entity + its steering/drive controllers."""

    def __init__(self, scene, *, merge_fixed_links: bool) -> None:
        # add the URDF to the (unbuilt) scene; DOFs/gains resolved in configure()
        self.entity = scene.add_entity(
            gs.morphs.URDF(file=_URDF, pos=(0, 0, 0.05),
                           merge_fixed_links=merge_fixed_links,
                           links_to_keep=["camera_link"]))
        self.steering_model = "ackermann"

    def __getattr__(self, name):
        # forward unknown attributes to the wrapped Genesis entity so raw entity
        # methods (get_pos, set_qpos, set_friction_ratio, n_links, idx, …) stay
        # available. Only fires for attributes Car itself does not define.
        return getattr(object.__getattribute__(self, "entity"), name)

    # ------------------------------------------------ post-build configuration
    def configure(self, env_cfg: dict, device) -> None:
        """Resolve DOF indices, set gains/torque limits, measure the wheel
        radius, and select the steering model. Call after ``scene.build()``."""
        e = self.entity
        self.wheel_dofs = [e.get_joint(n).dof_idx_local for n in WHEEL_DOFS]
        self.steer_dofs = [e.get_joint(n).dof_idx_local for n in STEER_DOFS]
        e.set_dofs_kp(torch.full((2,), env_cfg["steer_kp"], device=device), self.steer_dofs)
        e.set_dofs_kv(torch.full((2,), env_cfg["steer_kv"], device=device), self.steer_dofs)
        e.set_dofs_kv(torch.full((4,), env_cfg["wheel_kv"], device=device), self.wheel_dofs)
        # cap drive torque near the traction limit; unbounded torque with a
        # P velocity controller causes wheel-slip limit cycles at high speed
        tq = env_cfg["wheel_max_torque"]
        e.set_dofs_force_range(torch.full((4,), -tq, device=device),
                               torch.full((4,), tq, device=device), self.wheel_dofs)
        # steering geometry: "ackermann" (per-wheel inner/outer split, the real
        # car) or "parallel" (both hinges at the same angle, the legacy path)
        self.steering_model = env_cfg.get("steering_model", "ackermann")
        if self.steering_model not in ("ackermann", "parallel"):
            raise ValueError("steering_model must be 'ackermann' or 'parallel', "
                             f"got {self.steering_model!r}")
        import trimesh
        self.wheel_radius = float(trimesh.load(_WHEEL_STL).extents[2]) / 2.0

    # ------------------------------------------------------- steering geometry
    def steer_targets(self, delta: torch.Tensor) -> torch.Tensor:
        """Per-hinge angles for a commanded center angle ``(N, 1)`` → ``(N, 2)``
        of ``[left, right]``. ``parallel`` copies; ``ackermann`` splits the
        inner/outer angles (:func:`ackermann_angles`)."""
        if self.steering_model == "parallel":
            return delta.repeat(1, 2)
        left, right = ackermann_angles(delta)
        return torch.cat([left, right], dim=1)

    # ----------------------------------------------------------- per-step control
    def drive(self, steer_center: torch.Tensor, speed: torch.Tensor) -> None:
        """Command the front steering angle + rear-wheel speed (both ``(N, 1)``)."""
        self.entity.control_dofs_position(self.steer_targets(steer_center), self.steer_dofs)
        wheel_omega = (speed / self.wheel_radius).repeat(1, 4)
        self.entity.control_dofs_velocity(wheel_omega, self.wheel_dofs)

    # ------------------------------------------------------------------- reset
    def reset_pose(self, qpos: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Teleport to ``qpos`` and zero the controllers for the given envs."""
        e = self.entity
        n = len(env_ids)
        dev = qpos.device
        e.set_qpos(qpos, envs_idx=env_ids)
        e.zero_all_dofs_velocity(envs_idx=env_ids)
        e.control_dofs_position(torch.zeros(n, 2, device=dev), self.steer_dofs, envs_idx=env_ids)
        e.control_dofs_velocity(torch.zeros(n, 4, device=dev), self.wheel_dofs, envs_idx=env_ids)

    # -------------------------------------------------------------- kinematics
    def kinematics(self):
        """``(pos, quat, vel, ang)`` of the base link."""
        e = self.entity
        return e.get_pos(), e.get_quat(), e.get_vel(), e.get_ang()
