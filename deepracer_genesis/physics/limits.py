"""Single source of truth for the DeepRacer car's physical limits.

Two tiers:

- **Immutable reference** — geometry read off the URDF and the operational
  action-space caps. These are facts about the car, not tunables; import them,
  never re-declare the literals.
- **Fixed normalization divisors** — the constant scales the feature vectors
  use (never episodic statistics).

Tunable knobs (control gains, domain-randomization ranges) live in
:mod:`deepracer_genesis.configs.cfgs`, not here.

Geometry is derived from ``assets/urdf/deepracer/deepracer_processed.urdf``:
``WHEELBASE_M`` = front steering-hinge x (0.082311) − rear-wheel x (−0.081663);
``FRONT_TRACK_M`` = 2 × steering-hinge |y| (0.079601).
"""

import math

import torch

# --- immutable geometry (URDF joint origins) -------------------------------
WHEELBASE_M: float = 0.163974      # front-to-rear axle distance L (m)
FRONT_TRACK_M: float = 0.159202    # left↔right front steering-hinge separation t (m)

# --- operational action-space caps (DeepRacer Box) -------------------------
# NOTE: this 30° cap is the *action* limit, NOT the URDF steering-hinge joint
# limit (±1 rad ≈ 57.3°, intentionally wider — the joint can physically exceed
# the commanded envelope).
MAX_STEERING_DEG: float = 30.0
MAX_STEER_RAD: float = math.radians(MAX_STEERING_DEG)
MIN_SPEED: float = 0.1              # m/s
MAX_SPEED: float = 4.0              # m/s

# --- fixed observation-normalization divisors (never episodic) -------------
YAW_RATE_NORM: float = 5.0          # rad/s
BETA_NORM: float = 0.5              # rad of sideslip ≈ heavy slide
CURVATURE_NORM: float = 2.5         # 1/m (r ≈ 0.4 m, about the tightest sane turn)
A_LAT_NORM: float = 20.0            # m/s²


def ackermann_angles(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Front-wheel steering angles for a commanded center (bicycle) angle.

    Splits a single commanded steering angle into the geometrically correct
    left/right hinge angles so the inner wheel steers more than the outer
    (true Ackermann), using the URDF wheelbase and front track. Written in the
    ``atan2`` form so it is numerically safe at ``delta = 0`` (both angles → 0,
    no division by the vanishing ``tan``).

    Args:
        delta: commanded center steering angle in radians, positive = left.
            Any shape; the geometry is applied elementwise.

    Returns:
        ``(left, right)`` steering-hinge angles, each the same shape as
        ``delta`` and in the ``STEER_DOFS`` order (left hinge, right hinge).
        On a left turn the left wheel is the inner wheel (larger magnitude);
        on a right turn the right wheel is.
    """
    tan_d = torch.tan(delta)
    half_t = 0.5 * FRONT_TRACK_M
    left = torch.atan2(WHEELBASE_M * tan_d, WHEELBASE_M - half_t * tan_d)
    right = torch.atan2(WHEELBASE_M * tan_d, WHEELBASE_M + half_t * tan_d)
    return left, right
