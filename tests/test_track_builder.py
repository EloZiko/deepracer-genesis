"""Geometry invariants for the procedural track builder."""

import numpy as np
import pytest

from deepracer_genesis.tools.track_builder import (
    build_route,
    route_from_waypoints,
)

# A 3x1 rectangle listed clockwise and counterclockwise (same shape/path).
_RECT_CW = [(0.0, 0.0), (0.0, 1.0), (3.0, 1.0), (3.0, 0.0)]
_RECT_CCW = [(0.0, 0.0), (3.0, 0.0), (3.0, 1.0), (0.0, 1.0)]


def _poly_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


@pytest.mark.parametrize("waypoints", [_RECT_CW, _RECT_CCW])
def test_inner_border_is_the_interior_regardless_of_winding(waypoints):
    """`inner` (cols 2:4) must enclose less area than `outer` for either winding.

    Regression: clockwise waypoints previously swapped the borders, hiding the
    road under the infield fill.
    """
    route = route_from_waypoints(waypoints, width=0.5)
    inner, outer = route[:, 2:4], route[:, 4:6]
    assert _poly_area(inner) < _poly_area(outer)


@pytest.mark.parametrize(
    "builder",
    [lambda w: route_from_waypoints(w, width=0.5),
     lambda w: build_route(w, half_width=0.25)],
)
def test_both_windings_yield_the_same_border_areas(builder):
    """CW and CCW inputs of one shape produce the same ribbon geometry."""
    cw, ccw = builder(_RECT_CW), builder(_RECT_CCW)
    assert _poly_area(cw[:, 2:4]) == pytest.approx(_poly_area(ccw[:, 2:4]), rel=0.02)
    assert _poly_area(cw[:, 4:6]) == pytest.approx(_poly_area(ccw[:, 4:6]), rel=0.02)
