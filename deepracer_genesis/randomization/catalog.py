"""The single declarative catalog of every domain-randomization knob (Part L).

Each knob maps to its :class:`~deepracer_genesis.randomization.spaces.Space`
(suggested default range), the layer it is *applied* at (definitions live in
this folder; application stays where it must — physics before stepping, visual
in the renderer, actuation/image as torchrl transforms), the ``cfg`` key it
lands in, and the signal(s) it perturbs (the Part K vocabulary). DR, HPO, and
the build-time observability check can all read this one table.

This is documentation-as-data: importing it has no effect on a run. The
suggested ``Space`` defaults are starting points, not the live config values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .spaces import FloatRange, IntRange, Space, SymRange

Layer = Literal["physics", "visual", "actuation", "image", "geometry"]


@dataclass(frozen=True)
class Knob:
    """One randomizable quantity.

    Attributes:
        name: Human-readable knob name.
        space: Suggested search/sample range (a ``Space``).
        layer: Where the knob is applied (physics/visual/actuation/image).
        cfg_key: Dotted config location the knob's value lands in.
        signals: Names of the signals this knob perturbs (Part K vocabulary).
        note: Optional clarification.
    """

    name: str
    space: Space
    layer: Layer
    cfg_key: str
    signals: tuple[str, ...] = ()
    note: str = ""


CATALOG: list[Knob] = [
    # ---- physics (applied at reset, before stepping) ----
    Knob("friction", FloatRange(0.6, 1.4), "physics", "rand.friction_range",
         ("v_forward", "lateral"), "per-link friction ratio"),
    Knob("mass_shift", SymRange(0.05), "physics", "rand.mass_shift_kg",
         ("v_forward",), "+/- kg per link"),
    Knob("com_shift", SymRange(0.01), "physics", "rand.com_shift_m",
         ("lateral", "yaw_rate"), "+/- m per link, 3 axes"),
    Knob("steer_kp_scale", FloatRange(0.8, 1.2), "physics", "rand.steer_kp_scale",
         ("heading_err",), "scales steering kp and kv"),
    Knob("wheel_kv_scale", FloatRange(0.8, 1.2), "physics", "rand.wheel_kv_scale",
         ("v_forward",)),
    Knob("armature", FloatRange(0.0, 0.01), "physics", "rand.armature_range",
         ("v_forward", "yaw_rate")),
    # ---- geometry (applied in the rulebook, feature mode only) ----
    Knob("track_width_scale", FloatRange(0.9, 1.15), "geometry", "rand.track_width_scale",
         ("off_track", "lateral", "half_width"),
         "per-episode scale on rulebook half-width; feature mode only "
         "(mesh width is fixed, so scaling it would desync a camera view)"),
    # ---- visual (applied in the renderer) ----
    Knob("world_color", FloatRange(0.0, 0.6), "visual", "vision.appearance.world_color",
         ("camera",), "per-episode YIQ remap strength"),
    Knob("camera_pitch_jitter", SymRange(2.0), "visual", "rand.camera_pitch_jitter_deg",
         ("camera",), "+/- deg mount pitch"),
    Knob("camera_pos_jitter", SymRange(0.01), "visual", "rand.camera_pos_jitter_m",
         ("camera",), "+/- m mount position"),
    Knob("pixel_noise", FloatRange(0.0, 0.05), "visual", "vision.pixel_noise",
         ("camera",), "gaussian pixel noise scale"),
    # ---- actuation (torchrl inverse-action transform) ----
    Knob("steer_noise", FloatRange(0.0, 0.05), "actuation", "action_dr.steer_noise",
         ("actions",), "gaussian steering-command noise scale"),
    Knob("speed_noise", FloatRange(0.0, 0.05), "actuation", "action_dr.speed_noise",
         ("actions",), "gaussian speed-command noise scale"),
    Knob("delay_steps", IntRange(0, 3), "actuation", "action_dr.delay_steps",
         ("actions",), "command latency in steps"),
    # ---- image (torchrl obs-space transform) ----
    Knob("brightness", FloatRange(0.7, 1.3), "image", "obs_dr.image_aug.brightness",
         ("camera",)),
    Knob("contrast", FloatRange(0.7, 1.3), "image", "obs_dr.image_aug.contrast",
         ("camera",)),
    Knob("saturation", FloatRange(0.7, 1.3), "image", "obs_dr.image_aug.saturation",
         ("camera",)),
    Knob("hue", FloatRange(0.0, 0.1), "image", "obs_dr.image_aug.hue",
         ("camera",), "IQ-plane rotation fraction"),
    Knob("blur", FloatRange(0.0, 0.5), "image", "obs_dr.image_aug.blur",
         ("camera",), "max gaussian sigma"),
]

# convenience index by name
BY_NAME: dict[str, Knob] = {k.name: k for k in CATALOG}


def by_layer(layer: Layer) -> list[Knob]:
    """Return every catalog knob applied at ``layer``."""
    return [k for k in CATALOG if k.layer == layer]
