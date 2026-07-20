"""Bake deterministic, cached track texture variants for rasterizer tooling
(not used for Madrona camera training, which lacks per-env variant dispatch).
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np

_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "deepracer_genesis",
                      "appearance")
_ROAD_RE = re.compile(r"road", re.IGNORECASE)
_LINE_RE = re.compile(r"line", re.IGNORECASE)
_ALT_SURFACE_RE = re.compile(r"_DIFF\.(png|jpe?g)$", re.IGNORECASE)


def _tint_image(img, rgb):
    """Multiply an image's RGB channels by an (r, g, b) factor triple,
    preserving the source's alpha-ness."""
    from PIL import Image
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    mode = "RGBA" if has_alpha else "RGB"
    arr = np.asarray(img.convert(mode)).astype(np.float32)
    arr[..., :3] *= np.asarray(rgb, dtype=np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode)


def generate_track_variants(mesh_path: str, n: int, *, seed: int = 0,
                            tint: tuple = (0.6, 1.4),
                            line_tint: tuple = (0.9, 1.1),
                            swap_road_materials: bool = True) -> list[str]:
    """Bake `n` appearance variants of the composite mesh at `mesh_path`.

    Returns:
        The list of variant .dae paths (cached and reproducible in the
        parameters).
    """
    from PIL import Image

    src_dir = os.path.dirname(mesh_path)
    tex_dir = os.path.join(src_dir, "textures")
    if not os.path.isdir(tex_dir):
        raise FileNotFoundError(f"no textures/ directory next to {mesh_path}")

    key_payload = json.dumps({"mesh": os.path.basename(mesh_path), "n": n,
                              "seed": seed, "tint": tint, "line_tint": line_tint,
                              "swap": swap_road_materials}, sort_keys=True)
    key = hashlib.sha1(key_payload.encode()).hexdigest()[:10]
    base = os.path.basename(mesh_path)
    root = os.path.join(_CACHE, base.rsplit(".", 1)[0], key)

    paths = [os.path.join(root, f"var_{i:02d}", base) for i in range(n)]
    if all(os.path.exists(p) for p in paths):
        return paths

    rng = np.random.default_rng(seed)
    textures = sorted(os.listdir(tex_dir))
    alternates = [t for t in textures if _ALT_SURFACE_RE.search(t)]

    dae_text = open(mesh_path, encoding="utf-8").read()
    for i, var_dae in enumerate(paths):
        var_dir = os.path.dirname(var_dae)
        var_tex = os.path.join(var_dir, "textures")
        os.makedirs(var_tex, exist_ok=True)

        body_rgb = rng.uniform(*tint, size=3)
        line_rgb = rng.uniform(*line_tint, size=3)
        swap_to = (rng.choice(alternates)
                   if swap_road_materials and alternates and rng.random() < 0.75
                   else None)

        # texture filenames must be UNIQUE per variant: genesis's mesh
        # preprocessing cache dedups byte-identical meshes, so identical DAE
        # copies would all resolve to the first variant's textures
        var_text = dae_text
        for name in textures:
            if _ALT_SURFACE_RE.search(name):
                continue                     # source material, not referenced
            var_name = f"v{i:02d}_{name}"
            var_text = var_text.replace(f"textures/{name}",
                                        f"textures/{var_name}")
            img = Image.open(os.path.join(tex_dir, name))
            if _ROAD_RE.search(name) and swap_to:
                # RECOLOR, don't replace: the road mesh's UVs are an atlas —
                # different track segments sample different regions of the
                # image, so any spatially-varying replacement turns one road
                # into a patchwork that changes as the car drives. The
                # original road texture is spatially uniform (that's why the
                # atlas is invisible); keep its structure and alpha, and move
                # only its color to the alternate material's mean palette.
                alt = np.asarray(Image.open(os.path.join(tex_dir, swap_to))
                                 .convert("RGB"), dtype=np.float32)
                palette = alt.reshape(-1, 3).mean(axis=0)
                mode = "RGBA" if img.mode == "RGBA" else "RGB"
                base = np.asarray(img.convert(mode)).astype(np.float32)
                gray = base[..., :3].mean(axis=-1, keepdims=True)
                gray /= max(float(gray.mean()), 1e-3)     # unit-mean structure
                base[..., :3] = (gray * palette).clip(0, 255)
                img = Image.fromarray(base.astype(np.uint8), mode)
            rgb = line_rgb if _LINE_RE.search(name) else body_rgb
            out = _tint_image(img, rgb)
            if name.lower().endswith((".jpg", ".jpeg")):
                out = out.convert("RGB")     # JPEG has no alpha channel
            out.save(os.path.join(var_tex, var_name))
        with open(var_dae, "w", encoding="utf-8") as f:
            f.write(var_text)
    return paths


def generate_field_planes(n: int, *, seed: int = 0, size_m: float = 60.0,
                          base_color: tuple = (0.30, 0.48, 0.32),
                          tint: tuple = (0.5, 1.5)) -> list[str]:
    """Bake `n` ground-plane OBJ quads with per-variant MTL diffuse colors.

    Returns the list of variant .obj paths (cached).
    """
    rng = np.random.default_rng(seed ^ 0x5EED)
    key = hashlib.sha1(json.dumps([n, seed, size_m, base_color, tint],
                                  sort_keys=True).encode()).hexdigest()[:10]
    root = os.path.join(_CACHE, "field_planes", key)
    paths = [os.path.join(root, f"field_{i:02d}.obj") for i in range(n)]
    if all(os.path.exists(p) for p in paths):
        return paths

    os.makedirs(root, exist_ok=True)
    s = size_m / 2
    for i, p in enumerate(paths):
        r, g, b = (np.asarray(base_color) * rng.uniform(*tint, size=3)).clip(0, 1)
        mtl = os.path.basename(p).replace(".obj", ".mtl")
        with open(os.path.join(root, mtl), "w") as f:
            f.write(f"newmtl field\nKd {r:.4f} {g:.4f} {b:.4f}\nKa 0 0 0\nKs 0 0 0\n")
        with open(p, "w") as f:
            f.write(f"mtllib {mtl}\n"
                    f"v -{s} -{s} 0\nv {s} -{s} 0\nv {s} {s} 0\nv -{s} {s} 0\n"
                    "vn 0 0 1\nvn 0 0 1\nvn 0 0 1\nvn 0 0 1\n"
                    "usemtl field\nf 1//1 2//2 3//3\nf 1//1 3//3 4//4\n")
    return paths
