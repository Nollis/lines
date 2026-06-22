"""Reality-check perturbations for the 3D models.

We've trained on a very specific data distribution: orthographic projection,
fixed view-jitter, Pillow + supersampling at 1.5-3 px strokes, always auto-fit
to canvas. *Any* real input differs along multiple axes. This module produces
perturbations along each axis independently so we can measure where the model
breaks.

The five-plus-one categories:

* ``baseline``       -- control; default render, no perturbation
* ``cv2``            -- re-render same primitives via OpenCV (different
                         rasterizer, different AA)
* ``stroke_thicker`` -- line_width = 4.5 (outside training range 1.5-3)
* ``stroke_thinner`` -- line_width = 0.8 (outside training range)
* ``jpeg``           -- render baseline, then JPEG-roundtrip at q=60
* ``off_center``     -- translate the primitive set by a deterministic random
                         offset (GT primitives are moved too)
* ``small_scale``    -- scale primitives toward the canvas center by ~0.5 (GT
                         primitives are scaled too)

Each ``make_probe_sample`` call returns ``(image, GT_primitives)``. For the
pixel-only perturbations GT is unchanged; for the spatial ones it transforms
with the image so the metric scores against the right target.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from lines.datagen.probe_render import jpeg_roundtrip, render_cv2
from lines.datagen.render import flatten_primitive, render_primitives
from lines.primitives import (
    Arc, Bezier, Circle, Ellipse, Line, PrimitiveSet,
)


# canonical list of supported perturbations
PERTURBATIONS = frozenset({
    "baseline",
    "cv2",
    "stroke_thicker",
    "stroke_thinner",
    "jpeg",
    "off_center",
    "small_scale",
})


def make_probe_sample(primitives: PrimitiveSet, canvas, perturbation: str,
                      seed: int) -> Tuple[np.ndarray, PrimitiveSet]:
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation: {perturbation!r}; valid: "
                         f"{sorted(PERTURBATIONS)}")

    if perturbation == "baseline":
        return _render_default(primitives, canvas), primitives

    if perturbation == "cv2":
        img = render_cv2(primitives, canvas.width, canvas.height, line_width=2)
        return img, primitives

    if perturbation == "stroke_thicker":
        img = render_primitives(primitives, canvas.width, canvas.height,
                                line_width=4.5)
        return img, primitives

    if perturbation == "stroke_thinner":
        img = render_primitives(primitives, canvas.width, canvas.height,
                                line_width=0.8)
        return img, primitives

    if perturbation == "jpeg":
        img = _render_default(primitives, canvas)
        return jpeg_roundtrip(img, quality=60), primitives

    if perturbation == "off_center":
        transformed = _apply_off_center(primitives, canvas, seed)
        return _render_default(transformed, canvas), transformed

    # small_scale
    transformed = _apply_small_scale(primitives, canvas, seed)
    return _render_default(transformed, canvas), transformed


# --- helpers ----------------------------------------------------------------

def _render_default(pset: PrimitiveSet, canvas) -> np.ndarray:
    return render_primitives(pset, canvas.width, canvas.height, line_width=2.0)


def _bbox(pset: PrimitiveSet) -> Tuple[float, float, float, float]:
    pts = [p for prim in pset.primitives for p in flatten_primitive(prim, 64)]
    arr = np.asarray(pts)
    return (float(arr[:, 0].min()), float(arr[:, 1].min()),
            float(arr[:, 0].max()), float(arr[:, 1].max()))


def _translate(prim, dx: float, dy: float):
    if isinstance(prim, Line):
        return Line(p1=(prim.p1[0] + dx, prim.p1[1] + dy),
                    p2=(prim.p2[0] + dx, prim.p2[1] + dy))
    if isinstance(prim, Arc):
        return Arc(p1=(prim.p1[0] + dx, prim.p1[1] + dy),
                   p2=(prim.p2[0] + dx, prim.p2[1] + dy), bulge=prim.bulge)
    if isinstance(prim, Circle):
        return Circle(center=(prim.center[0] + dx, prim.center[1] + dy),
                      radius=prim.radius)
    if isinstance(prim, Ellipse):
        return Ellipse(center=(prim.center[0] + dx, prim.center[1] + dy),
                       semi_major=prim.semi_major, semi_minor=prim.semi_minor,
                       rotation=prim.rotation)
    if isinstance(prim, Bezier):
        return Bezier(control_points=tuple(
            (p[0] + dx, p[1] + dy) for p in prim.control_points))
    return prim


def _scale_about(prim, scale: float, pivot: Tuple[float, float]):
    def _sp(p):
        return (pivot[0] + (p[0] - pivot[0]) * scale,
                pivot[1] + (p[1] - pivot[1]) * scale)

    if isinstance(prim, Line):
        return Line(p1=_sp(prim.p1), p2=_sp(prim.p2))
    if isinstance(prim, Arc):
        return Arc(p1=_sp(prim.p1), p2=_sp(prim.p2), bulge=prim.bulge)
    if isinstance(prim, Circle):
        return Circle(center=_sp(prim.center), radius=prim.radius * scale)
    if isinstance(prim, Ellipse):
        return Ellipse(center=_sp(prim.center),
                       semi_major=prim.semi_major * scale,
                       semi_minor=prim.semi_minor * scale,
                       rotation=prim.rotation)
    if isinstance(prim, Bezier):
        return Bezier(control_points=tuple(_sp(p) for p in prim.control_points))
    return prim


def _apply_off_center(pset: PrimitiveSet, canvas, seed: int) -> PrimitiveSet:
    """Translate so the bbox lands at a random (but in-canvas) location."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = _bbox(pset)
    margin = 2.0
    # ranges that keep the bbox inside the canvas
    dx_lo = margin - x0
    dx_hi = (canvas.width - margin) - x1
    dy_lo = margin - y0
    dy_hi = (canvas.height - margin) - y1
    if dx_lo > dx_hi:
        dx_lo, dx_hi = dx_hi, dx_lo
    if dy_lo > dy_hi:
        dy_lo, dy_hi = dy_hi, dy_lo
    dx = float(rng.uniform(dx_lo, dx_hi))
    dy = float(rng.uniform(dy_lo, dy_hi))
    return PrimitiveSet([_translate(p, dx, dy) for p in pset.primitives])


def _apply_small_scale(pset: PrimitiveSet, canvas, seed: int,
                       scale_range=(0.35, 0.6)) -> PrimitiveSet:
    """Scale primitives toward canvas center by a random factor."""
    rng = np.random.default_rng(seed)
    scale = float(rng.uniform(*scale_range))
    pivot = (canvas.width / 2.0, canvas.height / 2.0)
    return PrimitiveSet([_scale_about(p, scale, pivot) for p in pset.primitives])
