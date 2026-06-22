"""Rasterize primitive sets to clean line-art images.

Every primitive is flattened to a polyline (reusing the arc -> center-params
conversion from :mod:`lines.primitives`) and drawn as connected segments. This
gives one uniform render path and sidesteps any drawing-library arc-angle
convention. Anti-aliasing is produced by supersampling then downsampling.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from lines.primitives import Arc, Bezier, Circle, Ellipse, Line, PrimitiveSet

Point = Tuple[float, float]


def flatten_primitive(prim, n: int = 64) -> List[Point]:
    """Return an ordered list of points approximating the primitive's stroke."""
    if isinstance(prim, Line):
        return [tuple(map(float, prim.p1)), tuple(map(float, prim.p2))]

    if isinstance(prim, Arc):
        if prim.is_straight():
            return [tuple(map(float, prim.p1)), tuple(map(float, prim.p2))]
        (cx, cy), r, start, end = prim.to_center_params()
        return [(cx + r * math.cos(a), cy + r * math.sin(a))
                for a in _linspace(start, end, n)]

    if isinstance(prim, Circle):
        cx, cy = prim.center
        r = prim.radius
        pts = [(cx + r * math.cos(a), cy + r * math.sin(a))
               for a in _linspace(0.0, 2.0 * math.pi, n)]
        pts.append(pts[0])  # close the loop
        return pts

    if isinstance(prim, Bezier):
        p0, p1, p2, p3 = (tuple(map(float, p)) for p in prim.control_points)
        return [_cubic(p0, p1, p2, p3, t) for t in _linspace(0.0, 1.0, n)]

    if isinstance(prim, Ellipse):
        # parametric ellipse: x(t) = cx + a cos t cos θ − b sin t sin θ
        #                    y(t) = cy + a cos t sin θ + b sin t cos θ
        cx, cy = prim.center
        a, b, theta = prim.semi_major, prim.semi_minor, prim.rotation
        ca, sa = math.cos(theta), math.sin(theta)
        pts = []
        for t in _linspace(0.0, 2.0 * math.pi, n):
            ct, st = math.cos(t), math.sin(t)
            pts.append((cx + a * ct * ca - b * st * sa,
                        cy + a * ct * sa + b * st * ca))
        pts.append(pts[0])   # close the loop
        return pts

    raise TypeError(f"cannot flatten primitive of type {type(prim).__name__}")


def render_primitives(
    pset: PrimitiveSet,
    width: int,
    height: int,
    line_width: float = 2.0,
    supersample: int = 4,
    samples_per_primitive: int = 96,
    bg: int = 255,
    fg: int = 0,
) -> np.ndarray:
    """Render a primitive set to a ``(height, width)`` uint8 grayscale image."""
    s = max(1, int(supersample))
    big = Image.new("L", (width * s, height * s), color=bg)
    draw = ImageDraw.Draw(big)
    pen = max(1, int(round(line_width * s)))

    for prim in pset.primitives:
        pts = [(x * s, y * s) for x, y in flatten_primitive(prim, samples_per_primitive)]
        if len(pts) >= 2:
            draw.line(pts, fill=fg, width=pen, joint="curve")

    if s > 1:
        big = big.resize((width, height), Image.LANCZOS)
    return np.asarray(big, dtype=np.uint8)


# --- helpers ------------------------------------------------------------------

def _linspace(a: float, b: float, n: int) -> List[float]:
    if n < 2:
        return [a, b]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def _cubic(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    mt = 1.0 - t
    a, b, c, d = mt * mt * mt, 3 * mt * mt * t, 3 * mt * t * t, t * t * t
    return (
        a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
        a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
    )
