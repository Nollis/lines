"""Independent 'technical drawing' content generator (sim-to-real probe v2).

The training sampler (:mod:`lines.datagen.sampler2d`) produces random,
independent shape-soup. Real technical illustrations are *arranged*: concentric
circles, bolt-hole patterns, rectangles, crosshairs, tangent circles, filleted
corners. This module generates those structured layouts with exact primitive
ground truth, to probe whether the model generalizes to content distributions
it never trained on.

NEVER use this for training -- it exists only to build OOD probe splits.
"""

from __future__ import annotations

import math

import numpy as np

from lines.datagen.render import flatten_primitive
from lines.primitives import Arc, Circle, Line, PrimitiveSet

_MARGIN = 8.0


def _within(prim, canvas) -> bool:
    return all(0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height
               for x, y in flatten_primitive(prim, 96))


def _all_within(prims, canvas) -> bool:
    return all(_within(p, canvas) for p in prims) and all(p.is_valid() for p in prims)


def _center(rng, canvas, inset):
    """Random center keeping ``inset`` clear of every edge, or None if the
    shape is too large to fit."""
    if 2 * inset >= min(canvas.width, canvas.height):
        return None
    return (float(rng.uniform(inset, canvas.width - inset)),
            float(rng.uniform(inset, canvas.height - inset)))


# --- templates (each returns a list of primitives, or None to reject) ---------

def _concentric_circles(rng, canvas):
    c = _center(rng, canvas, canvas.width * 0.35)
    if c is None:
        return None
    cx, cy = c
    max_r = min(cx, cy, canvas.width - cx, canvas.height - cy) - _MARGIN
    if max_r < 16:
        return None
    n = int(rng.integers(2, 5))
    radii = sorted(rng.uniform(8, max_r, size=n))
    if len(set(np.round(radii, 1))) < n:
        return None
    return [Circle(center=(cx, cy), radius=float(r)) for r in radii]


def _bolt_circle(rng, canvas):
    c = _center(rng, canvas, canvas.width * 0.4)
    if c is None:
        return None
    cx, cy = c
    n = int(rng.integers(4, 7))
    hole_r = float(rng.uniform(4, 8))
    pitch = min(cx, cy, canvas.width - cx, canvas.height - cy) - hole_r - _MARGIN
    if pitch < 20:
        return None
    phase = float(rng.uniform(0, 2 * math.pi))
    prims = []
    for i in range(n):
        a = phase + 2 * math.pi * i / n
        prims.append(Circle(center=(cx + pitch * math.cos(a), cy + pitch * math.sin(a)),
                            radius=hole_r))
    if rng.random() < 0.5:
        prims.append(Circle(center=(cx, cy), radius=float(rng.uniform(6, hole_r + 4))))
    return prims[:8]


def _rectangle(rng, canvas):
    w = float(rng.uniform(30, canvas.width * 0.6))
    h = float(rng.uniform(20, canvas.height * 0.6))
    c = _center(rng, canvas, max(w, h) / 2 + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    ang = float(rng.uniform(0, math.pi)) if rng.random() < 0.5 else 0.0
    ca, sa = math.cos(ang), math.sin(ang)
    corners = []
    for dx, dy in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
        corners.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return [Line(p1=corners[i], p2=corners[(i + 1) % 4]) for i in range(4)]


def _crosshair(rng, canvas):
    c = _center(rng, canvas, canvas.width * 0.35)
    if c is None:
        return None
    cx, cy = c
    arm = min(cx, cy, canvas.width - cx, canvas.height - cy) - _MARGIN
    if arm < 14:
        return None
    prims = [
        Line(p1=(cx - arm, cy), p2=(cx + arm, cy)),
        Line(p1=(cx, cy - arm), p2=(cx, cy + arm)),
    ]
    if rng.random() < 0.6:
        prims.append(Circle(center=(cx, cy), radius=float(rng.uniform(8, arm * 0.7))))
    return prims


def _tangent_circles(rng, canvas):
    r1 = float(rng.uniform(10, 22))
    r2 = float(rng.uniform(10, 22))
    c = _center(rng, canvas, r1 + r2 + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    ang = float(rng.uniform(0, 2 * math.pi))
    c2 = (cx + (r1 + r2) * math.cos(ang), cy + (r1 + r2) * math.sin(ang))
    return [Circle(center=(cx, cy), radius=r1), Circle(center=c2, radius=r2)]


def _parallel_lines(rng, canvas):
    n = int(rng.integers(2, 4))
    ang = float(rng.uniform(0, math.pi))
    dx, dy = math.cos(ang), math.sin(ang)
    nx, ny = -dy, dx
    length = float(rng.uniform(30, canvas.width * 0.5))
    spacing = float(rng.uniform(8, 16))
    c = _center(rng, canvas, length / 2 + n * spacing + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    prims = []
    for i in range(n):
        off = (i - (n - 1) / 2) * spacing
        bx, by = cx + nx * off, cy + ny * off
        prims.append(Line(p1=(bx - dx * length / 2, by - dy * length / 2),
                          p2=(bx + dx * length / 2, by + dy * length / 2)))
    return prims


def _filleted_corner(rng, canvas):
    # two perpendicular arms joined by a quarter-circle fillet arc
    r = float(rng.uniform(10, 18))
    arm = float(rng.uniform(18, 32))
    c = _center(rng, canvas, arm + r + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    h_end = (cx + arm, cy)
    v_end = (cx, cy + arm)
    arc = Arc(p1=(cx + r, cy), p2=(cx, cy + r), bulge=math.tan(math.radians(90) / 4))
    return [
        Line(p1=(cx + r, cy), p2=h_end),
        Line(p1=(cx, cy + r), p2=v_end),
        arc,
    ]


TEMPLATES = {
    "concentric": _concentric_circles,
    "bolt_circle": _bolt_circle,
    "rectangle": _rectangle,
    "crosshair": _crosshair,
    "tangent": _tangent_circles,
    "parallel": _parallel_lines,
    "fillet": _filleted_corner,
}
_NAMES = list(TEMPLATES.keys())


def sample_technical_set(seed: int, canvas, _force: str | None = None,
                         _record: set | None = None) -> PrimitiveSet:
    """Sample one structured technical layout. Deterministic in ``seed``."""
    rng = np.random.default_rng(seed)
    if _force:
        order = [_force]
    else:
        order = list(_NAMES)
        rng.shuffle(order)
    for name in order:
        for _ in range(8):
            prims = TEMPLATES[name](rng, canvas)
            if prims and _all_within(prims, canvas):
                if _record is not None:
                    _record.add(name)
                return PrimitiveSet(prims)
    # robust fallback: a single centered circle always fits
    if _record is not None:
        _record.add("fallback")
    r = min(canvas.width, canvas.height) * 0.2
    return PrimitiveSet([Circle(center=(canvas.width / 2, canvas.height / 2), radius=r)])
