"""Second held-out probe (loop iteration 2).

Once the technical AND first held-out families are folded into training, neither
is out-of-distribution any more. This module provides a THIRD, disjoint set of
structural relationships to keep the generalization measurement honest:

* ``grid``      -- a lattice of crossed horizontal + vertical lines
* ``wheel``     -- a circle with radial spokes (circle + radial *together*; no
                   training family combines a rim circle with spokes)
* ``arc_chain`` -- connected scalloped arcs along a baseline

These differ from every training family (technical: concentric circles, bolt,
rectangle, crosshair, tangent, parallel, fillet; held-out-1: nested squares,
radial burst, circle chain). Probe-only -- never used for training.
"""

from __future__ import annotations

import math

import numpy as np

from lines.datagen.technical_layout import _all_within, _center
from lines.primitives import Arc, Circle, Line, PrimitiveSet

_MARGIN = 8.0


def _grid(rng, canvas):
    nh = int(rng.integers(2, 4))
    nv = int(rng.integers(2, 4))
    if nh + nv > 8:
        nv = 8 - nh
    bw = float(rng.uniform(40, min(canvas.width, canvas.height) * 0.7))
    bh = float(rng.uniform(40, min(canvas.width, canvas.height) * 0.7))
    c = _center(rng, canvas, max(bw, bh) / 2 + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    x0, x1 = cx - bw / 2, cx + bw / 2
    y0, y1 = cy - bh / 2, cy + bh / 2
    prims = []
    for i in range(nh):
        y = y0 + (y1 - y0) * (i + 0.5) / nh
        prims.append(Line(p1=(x0, y), p2=(x1, y)))
    for j in range(nv):
        x = x0 + (x1 - x0) * (j + 0.5) / nv
        prims.append(Line(p1=(x, y0), p2=(x, y1)))
    return prims


def _wheel(rng, canvas):
    R = float(rng.uniform(18, 38))
    c = _center(rng, canvas, R + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    n = int(rng.integers(3, 6))
    phase = float(rng.uniform(0, 2 * math.pi))
    prims = [Circle(center=(cx, cy), radius=R)]
    for i in range(n):
        a = phase + 2 * math.pi * i / n
        prims.append(Line(p1=(cx, cy), p2=(cx + R * math.cos(a), cy + R * math.sin(a))))
    return prims[:8]


def _arc_chain(rng, canvas):
    n = int(rng.integers(2, 4))
    seg = float(rng.uniform(18, 34))
    bulge = float(rng.uniform(0.4, 0.9))
    ang = float(rng.uniform(0, math.pi))
    dx, dy = math.cos(ang), math.sin(ang)
    span = seg * n
    sag = seg / 2 * bulge
    c = _center(rng, canvas, span / 2 + sag + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    start = (cx - dx * span / 2, cy - dy * span / 2)
    prims = []
    for i in range(n):
        p1 = (start[0] + dx * seg * i, start[1] + dy * seg * i)
        p2 = (start[0] + dx * seg * (i + 1), start[1] + dy * seg * (i + 1))
        sign = 1.0 if i % 2 == 0 else -1.0   # alternate -> wave
        prims.append(Arc(p1=p1, p2=p2, bulge=sign * bulge))
    return prims


TEMPLATES = {"grid": _grid, "wheel": _wheel, "arc_chain": _arc_chain}
_NAMES = list(TEMPLATES.keys())


def sample_heldout2_set(seed: int, canvas, _force: str | None = None,
                        _record: set | None = None) -> PrimitiveSet:
    rng = np.random.default_rng(seed)
    order = [_force] if _force else list(_NAMES)
    if not _force:
        rng.shuffle(order)
    for name in order:
        for _ in range(8):
            prims = TEMPLATES[name](rng, canvas)
            if prims and _all_within(prims, canvas):
                if _record is not None:
                    _record.add(name)
                return PrimitiveSet(prims)
    if _record is not None:
        _record.add("fallback")
    r = min(canvas.width, canvas.height) * 0.18
    return PrimitiveSet([Circle(center=(canvas.width / 2, canvas.height / 2), radius=r)])
