"""Held-out structured layouts for the post-enrichment generalization probe.

After the training generator is enriched with the :mod:`technical_layout`
families, those families are no longer out-of-distribution. To keep an honest
sim-to-real measurement we need *new* structural relationships the training set
never contains:

* ``nested_squares``  -- concentric (shared-center) rectangles
* ``radial_burst``    -- many lines sharing one endpoint (asterisk/star)
* ``circle_chain``    -- equal circles with collinear, evenly-spaced centers

These relationships differ from every training family (which has concentric
*circles*, single rectangles, 2-line crosshairs, and 2-circle tangencies, but
none of the above). Probe-only -- never used for training.
"""

from __future__ import annotations

import math

import numpy as np

from lines.datagen.technical_layout import _all_within, _center
from lines.primitives import Circle, Line, PrimitiveSet

_MARGIN = 8.0


def _nested_squares(rng, canvas):
    n = int(rng.integers(2, 4))
    big = float(rng.uniform(40, min(canvas.width, canvas.height) * 0.7))
    c = _center(rng, canvas, big / 2 + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    sizes = sorted(rng.uniform(14, big, size=n))[::-1]
    if len(set(np.round(sizes, 1))) < n:
        return None
    prims = []
    for s in sizes:
        h = s / 2
        corners = [(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h), (cx - h, cy + h)]
        for i in range(4):
            prims.append(Line(p1=corners[i], p2=corners[(i + 1) % 4]))
    return prims[:8]


def _radial_burst(rng, canvas):
    n = int(rng.integers(3, 5))
    arm = float(rng.uniform(20, 45))
    c = _center(rng, canvas, arm + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    phase = float(rng.uniform(0, math.pi))
    prims = []
    for i in range(n):
        a = phase + math.pi * i / n   # spread over a half-turn so lines are distinct
        prims.append(Line(p1=(cx, cy),
                          p2=(cx + arm * math.cos(a), cy + arm * math.sin(a))))
    return prims


def _circle_chain(rng, canvas):
    n = int(rng.integers(3, 5))
    r = float(rng.uniform(8, 14))
    gap = float(rng.uniform(0, 6))
    step = 2 * r + gap
    ang = float(rng.uniform(0, math.pi))
    dx, dy = math.cos(ang), math.sin(ang)
    span = step * (n - 1)
    c = _center(rng, canvas, span / 2 + r + _MARGIN)
    if c is None:
        return None
    cx, cy = c
    prims = []
    for i in range(n):
        off = (i - (n - 1) / 2) * step
        prims.append(Circle(center=(cx + dx * off, cy + dy * off), radius=r))
    return prims


TEMPLATES = {
    "nested_squares": _nested_squares,
    "radial_burst": _radial_burst,
    "circle_chain": _circle_chain,
}
_NAMES = list(TEMPLATES.keys())


def sample_heldout_set(seed: int, canvas, _force: str | None = None,
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
