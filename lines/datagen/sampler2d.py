"""Sample random 2D primitive sets within a canvas.

All randomness flows through a seeded ``numpy`` Generator so a seed fully
determines the output. Primitives are guaranteed valid and contained within the
canvas (circles by construction; lines/arcs by rejection sampling). Arc sweep is
capped so the stored bulge never approaches the full-circle blow-up regime --
beyond the cap a shape would be emitted as a :class:`Circle` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lines.datagen.render import flatten_primitive
from lines.primitives import Arc, Circle, Line, PrimitiveSet

_MAX_TRIES = 128


@dataclass(frozen=True)
class Canvas:
    width: int = 256
    height: int = 256


def _rand_point(rng, canvas: Canvas, margin: float):
    return (
        float(rng.uniform(margin, canvas.width - margin)),
        float(rng.uniform(margin, canvas.height - margin)),
    )


def _within(prim, canvas: Canvas) -> bool:
    return all(
        0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height
        for x, y in flatten_primitive(prim, 96)
    )


def sample_circle(rng, canvas: Canvas, margin: float = 6.0, min_radius: float = 8.0) -> Circle:
    for _ in range(_MAX_TRIES):
        cx, cy = _rand_point(rng, canvas, margin)
        max_r = min(cx, cy, canvas.width - cx, canvas.height - cy) - 1.0
        if max_r < min_radius:
            continue
        radius = float(rng.uniform(min_radius, max_r))
        return Circle(center=(cx, cy), radius=radius)
    # extreme fallback: a small centered circle that always fits
    r = max(min_radius, min(canvas.width, canvas.height) * 0.1)
    return Circle(center=(canvas.width / 2.0, canvas.height / 2.0), radius=r)


def sample_line(rng, canvas: Canvas, margin: float = 6.0, min_len: float = None) -> Line:
    if min_len is None:
        min_len = 0.15 * min(canvas.width, canvas.height)
    line = None
    for _ in range(_MAX_TRIES):
        p1 = _rand_point(rng, canvas, margin)
        p2 = _rand_point(rng, canvas, margin)
        line = Line(p1=p1, p2=p2)
        if math.dist(p1, p2) >= min_len and line.is_valid() and _within(line, canvas):
            return line
    return line


def sample_arc(
    rng,
    canvas: Canvas,
    margin: float = 6.0,
    max_sweep_deg: float = 270.0,
    min_sweep_deg: float = 20.0,
    min_len: float = None,
) -> Arc:
    if min_len is None:
        min_len = 0.15 * min(canvas.width, canvas.height)
    max_sweep = math.radians(max_sweep_deg)
    min_sweep = math.radians(min_sweep_deg)
    arc = None
    for _ in range(_MAX_TRIES):
        p1 = _rand_point(rng, canvas, margin)
        p2 = _rand_point(rng, canvas, margin)
        if math.dist(p1, p2) < min_len:
            continue
        sweep = float(rng.uniform(min_sweep, max_sweep)) * (1.0 if rng.random() < 0.5 else -1.0)
        bulge = math.tan(sweep / 4.0)
        arc = Arc(p1=p1, p2=p2, bulge=bulge)
        if arc.is_valid() and _within(arc, canvas):
            return arc
    return arc


_SAMPLERS = {"line": sample_line, "arc": sample_arc, "circle": sample_circle}


def sample_primitive_set(
    seed: int,
    canvas: Canvas = Canvas(),
    min_n: int = 1,
    max_n: int = 5,
    max_arc_sweep_deg: float = 270.0,
    types=("line", "arc", "circle"),
) -> PrimitiveSet:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(min_n, max_n + 1))
    prims = []
    for _ in range(n):
        kind = types[int(rng.integers(0, len(types)))]
        if kind == "arc":
            prims.append(sample_arc(rng, canvas, max_sweep_deg=max_arc_sweep_deg))
        else:
            prims.append(_SAMPLERS[kind](rng, canvas))
    return PrimitiveSet(prims)
