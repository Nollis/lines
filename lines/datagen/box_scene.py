"""3D box scenes -> 2D line primitives (3D roadmap, Stage 1).

A random box in a random orientation, orthographically projected to its visible
edges, fitted to the canvas. Every primitive is a `Line` (a box has only
straight edges), so the existing schema, metric, and refinement are reused
unchanged -- only the content is new. Ground truth is exact (the projected
endpoints of each visible edge).
"""

from __future__ import annotations

import numpy as np

from lines.datagen.projection import (
    Camera, box_mesh, fit_segments_to_canvas, project_visible_edges,
    random_rotation, rotate_mesh,
)
from lines.primitives import Line, PrimitiveSet

_MIN_EDGE_PX = 4.0   # drop near-end-on edges that project to a tiny segment


def sample_box_scene(seed: int, canvas, margin: float = 14.0,
                     view_jitter: float = 0.35,
                     randomize_framing: bool = False) -> PrimitiveSet:
    """``randomize_framing=True`` jitters fit-to-canvas margin + offset so the
    trained model sees translation + scale variation (the reality-probe fix).
    Off by default to preserve existing behavior.
    """
    rng = np.random.default_rng(seed)
    dims = rng.uniform(1.0, 3.0, size=3)
    mesh = rotate_mesh(box_mesh(*dims), random_rotation(rng))
    direction = (-float(rng.uniform(-view_jitter, view_jitter)),
                 -float(rng.uniform(-view_jitter, view_jitter)),
                 -1.0)
    cam = Camera.looking_from(direction=direction)
    segs = fit_segments_to_canvas(project_visible_edges(mesh, cam), canvas,
                                   margin, randomize=randomize_framing, rng=rng)

    lines = []
    for p1, p2 in segs:
        line = Line(p1=(float(p1[0]), float(p1[1])), p2=(float(p2[0]), float(p2[1])))
        if line.is_valid() and np.hypot(p2[0] - p1[0], p2[1] - p1[1]) >= _MIN_EDGE_PX:
            lines.append(line)
    if not lines:   # extreme degenerate view -> retry deterministically
        return sample_box_scene(seed + 1, canvas, margin, view_jitter,
                                randomize_framing=randomize_framing)
    return PrimitiveSet(lines)
