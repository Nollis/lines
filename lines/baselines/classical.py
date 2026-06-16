"""Classical skeletonize + primitive-fitting baseline.

Pipeline: binarize -> skeletonize -> connected components -> fit the best
primitive per component. Per component we fit a line (total least squares) and a
circle (algebraic), then classify:

* small line residual                  -> ``Line``
* curved + points wrap fully around    -> ``Circle``
* curved + a large angular gap (open)  -> ``Arc``

Loop-vs-open is decided by **angular coverage** around the fitted center, not by
skeleton neighbour counts: staircased digital strokes make neighbour counts
unreliable (interior pixels can read as degree 3, endpoints as degree 2).

This is the reference the learned model must beat; it is intentionally simple
and weak at junctions/crossings, where strokes merge into one component and fit
poorly -- exactly the regime where the model is expected to win.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize

from lines.primitives import Arc, Circle, Line, PrimitiveSet

Point = Tuple[float, float]
_8CONN = np.ones((3, 3), dtype=int)
_CLOSED_GAP = math.radians(50)  # max angular gap below which points wrap into a circle


class ClassicalBaseline:
    def __init__(self, ink_threshold: int = 128, min_branch_len: int = 8,
                 straight_residual_px: float = 1.5, line_vs_arc_ratio: float = 1.2):
        self.ink_threshold = ink_threshold
        self.min_branch_len = min_branch_len
        self.straight_residual_px = straight_residual_px
        self.line_vs_arc_ratio = line_vs_arc_ratio

    def __call__(self, image) -> PrimitiveSet:
        return self.predict(image)

    def predict(self, image) -> PrimitiveSet:
        ink = np.asarray(image) < self.ink_threshold
        if not ink.any():
            return PrimitiveSet([])

        skel = skeletonize(ink)
        labelled, n = label(skel, structure=_8CONN)

        prims = []
        for k in range(1, n + 1):
            rows, cols = np.nonzero(labelled == k)
            if len(rows) < self.min_branch_len:
                continue
            pts = np.column_stack([cols.astype(float), rows.astype(float)])  # (x, y)
            prim = self._fit(pts)
            if prim is not None and prim.is_valid():
                prims.append(prim)
        return PrimitiveSet(prims)

    def _fit(self, pts: np.ndarray):
        (cx, cy), r, circ_resid = _fit_circle(pts)
        line_resid = _line_residual(pts)
        maxdim = float(max(pts[:, 0].max(), pts[:, 1].max(), 1.0))

        straight = (
            line_resid <= self.straight_residual_px
            or line_resid <= circ_resid * self.line_vs_arc_ratio
            or r > 4.0 * maxdim
        )
        if straight:
            a, b = _terminal_pair(pts)
            return Line(p1=a, p2=b)

        if _max_angular_gap(pts, cx, cy) < _CLOSED_GAP:
            return Circle(center=(cx, cy), radius=r)
        return _arc_from_points(pts, cx, cy)


# --- geometry helpers ---------------------------------------------------------

def _terminal_pair(pts: np.ndarray) -> Tuple[Point, Point]:
    """The two farthest-apart points -- the stroke ends."""
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    return tuple(pts[i]), tuple(pts[j])


def _max_angular_gap(pts: np.ndarray, cx: float, cy: float) -> float:
    ang = np.sort(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx))
    if len(ang) < 2:
        return 2 * math.pi
    gaps = np.diff(ang)
    wrap = 2 * math.pi - (ang[-1] - ang[0])
    return float(max(gaps.max(), wrap))


def _arc_from_points(pts: np.ndarray, cx: float, cy: float):
    """Order points by angle, break at the largest gap, estimate signed sweep."""
    ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(ang)
    a_sorted = ang[order]
    gaps = np.append(np.diff(a_sorted), 2 * math.pi - (a_sorted[-1] - a_sorted[0]))
    gi = int(np.argmax(gaps))
    start_idx = 0 if gi == len(gaps) - 1 else gi + 1
    seq = np.roll(order, -start_idx)
    ordered = pts[seq]
    aa = np.unwrap(np.arctan2(ordered[:, 1] - cy, ordered[:, 0] - cx))
    sweep = float(aa[-1] - aa[0])
    if abs(sweep) < 1e-6:
        return Line(p1=tuple(ordered[0]), p2=tuple(ordered[-1]))
    return Arc(p1=tuple(ordered[0]), p2=tuple(ordered[-1]), bulge=math.tan(sweep / 4.0))


# --- fitting ------------------------------------------------------------------

def _fit_circle(pts: np.ndarray):
    """Algebraic (Kasa) circle fit. Returns (center, radius, rms residual)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = math.sqrt(max(c + cx * cx + cy * cy, 0.0))
    dists = np.hypot(x - cx, y - cy)
    resid = float(np.sqrt(np.mean((dists - r) ** 2)))
    return (float(cx), float(cy)), float(r), resid


def _line_residual(pts: np.ndarray) -> float:
    mean = pts.mean(axis=0)
    centred = pts - mean
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    return float(np.sqrt(np.mean((centred @ normal) ** 2)))
