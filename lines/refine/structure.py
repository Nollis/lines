"""Structural post-processing of set-prediction output (plan Unit A1).

The set predictor emits independent primitives with no junction structure, so
connected drawings (boxes) come out as tangles: near-duplicate lines and corners
that don't quite meet. This is a cheap, no-retraining cleanup:

* ``merge_junctions`` -- cluster near-coincident line endpoints and snap each
  cluster to its centroid, so edges that nearly meet share an exact corner.
* ``dedup_primitives`` -- drop near-duplicate lines (same endpoints within a
  tolerance, in either orientation).

It is the YAGNI hedge for Gate 2: if this recovers enough structure, the
architecture rewrite can be deferred. Non-line primitives pass through
untouched (junction structure is a line-drawing concern here).
"""

from __future__ import annotations

import math
from typing import List

from lines.primitives import Line, PrimitiveSet


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def merge_junctions(pset: PrimitiveSet, tol: float = 3.0) -> PrimitiveSet:
    """Snap near-coincident line endpoints to shared corners (greedy clustering)."""
    # gather endpoints of line primitives as (prim_index, end_name, point)
    refs = []
    for i, prim in enumerate(pset.primitives):
        if isinstance(prim, Line):
            refs.append((i, "p1", prim.p1))
            refs.append((i, "p2", prim.p2))

    clusters: List[List[int]] = []          # each: list of ref indices
    centroids: List[tuple] = []
    for ri, (_, _, pt) in enumerate(refs):
        placed = False
        for ci, c in enumerate(centroids):
            if _dist(pt, c) <= tol:
                clusters[ci].append(ri)
                pts = [refs[k][2] for k in clusters[ci]]
                centroids[ci] = (sum(p[0] for p in pts) / len(pts),
                                 sum(p[1] for p in pts) / len(pts))
                placed = True
                break
        if not placed:
            clusters.append([ri])
            centroids.append((pt[0], pt[1]))

    # map each ref to its cluster centroid
    snapped = {}
    for ci, cluster in enumerate(clusters):
        for ri in cluster:
            snapped[ri] = centroids[ci]

    out = list(pset.primitives)
    for ri, (pi, end, _) in enumerate(refs):
        new_pt = snapped[ri]
        prim = out[pi]
        if end == "p1":
            out[pi] = Line(p1=new_pt, p2=prim.p2)
        else:
            out[pi] = Line(p1=prim.p1, p2=new_pt)
    return PrimitiveSet(out)


def _same_line(a: Line, b: Line, tol: float) -> bool:
    forward = _dist(a.p1, b.p1) <= tol and _dist(a.p2, b.p2) <= tol
    reverse = _dist(a.p1, b.p2) <= tol and _dist(a.p2, b.p1) <= tol
    return forward or reverse


def dedup_primitives(pset: PrimitiveSet, tol: float = 3.0) -> PrimitiveSet:
    """Drop near-duplicate lines (either orientation). Non-lines pass through."""
    kept = []
    for prim in pset.primitives:
        if isinstance(prim, Line):
            if any(isinstance(k, Line) and _same_line(prim, k, tol) for k in kept):
                continue
        kept.append(prim)
    return PrimitiveSet(kept)


def structure_postprocess(pset: PrimitiveSet, canvas, junction_tol: float = 3.0,
                          dup_tol: float = 3.0) -> PrimitiveSet:
    """Merge junctions, drop duplicates, and discard degenerate primitives."""
    merged = merge_junctions(pset, tol=junction_tol)
    deduped = dedup_primitives(merged, tol=dup_tol)
    valid = PrimitiveSet([p for p in deduped.primitives if p.is_valid()])
    return valid
