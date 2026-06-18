"""Tests for structural post-processing (plan Unit A1).

Cheap, no-retraining cleanup of set-prediction output: snap near-coincident
endpoints to shared corners (junctions) and drop near-duplicate primitives.
The Gate-2 question is whether this alone recovers enough structure.
"""

import math

from lines.datagen.sampler2d import Canvas
from lines.primitives import Circle, Line, PrimitiveSet
from lines.refine.structure import merge_junctions, dedup_primitives, structure_postprocess

CANVAS = Canvas(64, 64)


def _pt_eq(a, b, tol=1e-6):
    return math.hypot(a[0] - b[0], a[1] - b[1]) <= tol


# --- junction merge -----------------------------------------------------------

def test_near_coincident_endpoints_snap_to_shared_corner():
    # two lines whose endpoints are 2px apart near (50,50)
    a = Line(p1=(10.0, 10.0), p2=(50.0, 50.0))
    b = Line(p1=(51.0, 49.0), p2=(90.0, 10.0))
    out = merge_junctions(PrimitiveSet([a, b]), tol=4.0)
    la, lb = out.primitives
    # a.p2 and b.p1 must now be the exact same point
    assert _pt_eq(la.p2, lb.p1)


def test_clean_corners_unchanged_when_below_tolerance_only_for_true_neighbors():
    # endpoints far apart must NOT be merged
    a = Line(p1=(5.0, 5.0), p2=(20.0, 20.0))
    b = Line(p1=(55.0, 55.0), p2=(40.0, 40.0))
    out = merge_junctions(PrimitiveSet([a, b]), tol=4.0)
    assert _pt_eq(out.primitives[0].p1, (5.0, 5.0))
    assert _pt_eq(out.primitives[1].p1, (55.0, 55.0))


# --- dedup --------------------------------------------------------------------

def test_near_duplicate_lines_are_merged_to_one():
    a = Line(p1=(8.0, 8.0), p2=(56.0, 56.0))
    b = Line(p1=(9.0, 7.0), p2=(57.0, 55.0))   # ~1.4px off -> duplicate
    out = dedup_primitives(PrimitiveSet([a, b]), tol=3.0)
    assert len(out.primitives) == 1


def test_reversed_duplicate_is_detected():
    a = Line(p1=(8.0, 8.0), p2=(56.0, 56.0))
    b = Line(p1=(56.0, 56.0), p2=(8.0, 8.0))   # same line, reversed
    out = dedup_primitives(PrimitiveSet([a, b]), tol=3.0)
    assert len(out.primitives) == 1


def test_distinct_lines_are_kept():
    a = Line(p1=(8.0, 8.0), p2=(56.0, 56.0))
    b = Line(p1=(8.0, 56.0), p2=(56.0, 8.0))   # the other diagonal
    out = dedup_primitives(PrimitiveSet([a, b]), tol=3.0)
    assert len(out.primitives) == 2


# --- combined / idempotence ---------------------------------------------------

def test_clean_square_is_unchanged():
    # a clean 4-line square with shared corners should survive post-processing
    sq = PrimitiveSet([
        Line(p1=(10.0, 10.0), p2=(50.0, 10.0)),
        Line(p1=(50.0, 10.0), p2=(50.0, 50.0)),
        Line(p1=(50.0, 50.0), p2=(10.0, 50.0)),
        Line(p1=(10.0, 50.0), p2=(10.0, 10.0)),
    ])
    out = structure_postprocess(sq, CANVAS)
    assert len(out.primitives) == 4
    assert out.approx_equal(sq, tol=1e-3)


def test_tangle_collapses_toward_the_true_lines():
    gt_like = PrimitiveSet([Line(p1=(8.0, 8.0), p2=(56.0, 56.0))])
    tangle = PrimitiveSet([
        Line(p1=(8.0, 8.0), p2=(56.0, 56.0)),
        Line(p1=(9.0, 7.0), p2=(57.0, 55.0)),
        Line(p1=(7.0, 9.0), p2=(55.0, 57.0)),
        Line(p1=(8.0, 10.0), p2=(56.0, 54.0)),
    ])
    out = structure_postprocess(tangle, CANVAS)
    assert len(out.primitives) < len(tangle.primitives)   # duplicates collapsed
    assert len(out.primitives) >= 1


def test_postprocess_leaves_non_line_primitives_intact():
    pset = PrimitiveSet([Circle(center=(32.0, 32.0), radius=12.0),
                         Line(p1=(5.0, 5.0), p2=(40.0, 40.0))])
    out = structure_postprocess(pset, CANVAS)
    assert any(p.type == "circle" for p in out.primitives)
    assert any(p.type == "line" for p in out.primitives)
