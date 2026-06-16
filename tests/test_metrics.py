"""Tests for the primitive-match metric (Unit 3).

Tests assert *properties* the metric must have, not exact weights:
perfect score on identical sets, representational equivalence resolved by the
render-based component, monotonic degradation, and false-pos/neg penalties.
"""

import math

from lines.eval.metrics import evaluate
from lines.datagen.sampler2d import Canvas
from lines.primitives import Arc, Circle, Line, PrimitiveSet

CANVAS = Canvas(100, 100)


def _circle(cx=50, cy=50, r=20):
    return PrimitiveSet([Circle(center=(cx, cy), radius=r)])


def test_identical_sets_score_perfectly():
    gt = _circle()
    m = evaluate(gt, gt, CANVAS)
    assert math.isclose(m["score"], 1.0, abs_tol=1e-6)
    assert math.isclose(m["render_iou"], 1.0, abs_tol=1e-6)
    assert math.isclose(m["type_accuracy"], 1.0, abs_tol=1e-6)
    assert m["geometric_error"] < 1e-6
    assert m["false_positives"] == 0 and m["false_negatives"] == 0


def test_straight_arc_and_line_are_scored_equivalent_by_render():
    # a bulge=0 arc and the line with the same endpoints render identically;
    # the render-based + geometric components must treat them as equivalent
    p1, p2 = (10.0, 20.0), (80.0, 70.0)
    pred = PrimitiveSet([Line(p1=p1, p2=p2)])
    gt = PrimitiveSet([Arc(p1=p1, p2=p2, bulge=0.0)])
    m = evaluate(pred, gt, CANVAS)
    assert m["render_iou"] > 0.95          # geometrically identical
    assert m["geometric_error"] < 0.02
    assert math.isclose(m["type_accuracy"], 0.0, abs_tol=1e-6)  # types still differ


def test_extra_predicted_primitive_is_penalized_as_false_positive():
    gt = _circle()
    pred = PrimitiveSet([Circle(center=(50, 50), radius=20), Line(p1=(5, 5), p2=(90, 90))])
    m = evaluate(pred, gt, CANVAS)
    assert m["false_positives"] == 1
    assert m["score"] < 1.0


def test_missing_primitive_is_penalized_as_false_negative():
    gt = PrimitiveSet([Circle(center=(30, 30), radius=12), Circle(center=(70, 70), radius=12)])
    pred = PrimitiveSet([Circle(center=(30, 30), radius=12)])
    m = evaluate(pred, gt, CANVAS)
    assert m["false_negatives"] == 1
    assert m["score"] < 1.0


def test_geometric_error_increases_monotonically_with_radius_offset():
    gt = _circle(r=20)
    errors = []
    for delta in (0.0, 2.0, 5.0, 10.0):
        pred = _circle(r=20 + delta)
        errors.append(evaluate(pred, gt, CANVAS)["geometric_error"])
    assert errors == sorted(errors)
    assert errors[0] < errors[-1]


def test_empty_prediction_against_nonempty_gt_scores_zero_without_crashing():
    m = evaluate(PrimitiveSet([]), _circle(), CANVAS)
    assert math.isclose(m["render_iou"], 0.0, abs_tol=1e-6)
    assert m["false_negatives"] == 1
    assert math.isclose(m["score"], 0.0, abs_tol=1e-6)


def test_displaced_primitive_lowers_render_iou():
    gt = _circle(cx=35, cy=35, r=12)
    pred = _circle(cx=70, cy=70, r=12)  # same shape, far away
    m = evaluate(pred, gt, CANVAS)
    assert m["render_iou"] < 0.5
