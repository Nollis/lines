"""Tests for the strict primitive-F1 metric (plan Unit M1).

A predicted primitive counts as a TRUE match only if it is Hungarian-assigned to
a ground-truth primitive with normalized geometric error below a threshold AND
the predicted type is correct. This makes over-prediction (tangles) and missing
primitives both visible -- unlike render-IoU, which a tangle can satisfy.
"""

import math

from lines.eval.metrics import evaluate
from lines.datagen.sampler2d import Canvas
from lines.primitives import Circle, Line, PrimitiveSet

CANVAS = Canvas(64, 64)


def test_identical_sets_score_f1_one():
    gt = PrimitiveSet([Line(p1=(10.0, 10.0), p2=(50.0, 50.0)),
                       Circle(center=(30.0, 30.0), radius=10.0)])
    m = evaluate(gt, gt, CANVAS)
    assert math.isclose(m["precision"], 1.0, abs_tol=1e-9)
    assert math.isclose(m["recall"], 1.0, abs_tol=1e-9)
    assert math.isclose(m["f1"], 1.0, abs_tol=1e-9)


def test_both_empty_is_perfect_f1():
    m = evaluate(PrimitiveSet([]), PrimitiveSet([]), CANVAS)
    assert math.isclose(m["f1"], 1.0, abs_tol=1e-9)


def test_over_prediction_tanks_precision():
    gt = PrimitiveSet([Line(p1=(10.0, 10.0), p2=(50.0, 50.0))])
    # one correct line + 3 spurious extras
    pred = PrimitiveSet([
        Line(p1=(10.0, 10.0), p2=(50.0, 50.0)),
        Line(p1=(5.0, 40.0), p2=(40.0, 5.0)),
        Line(p1=(20.0, 60.0), p2=(60.0, 20.0)),
        Line(p1=(0.0, 30.0), p2=(60.0, 30.0)),
    ])
    m = evaluate(pred, gt, CANVAS)
    assert math.isclose(m["recall"], 1.0, abs_tol=1e-9)      # the one GT is found
    assert m["precision"] <= 0.30                            # 1 of 4 predictions valid
    assert m["f1"] < 0.5


def test_missing_primitives_tank_recall():
    gt = PrimitiveSet([Line(p1=(10.0, 10.0), p2=(50.0, 10.0)),
                       Line(p1=(10.0, 20.0), p2=(50.0, 20.0)),
                       Line(p1=(10.0, 30.0), p2=(50.0, 30.0)),
                       Line(p1=(10.0, 40.0), p2=(50.0, 40.0))])
    pred = PrimitiveSet([Line(p1=(10.0, 10.0), p2=(50.0, 10.0))])   # only 1 of 4
    m = evaluate(pred, gt, CANVAS)
    assert math.isclose(m["precision"], 1.0, abs_tol=1e-9)
    assert m["recall"] <= 0.30


def test_correct_position_wrong_type_is_not_a_true_match():
    # a circle and a line over the same place are NOT the same primitive
    gt = PrimitiveSet([Circle(center=(32.0, 32.0), radius=15.0)])
    pred = PrimitiveSet([Line(p1=(17.0, 32.0), p2=(47.0, 32.0))])
    m = evaluate(pred, gt, CANVAS)
    assert math.isclose(m["f1"], 0.0, abs_tol=1e-9)


def test_tangle_scores_low_f1_despite_high_render_iou():
    # the Stage-1 failure in miniature: many near-duplicate lines over one GT line
    gt = PrimitiveSet([Line(p1=(8.0, 8.0), p2=(56.0, 56.0))])
    pred = PrimitiveSet([
        Line(p1=(8.0, 8.0), p2=(56.0, 56.0)),
        Line(p1=(9.0, 7.0), p2=(57.0, 55.0)),
        Line(p1=(7.0, 9.0), p2=(55.0, 57.0)),
        Line(p1=(8.0, 10.0), p2=(56.0, 54.0)),
        Line(p1=(10.0, 8.0), p2=(54.0, 56.0)),
    ])
    m = evaluate(pred, gt, CANVAS)
    assert m["render_iou"] > 0.6      # the tangle DOES cover the ink (render is forgiving)
    assert m["f1"] < 0.4              # but F1 sees the 4 spurious lines
    assert m["render_iou"] - m["f1"] > 0.25   # the masking gap, made explicit


def test_f1_is_harmonic_mean_of_precision_and_recall():
    gt = PrimitiveSet([Line(p1=(10.0, 10.0), p2=(50.0, 10.0)),
                       Line(p1=(10.0, 20.0), p2=(50.0, 20.0))])
    pred = PrimitiveSet([Line(p1=(10.0, 10.0), p2=(50.0, 10.0)),
                         Line(p1=(5.0, 55.0), p2=(55.0, 55.0)),
                         Line(p1=(5.0, 60.0), p2=(55.0, 60.0))])
    m = evaluate(pred, gt, CANVAS)
    p, r = m["precision"], m["recall"]
    expected = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    assert math.isclose(m["f1"], expected, abs_tol=1e-9)
