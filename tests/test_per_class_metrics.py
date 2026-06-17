"""Tests for per-class metric breakdown."""

import math

import pytest

from lines.eval.harness import run_predictor
from lines.datagen.sampler2d import Canvas
from lines.eval.metrics import evaluate
from lines.primitives import Arc, Circle, Line, PrimitiveSet

CANVAS = Canvas(64, 64)


def _two_class_sets():
    gt = PrimitiveSet([
        Line(p1=(5.0, 5.0), p2=(60.0, 5.0)),
        Circle(center=(32.0, 40.0), radius=10.0),
    ])
    pred = PrimitiveSet([
        Line(p1=(5.0, 5.0), p2=(60.0, 5.0)),
        Circle(center=(35.0, 40.0), radius=10.0),
    ])
    return pred, gt


def test_evaluate_returns_per_class_counts():
    pred, gt = _two_class_sets()
    m = evaluate(pred, gt, CANVAS)
    assert "per_class" in m
    pc = m["per_class"]
    assert pc["line"]["n_gt"] == 1
    assert pc["circle"]["n_gt"] == 1
    assert pc["line"]["n_matched"] == 1
    assert pc["circle"]["n_matched"] == 1


def test_perfect_line_class_scores_perfectly():
    pred, gt = _two_class_sets()
    m = evaluate(pred, gt, CANVAS)
    line = m["per_class"]["line"]
    assert line["type_accuracy"] == pytest.approx(1.0)
    assert line["geometric_error"] < 1e-6


def test_class_with_no_gt_does_not_crash():
    gt = PrimitiveSet([Line(p1=(0.0, 0.0), p2=(10.0, 10.0))])
    pred = PrimitiveSet([Line(p1=(0.0, 0.0), p2=(10.0, 10.0))])
    m = evaluate(pred, gt, CANVAS)
    arc = m["per_class"]["arc"]
    assert arc["n_gt"] == 0
    assert arc["n_matched"] == 0
    # type_accuracy is None / NaN when there's nothing to score
    assert arc["type_accuracy"] is None


def test_harness_aggregates_per_class_means(tmp_path):
    from lines.datagen.dataset import Dataset, write_dataset
    write_dataset(tmp_path, n_samples=4, seed=0, canvas=CANVAS, randomize=False)
    ds = Dataset(tmp_path)
    # oracle predictor
    rep = run_predictor(lambda image, idx=[-1]: (idx.__setitem__(0, idx[0]+1) or ds[idx[0]][1]), ds, CANVAS)
    assert "per_class" in rep
    for kind in ("line", "arc", "circle"):
        assert "n_gt" in rep["per_class"][kind]
