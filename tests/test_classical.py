"""Tests for the classical skeletonize+fitting baseline (Unit 4).

Assertions are behavioral and tolerance-loose: a classical baseline is the bar
the model must beat, not a precision instrument. Geometric agreement is checked
via the Unit 3 metric (render-IoU) rather than brittle parameter matching.
"""

from lines.baselines.classical import ClassicalBaseline
from lines.datagen.dataset import write_dataset
from lines.datagen.dataset import Dataset
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas
from lines.eval.harness import run_predictor
from lines.eval.metrics import evaluate
from lines.primitives import Arc, Circle, Line, PrimitiveSet

CANVAS = Canvas(128, 128)
BASE = ClassicalBaseline()


def _render(pset):
    return render_primitives(pset, CANVAS.width, CANVAS.height, line_width=2.0)


def test_single_circle_is_recovered_as_one_circle():
    gt = PrimitiveSet([Circle(center=(64, 64), radius=30)])
    pred = BASE(_render(gt))
    assert len(pred.primitives) == 1
    assert pred.primitives[0].type == "circle"
    assert evaluate(pred, gt, CANVAS)["render_iou"] > 0.8


def test_single_line_is_recovered_as_one_open_primitive():
    gt = PrimitiveSet([Line(p1=(20, 30), p2=(100, 95))])
    pred = BASE(_render(gt))
    assert len(pred.primitives) == 1
    assert pred.primitives[0].type == "line"
    assert evaluate(pred, gt, CANVAS)["render_iou"] > 0.7


def test_single_arc_is_recovered_as_an_open_curve_not_a_full_circle():
    gt = PrimitiveSet([Arc(p1=(30, 100), p2=(100, 100), bulge=0.6)])
    pred = BASE(_render(gt))
    assert len(pred.primitives) == 1
    # an arc must not be mistaken for a closed circle
    assert pred.primitives[0].type in ("arc", "line")
    # classical arc fitting recovers good geometry but thin curved strokes score
    # modestly on render-IoU even when well-fit -- this is the bar the model beats
    assert evaluate(pred, gt, CANVAS)["render_iou"] > 0.6


def test_empty_image_yields_empty_set():
    blank = _render(PrimitiveSet([]))
    assert BASE(blank).primitives == []


def test_two_separated_lines_recovered_as_two_lines():
    gt = PrimitiveSet([
        Line(p1=(15, 20), p2=(15, 110)),
        Line(p1=(110, 20), p2=(110, 110)),
    ])
    pred = BASE(_render(gt))
    assert len(pred.primitives) == 2
    assert all(p.type == "line" for p in pred.primitives)


def test_baseline_runs_through_harness_and_records_a_reference_score(tmp_path):
    write_dataset(tmp_path, n_samples=8, seed=0, canvas=CANVAS, types=("line", "circle"))
    ds = Dataset(tmp_path)
    report = run_predictor(BASE, ds, CANVAS)
    assert report["n"] == 8
    assert 0.0 <= report["mean_score"] <= 1.0
    assert len(report["per_sample"]) == 8
