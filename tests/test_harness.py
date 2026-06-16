"""Tests for the evaluation harness (Unit 3)."""

import math

from lines.eval.harness import run_predictor
from lines.datagen.dataset import Dataset
from lines.datagen.sampler2d import Canvas
from lines.datagen.dataset import write_dataset
from lines.primitives import PrimitiveSet

CANVAS = Canvas(96, 96)


def test_oracle_predictor_scores_perfectly(tmp_path):
    write_dataset(tmp_path, n_samples=1, seed=3, canvas=CANVAS, randomize=False)
    ds = Dataset(tmp_path)
    _img, gt = ds[0]
    report = run_predictor(lambda image: gt, ds, CANVAS)  # returns the exact gt
    assert math.isclose(report["mean_score"], 1.0, abs_tol=1e-6)
    assert len(report["per_sample"]) == 1


def test_empty_predictor_reports_zero_overlap_and_full_length(tmp_path):
    write_dataset(tmp_path, n_samples=4, seed=0, canvas=CANVAS)
    ds = Dataset(tmp_path)
    report = run_predictor(lambda image: PrimitiveSet([]), ds, CANVAS)
    assert len(report["per_sample"]) == 4
    assert math.isclose(report["mean_render_iou"], 0.0, abs_tol=1e-6)
    # mean is the arithmetic mean of the per-sample scores
    manual = sum(s["score"] for s in report["per_sample"]) / 4
    assert math.isclose(report["mean_score"], manual, abs_tol=1e-9)
