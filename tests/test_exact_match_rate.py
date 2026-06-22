"""Tests for per-image exact-/near-match aggregation in the harness.

Mean F1 averages over edges and hides the *distribution* of per-image quality:
a panel that's 60% perfect boxes and 40% catastrophic ones can still average
F1=0.9. ``exact_match_rate`` and ``near_match_rate`` surface that distribution.
"""

import math
from pathlib import Path

from lines.datagen.dataset import Dataset, write_dataset
from lines.datagen.sampler2d import Canvas
from lines.eval.harness import run_predictor
from lines.primitives import Line, PrimitiveSet


CANVAS = Canvas(64, 64)


def _oracle(dataset):
    """Predictor that returns the ground truth (perfect)."""
    state = {"idx": -1}
    def pred(_image):
        state["idx"] += 1
        return dataset[state["idx"]][1]
    return pred


def _empty():
    return lambda _image: PrimitiveSet([])


def _half_oracle(dataset):
    """Half of samples are perfect; the other half are empty."""
    state = {"idx": -1}
    def pred(_image):
        state["idx"] += 1
        return dataset[state["idx"]][1] if state["idx"] % 2 == 0 else PrimitiveSet([])
    return pred


def test_oracle_predictor_has_full_exact_match_rate(tmp_path: Path):
    write_dataset(tmp_path, n_samples=8, seed=0, canvas=CANVAS, randomize=False)
    ds = Dataset(tmp_path)
    r = run_predictor(_oracle(ds), ds, CANVAS)
    assert math.isclose(r["exact_match_rate"], 1.0, abs_tol=1e-9)
    assert math.isclose(r["near_match_rate"], 1.0, abs_tol=1e-9)


def test_empty_predictor_has_zero_exact_match_rate(tmp_path: Path):
    write_dataset(tmp_path, n_samples=8, seed=0, canvas=CANVAS, randomize=False)
    ds = Dataset(tmp_path)
    r = run_predictor(_empty(), ds, CANVAS)
    assert math.isclose(r["exact_match_rate"], 0.0, abs_tol=1e-9)
    assert math.isclose(r["near_match_rate"], 0.0, abs_tol=1e-9)


def test_half_perfect_half_empty_is_about_half_exact(tmp_path: Path):
    write_dataset(tmp_path, n_samples=8, seed=0, canvas=CANVAS, randomize=False)
    ds = Dataset(tmp_path)
    r = run_predictor(_half_oracle(ds), ds, CANVAS)
    # 4 perfect + 4 empty -> exact_match_rate = 0.5 (the bimodal case mean-F1 hides)
    assert math.isclose(r["exact_match_rate"], 0.5, abs_tol=1e-9)


def test_exact_match_rate_appears_in_aggregate_keys(tmp_path: Path):
    write_dataset(tmp_path, n_samples=2, seed=0, canvas=CANVAS, randomize=False)
    ds = Dataset(tmp_path)
    r = run_predictor(_oracle(ds), ds, CANVAS)
    assert "exact_match_rate" in r
    assert "near_match_rate" in r
    assert "n_perfect" in r and "n_near" in r
