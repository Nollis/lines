"""Tests for the synthetic 2D primitive sampler (Unit 2)."""

import math

import numpy as np

from lines.datagen.render import flatten_primitive
from lines.datagen.sampler2d import (
    Canvas,
    sample_arc,
    sample_circle,
    sample_line,
    sample_primitive_set,
)

CANVAS = Canvas(128, 128)


def _within(prim, canvas):
    return all(0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height
               for x, y in flatten_primitive(prim, 96))


# --- per-type samplers --------------------------------------------------------

def test_sample_circle_is_valid_and_inside_canvas():
    rng = np.random.default_rng(0)
    for _ in range(50):
        c = sample_circle(rng, CANVAS)
        assert c.is_valid()
        assert _within(c, CANVAS)


def test_sample_line_is_valid_and_inside_canvas():
    rng = np.random.default_rng(1)
    for _ in range(50):
        ln = sample_line(rng, CANVAS)
        assert ln.is_valid()
        assert _within(ln, CANVAS)


def test_sample_arc_sweep_stays_within_cap():
    rng = np.random.default_rng(2)
    cap = math.radians(270)
    for _ in range(100):
        arc = sample_arc(rng, CANVAS, max_sweep_deg=270)
        assert arc.is_valid()
        _c, _r, start, end = arc.to_center_params()
        assert abs(end - start) <= cap + 1e-9


# --- top-level set sampler ----------------------------------------------------

def test_sample_primitive_set_is_deterministic_for_a_seed():
    a = sample_primitive_set(42, canvas=CANVAS)
    b = sample_primitive_set(42, canvas=CANVAS)
    assert a.approx_equal(b)


def test_different_seeds_generally_differ():
    a = sample_primitive_set(1, canvas=CANVAS)
    b = sample_primitive_set(2, canvas=CANVAS)
    assert not a.approx_equal(b)


def test_count_respects_bounds_across_seeds():
    for seed in range(100):
        pset = sample_primitive_set(seed, canvas=CANVAS, min_n=1, max_n=5)
        assert 1 <= len(pset.primitives) <= 5


def test_all_sampled_primitives_are_valid_and_inside_canvas():
    for seed in range(100):
        pset = sample_primitive_set(seed, canvas=CANVAS)
        for prim in pset.primitives:
            assert prim.is_valid()
            assert _within(prim, CANVAS)


def test_only_requested_types_are_emitted():
    for seed in range(50):
        pset = sample_primitive_set(seed, canvas=CANVAS, types=("circle",))
        assert all(p.type == "circle" for p in pset.primitives)
