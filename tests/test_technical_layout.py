"""Tests for the technical-layout generator (sim-to-real probe v2).

This is an INDEPENDENT content distribution from the training sampler: it makes
structured technical-drawing arrangements (concentric circles, bolt patterns,
rectangles, crosshairs, tangent circles, filleted corners) rather than random
shape-soup. Used only to build out-of-distribution probe splits with exact
ground truth -- never for training.
"""

import numpy as np

from lines.datagen.render import flatten_primitive
from lines.datagen.sampler2d import Canvas
from lines.datagen.technical_layout import sample_technical_set, TEMPLATES

CANVAS = Canvas(128, 128)


def _within(prim, canvas):
    return all(0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height
               for x, y in flatten_primitive(prim, 96))


def test_is_deterministic_for_a_seed():
    a = sample_technical_set(7, CANVAS)
    b = sample_technical_set(7, CANVAS)
    assert a.approx_equal(b)


def test_all_primitives_valid_and_in_canvas_across_seeds():
    for seed in range(200):
        pset = sample_technical_set(seed, CANVAS)
        assert len(pset.primitives) >= 1
        for prim in pset.primitives:
            assert prim.is_valid()
            assert _within(prim, CANVAS)


def test_primitive_count_stays_within_query_budget():
    # model has 8 queries; layouts must not exceed that
    for seed in range(200):
        pset = sample_technical_set(seed, CANVAS)
        assert len(pset.primitives) <= 8


def test_multiple_templates_are_exercised():
    names = set()
    for seed in range(200):
        sample_technical_set(seed, CANVAS, _record=names)
    assert len(names) >= 4   # variety, not one template repeated


def test_every_template_produces_valid_output():
    for name in TEMPLATES:
        produced = False
        for seed in range(300):
            pset = sample_technical_set(seed, CANVAS, _force=name)
            if pset.primitives:
                produced = True
                for prim in pset.primitives:
                    assert prim.is_valid() and _within(prim, CANVAS)
                break
        assert produced, f"template {name} never produced output"


def test_content_differs_from_random_sampler():
    # a structured layout should not coincide with the random sampler's output
    from lines.datagen.sampler2d import sample_primitive_set
    tech = sample_technical_set(3, CANVAS)
    rand = sample_primitive_set(3, canvas=CANVAS)
    assert not tech.approx_equal(rand)
