"""Tests for the 3D box-scene generator (Stage 1)."""

from lines.datagen.box_scene import sample_box_scene
from lines.datagen.render import flatten_primitive
from lines.datagen.sampler2d import Canvas

CANVAS = Canvas(128, 128)


def _within(prim, canvas):
    return all(0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height
               for x, y in flatten_primitive(prim, 96))


def test_deterministic_for_seed():
    assert sample_box_scene(7, CANVAS).approx_equal(sample_box_scene(7, CANVAS))


def test_all_lines_valid_and_in_canvas():
    for seed in range(200):
        pset = sample_box_scene(seed, CANVAS)
        assert len(pset.primitives) >= 1
        for prim in pset.primitives:
            assert prim.type == "line"
            assert prim.is_valid() and _within(prim, CANVAS)


def test_edge_count_within_box_visibility_range():
    for seed in range(200):
        n = len(sample_box_scene(seed, CANVAS).primitives)
        assert 3 <= n <= 9   # a convex box shows between ~4 and 9 visible edges


def test_count_fits_query_budget_of_16():
    for seed in range(200):
        assert len(sample_box_scene(seed, CANVAS).primitives) <= 16
