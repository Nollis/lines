"""Tests for the second held-out layout generator (loop iteration 2)."""

from lines.datagen.heldout2_layout import sample_heldout2_set, TEMPLATES
from lines.datagen.render import flatten_primitive
from lines.datagen.sampler2d import Canvas

CANVAS = Canvas(128, 128)


def _within(prim, canvas):
    return all(0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height
               for x, y in flatten_primitive(prim, 96))


def test_deterministic_for_seed():
    assert sample_heldout2_set(5, CANVAS).approx_equal(sample_heldout2_set(5, CANVAS))


def test_valid_and_in_canvas_across_seeds():
    for seed in range(200):
        pset = sample_heldout2_set(seed, CANVAS)
        assert 1 <= len(pset.primitives) <= 8
        for prim in pset.primitives:
            assert prim.is_valid() and _within(prim, CANVAS)


def test_every_template_produces_valid_output():
    for name in TEMPLATES:
        ok = False
        for seed in range(300):
            pset = sample_heldout2_set(seed, CANVAS, _force=name)
            if pset.primitives:
                ok = True
                assert all(p.is_valid() and _within(p, CANVAS) for p in pset.primitives)
                break
        assert ok, f"template {name} never produced output"


def test_disjoint_from_all_training_families():
    from lines.datagen.technical_layout import TEMPLATES as T1
    from lines.datagen.heldout_layout import TEMPLATES as T2
    assert set(TEMPLATES).isdisjoint(set(T1))
    assert set(TEMPLATES).isdisjoint(set(T2))
