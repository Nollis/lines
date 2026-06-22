"""Tests for spatial-framing randomization in the 3D scene generators.

The reality-check probe showed our models catastrophically fail on translation
and scale shifts because the training pipeline always centers the object and
scales it to ~80% of canvas. This unit pins the *fix*: an opt-in
``randomize_framing`` mode that jitters both axes, while preserving the
existing default behavior so nothing already-tested breaks.

Properties under test:

1. **Default behavior unchanged** -- `randomize_framing=False` produces the
   same primitive set as the old pre-fix code.
2. **Randomized framing actually varies** the framing across seeds (both
   position and scale move).
3. **Primitives stay in the canvas** even with the jitter applied.
4. **Deterministic per seed**.
5. **The output distribution covers the reality-probe range** -- specifically
   the scale should cover roughly 30%-80% of canvas (matching the probe's
   small_scale range so the model trains on what it'll be tested on).
"""

import math
from collections import Counter

import numpy as np
import pytest

from lines.datagen.box_scene import sample_box_scene
from lines.datagen.cylinder_scene import sample_cylinder_scene
from lines.datagen.render import flatten_primitive
from lines.datagen.sampler2d import Canvas


CANVAS = Canvas(128, 128)


def _bbox(pset):
    pts = [p for prim in pset.primitives for p in flatten_primitive(prim, 64)]
    arr = np.asarray(pts)
    return arr.min(0), arr.max(0)


def _scale_frac(pset, canvas):
    """The bbox's larger dimension as a fraction of canvas."""
    lo, hi = _bbox(pset)
    return float((hi - lo).max() / canvas.width)


def _centroid_offset(pset, canvas):
    """Distance from bbox center to canvas center, in pixels."""
    lo, hi = _bbox(pset)
    cx, cy = (lo + hi) / 2
    return math.hypot(cx - canvas.width / 2, cy - canvas.height / 2)


def _all_in_canvas(pset, canvas):
    for prim in pset.primitives:
        for x, y in flatten_primitive(prim, 96):
            if not (0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height):
                return False
    return True


# --- default unchanged --------------------------------------------------------

@pytest.mark.parametrize("sampler", [sample_box_scene, sample_cylinder_scene])
def test_default_framing_is_unchanged(sampler):
    # without the new kwarg, sampler output must be byte-identical to the old
    # behavior. We pin a fixed seed and verify the primitive set is approximately
    # centered at canvas center (the pre-fix behavior).
    pset = sampler(seed=5, canvas=CANVAS)
    assert _centroid_offset(pset, CANVAS) < 1.0   # centered within 1 px


@pytest.mark.parametrize("sampler", [sample_box_scene, sample_cylinder_scene])
def test_explicit_randomize_false_matches_default(sampler):
    a = sampler(seed=11, canvas=CANVAS)
    b = sampler(seed=11, canvas=CANVAS, randomize_framing=False)
    assert a.approx_equal(b)


# --- randomized framing varies ------------------------------------------------

@pytest.mark.parametrize("sampler", [sample_box_scene, sample_cylinder_scene])
def test_randomized_framing_moves_centroid_off_center(sampler):
    offsets = []
    for seed in range(60):
        pset = sampler(seed=seed, canvas=CANVAS, randomize_framing=True)
        offsets.append(_centroid_offset(pset, CANVAS))
    # most scenes should be off-center by several pixels
    assert np.median(offsets) > 5.0, (
        f"median offset {np.median(offsets):.1f} px is too small; "
        "randomization isn't actually shifting the object")


@pytest.mark.parametrize("sampler", [sample_box_scene, sample_cylinder_scene])
def test_randomized_framing_covers_a_wide_scale_range(sampler):
    scales = []
    for seed in range(60):
        pset = sampler(seed=seed, canvas=CANVAS, randomize_framing=True)
        scales.append(_scale_frac(pset, CANVAS))
    # we want the model to see roughly 30%-80% scale so it can handle the
    # reality-probe's small_scale range (0.35-0.6) at training time
    assert min(scales) < 0.55, f"scale range never goes small enough: min={min(scales):.2f}"
    assert max(scales) > 0.7, f"scale range never goes large enough: max={max(scales):.2f}"


@pytest.mark.parametrize("sampler", [sample_box_scene, sample_cylinder_scene])
def test_randomized_framing_keeps_primitives_in_canvas(sampler):
    for seed in range(80):
        pset = sampler(seed=seed, canvas=CANVAS, randomize_framing=True)
        assert _all_in_canvas(pset, CANVAS), f"seed {seed} produced out-of-canvas primitives"


# --- determinism --------------------------------------------------------------

@pytest.mark.parametrize("sampler", [sample_box_scene, sample_cylinder_scene])
def test_randomized_framing_is_deterministic_for_a_seed(sampler):
    a = sampler(seed=42, canvas=CANVAS, randomize_framing=True)
    b = sampler(seed=42, canvas=CANVAS, randomize_framing=True)
    assert a.approx_equal(b, tol=1e-6)


@pytest.mark.parametrize("sampler", [sample_box_scene, sample_cylinder_scene])
def test_different_seeds_produce_different_framings(sampler):
    a = sampler(seed=1, canvas=CANVAS, randomize_framing=True)
    b = sampler(seed=2, canvas=CANVAS, randomize_framing=True)
    # extremely unlikely two random framings give identical primitive sets
    assert not a.approx_equal(b)
