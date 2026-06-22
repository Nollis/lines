"""Tests for the 3D reality-check perturbations.

Each perturbation has to actually do what it's named after; otherwise the
probe results are vacuous. Two structural rules:

* **pixel-only** perturbations (cv2, stroke_thicker, stroke_thinner, jpeg)
  leave the GT primitives unchanged.
* **spatial** perturbations (off_center, small_scale) transform the GT
  primitives to match the new image. Otherwise the metric scores against
  the wrong target.
"""

import math

import numpy as np
import pytest

from lines.datagen.cylinder_scene import sample_cylinder_scene
from lines.datagen.realityprobe import PERTURBATIONS, make_probe_sample
from lines.datagen.render import flatten_primitive
from lines.datagen.sampler2d import Canvas


CANVAS = Canvas(128, 128)


def _ink(img):
    return (img < 128).sum()


def _centroid(img):
    ys, xs = np.where(img < 128)
    if len(xs) == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _extent(img):
    ys, xs = np.where(img < 128)
    if len(xs) == 0:
        return 0.0
    return float(max(xs.max() - xs.min(), ys.max() - ys.min()))


# --- shape / dtype contract --------------------------------------------------

@pytest.mark.parametrize("perturbation", sorted(PERTURBATIONS))
def test_every_perturbation_returns_a_uint8_grayscale_image(perturbation):
    pset = sample_cylinder_scene(seed=3, canvas=CANVAS)
    img, _ = make_probe_sample(pset, CANVAS, perturbation, seed=0)
    assert img.dtype == np.uint8
    assert img.shape == (CANVAS.height, CANVAS.width)


@pytest.mark.parametrize("perturbation", sorted(PERTURBATIONS))
def test_every_perturbation_returns_a_valid_primitive_set(perturbation):
    pset = sample_cylinder_scene(seed=3, canvas=CANVAS)
    _, gt = make_probe_sample(pset, CANVAS, perturbation, seed=0)
    assert len(gt.primitives) >= 1
    for prim in gt.primitives:
        assert prim.is_valid()


# --- pixel-only perturbations: GT primitives unchanged ----------------------

@pytest.mark.parametrize("perturbation",
                         ["cv2", "stroke_thicker", "stroke_thinner", "jpeg"])
def test_pixel_only_perturbation_preserves_gt(perturbation):
    pset = sample_cylinder_scene(seed=5, canvas=CANVAS)
    _, gt = make_probe_sample(pset, CANVAS, perturbation, seed=0)
    assert gt.approx_equal(pset, tol=1e-6)


# --- per-perturbation behavior ----------------------------------------------

def test_baseline_pixels_match_a_direct_render():
    # the baseline probe should be byte-identical to the default render path
    from lines.datagen.render import render_primitives
    pset = sample_cylinder_scene(seed=7, canvas=CANVAS)
    img_probe, _ = make_probe_sample(pset, CANVAS, "baseline", seed=0)
    img_ref = render_primitives(pset, CANVAS.width, CANVAS.height, line_width=2.0)
    assert np.array_equal(img_probe, img_ref)


def test_cv2_perturbation_differs_from_baseline():
    pset = sample_cylinder_scene(seed=11, canvas=CANVAS)
    img_base, _ = make_probe_sample(pset, CANVAS, "baseline", seed=0)
    img_cv2, _ = make_probe_sample(pset, CANVAS, "cv2", seed=0)
    assert not np.array_equal(img_base, img_cv2)


def test_stroke_thicker_has_substantially_more_ink():
    pset = sample_cylinder_scene(seed=13, canvas=CANVAS)
    img_base, _ = make_probe_sample(pset, CANVAS, "baseline", seed=0)
    img_thick, _ = make_probe_sample(pset, CANVAS, "stroke_thicker", seed=0)
    assert _ink(img_thick) > 1.4 * _ink(img_base)


def test_stroke_thinner_has_substantially_less_ink():
    pset = sample_cylinder_scene(seed=17, canvas=CANVAS)
    img_base, _ = make_probe_sample(pset, CANVAS, "baseline", seed=0)
    img_thin, _ = make_probe_sample(pset, CANVAS, "stroke_thinner", seed=0)
    assert _ink(img_thin) < 0.7 * _ink(img_base)


def test_jpeg_introduces_intermediate_gray_pixels():
    pset = sample_cylinder_scene(seed=19, canvas=CANVAS)
    img_base, _ = make_probe_sample(pset, CANVAS, "baseline", seed=0)
    img_jpeg, _ = make_probe_sample(pset, CANVAS, "jpeg", seed=0)
    mid_base = ((img_base > 10) & (img_base < 245)).sum()
    mid_jpeg = ((img_jpeg > 10) & (img_jpeg < 245)).sum()
    assert mid_jpeg > 1.3 * mid_base


# --- spatial perturbations: GT transforms with the image --------------------

def test_off_center_moves_inked_centroid_off_canvas_center():
    pset = sample_cylinder_scene(seed=23, canvas=CANVAS)
    img_base, gt_base = make_probe_sample(pset, CANVAS, "baseline", seed=0)
    img_off, gt_off = make_probe_sample(pset, CANVAS, "off_center", seed=42)
    cb, co = _centroid(img_base), _centroid(img_off)
    assert math.hypot(co[0] - cb[0], co[1] - cb[1]) > 5.0
    # GT primitives must transform with the image (otherwise metric is wrong)
    assert not gt_off.approx_equal(gt_base)


def test_off_center_keeps_primitives_inside_canvas():
    for seed in range(20):
        pset = sample_cylinder_scene(seed=seed, canvas=CANVAS)
        _, gt = make_probe_sample(pset, CANVAS, "off_center", seed=seed * 7)
        for prim in gt.primitives:
            for x, y in flatten_primitive(prim, 64):
                assert 0.0 <= x <= CANVAS.width
                assert 0.0 <= y <= CANVAS.height


def test_small_scale_shrinks_inked_extent():
    pset = sample_cylinder_scene(seed=29, canvas=CANVAS)
    img_base, gt_base = make_probe_sample(pset, CANVAS, "baseline", seed=0)
    img_small, gt_small = make_probe_sample(pset, CANVAS, "small_scale", seed=0)
    assert _extent(img_small) < 0.7 * _extent(img_base)
    assert not gt_small.approx_equal(gt_base)   # GT scaled too


def test_small_scale_keeps_primitives_inside_canvas():
    for seed in range(20):
        pset = sample_cylinder_scene(seed=seed, canvas=CANVAS)
        _, gt = make_probe_sample(pset, CANVAS, "small_scale", seed=seed * 11)
        for prim in gt.primitives:
            for x, y in flatten_primitive(prim, 64):
                assert 0.0 <= x <= CANVAS.width
                assert 0.0 <= y <= CANVAS.height


# --- determinism ------------------------------------------------------------

@pytest.mark.parametrize("perturbation", sorted(PERTURBATIONS))
def test_perturbation_is_deterministic_for_a_seed(perturbation):
    pset = sample_cylinder_scene(seed=2, canvas=CANVAS)
    a_img, a_gt = make_probe_sample(pset, CANVAS, perturbation, seed=12345)
    b_img, b_gt = make_probe_sample(pset, CANVAS, perturbation, seed=12345)
    assert np.array_equal(a_img, b_img)
    assert a_gt.approx_equal(b_gt)


def test_unknown_perturbation_raises():
    pset = sample_cylinder_scene(seed=0, canvas=CANVAS)
    with pytest.raises(ValueError, match="unknown perturbation"):
        make_probe_sample(pset, CANVAS, "no_such_thing", seed=0)
