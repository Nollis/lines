"""Tests for dataset write/reload and render-param randomization (Unit 2)."""

import numpy as np

from lines.datagen.dataset import Dataset, write_dataset
from lines.datagen.randomize import sample_render_params
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas, sample_primitive_set


CANVAS = Canvas(96, 96)


def test_render_params_are_deterministic_and_in_range():
    a = sample_render_params(np.random.default_rng(7))
    b = sample_render_params(np.random.default_rng(7))
    assert a == b
    assert 1.0 <= a.line_width <= 4.0
    assert a.supersample >= 1


def test_write_creates_images_and_manifest(tmp_path):
    write_dataset(tmp_path, n_samples=5, seed=0, canvas=CANVAS)
    assert (tmp_path / "manifest.json").exists()
    assert len(list((tmp_path / "images").glob("*.png"))) == 5


def test_dataset_reloads_with_correct_length(tmp_path):
    write_dataset(tmp_path, n_samples=4, seed=10, canvas=CANVAS)
    assert len(Dataset(tmp_path)) == 4


def test_manifest_primitives_reconstruct_what_was_sampled(tmp_path):
    write_dataset(tmp_path, n_samples=6, seed=100, canvas=CANVAS)
    ds = Dataset(tmp_path)
    for i in range(len(ds)):
        _img, pset = ds[i]
        expected = sample_primitive_set(100 + i, canvas=CANVAS)
        assert pset.approx_equal(expected)


def test_reloaded_image_has_expected_shape_and_ink(tmp_path):
    write_dataset(tmp_path, n_samples=3, seed=0, canvas=CANVAS)
    ds = Dataset(tmp_path)
    img, _pset = ds[0]
    assert img.shape == (CANVAS.height, CANVAS.width)
    assert img.dtype == np.uint8
    assert img.min() < 128  # ink present


def test_reloaded_image_matches_rerender_from_manifest(tmp_path):
    # the stored image must be reproducible from its manifest primitives + render params
    write_dataset(tmp_path, n_samples=3, seed=5, canvas=CANVAS, randomize=False)
    ds = Dataset(tmp_path)
    img, pset = ds[0]
    rerender = render_primitives(pset, CANVAS.width, CANVAS.height,
                                 line_width=ds.line_width(0), supersample=ds.supersample(0))
    assert np.array_equal(img, rerender)
