"""Tests for the independent (OpenCV) probe renderer.

This renderer exists ONLY to produce out-of-distribution images for the
sim-to-real probe: same primitives, a different rasterizer. It must never be
used to make training data.
"""

import numpy as np

from lines.datagen.probe_render import render_cv2, jpeg_roundtrip
from lines.datagen.sampler2d import Canvas
from lines.primitives import Arc, Circle, Line, PrimitiveSet

CANVAS = Canvas(96, 96)


def test_renders_expected_format_and_shape():
    img = render_cv2(PrimitiveSet([Circle(center=(48.0, 48.0), radius=20.0)]),
                     CANVAS.width, CANVAS.height)
    assert img.shape == (96, 96)
    assert img.dtype == np.uint8
    assert img[0, 0] == 255       # white background
    assert img.min() < 128        # black ink present


def test_line_marks_its_row():
    img = render_cv2(PrimitiveSet([Line(p1=(10.0, 48.0), p2=(86.0, 48.0))]),
                     CANVAS.width, CANVAS.height, line_width=2)
    assert img[48, 40] < 128
    assert img[10, :].min() == 255


def test_arc_draws_ink_without_crashing():
    img = render_cv2(PrimitiveSet([Arc(p1=(20.0, 70.0), p2=(76.0, 70.0), bulge=0.6)]),
                     CANVAS.width, CANVAS.height)
    assert img.min() < 128


def test_empty_set_is_all_white():
    img = render_cv2(PrimitiveSet([]), CANVAS.width, CANVAS.height)
    assert (img == 255).all()


def test_independent_renderer_differs_from_training_renderer():
    # the whole point: same primitives, a different rasterizer -> different pixels
    from lines.datagen.render import render_primitives
    pset = PrimitiveSet([Circle(center=(48.0, 48.0), radius=20.0),
                         Line(p1=(5.0, 5.0), p2=(90.0, 90.0))])
    ours = render_primitives(pset, 96, 96)
    theirs = render_cv2(pset, 96, 96)
    assert not np.array_equal(ours, theirs)        # genuinely a different image
    # but structurally similar: both ink the same broad region
    overlap = ((ours < 128) & (theirs < 128)).sum()
    assert overlap > 0


def test_jpeg_roundtrip_is_lossy_but_preserves_structure():
    img = render_cv2(PrimitiveSet([Circle(center=(48.0, 48.0), radius=20.0)]),
                     CANVAS.width, CANVAS.height)
    jpg = jpeg_roundtrip(img, quality=60)
    assert jpg.shape == img.shape
    assert not np.array_equal(jpg, img)            # compression changed pixels
    # introduces intermediate gray values absent from the clean render
    assert ((jpg > 10) & (jpg < 245)).sum() > ((img > 10) & (img < 245)).sum()
