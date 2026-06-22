"""Renderer + flatten tests for Ellipse (Stage 2 Unit E3).

E2 added the ellipse branch to `flatten_primitive` (so the AR pipeline
doesn't crash when the model emits an ellipse token); this unit pins the
correctness properties:

* flatten produces a closed polyline whose points lie *exactly* on the
  parametric ellipse curve (verified algebraically, not via tolerance);
* extremes for an axis-aligned ellipse hit the expected ``(cx +/- a, cy)``
  and ``(cx, cy +/- b)`` points;
* ``Ellipse(a=b=r, theta=0)`` renders ~identically to a matching ``Circle``;
* a 45-deg-rotated long-thin ellipse renders with a roughly square bounding
  box (the major axis points where rotation says it does).

If any of these break later, the metric (Chamfer over flattened polylines)
and any visual eval will silently mis-score ellipses.
"""

import math

import numpy as np
import pytest

from lines.datagen.render import flatten_primitive, render_primitives
from lines.primitives import Circle, Ellipse, PrimitiveSet


CANVAS = (128, 128)


# --- flatten contract ---------------------------------------------------------

def test_flatten_ellipse_is_a_closed_polyline():
    e = Ellipse(center=(64.0, 64.0), semi_major=20.0,
                semi_minor=10.0, rotation=0.3)
    pts = flatten_primitive(e, n=64)
    assert len(pts) >= 64
    assert pts[0] == pts[-1]                 # explicitly closed


def test_flatten_ellipse_returns_enough_points_at_low_n():
    # the model/eval path always passes n>=32; make sure we don't truncate
    e = Ellipse(center=(64.0, 64.0), semi_major=20.0,
                semi_minor=10.0, rotation=0.0)
    pts = flatten_primitive(e, n=32)
    assert len(pts) >= 32


@pytest.mark.parametrize("e", [
    Ellipse(center=(64.0, 64.0), semi_major=20.0, semi_minor=10.0, rotation=0.0),
    Ellipse(center=(64.0, 64.0), semi_major=20.0, semi_minor=8.0, rotation=0.4),
    Ellipse(center=(40.0, 90.0), semi_major=15.0, semi_minor=15.0, rotation=0.0),
    Ellipse(center=(60.0, 60.0), semi_major=30.0, semi_minor=4.0, rotation=math.pi / 3),
])
def test_flatten_points_lie_on_the_parametric_ellipse_exactly(e):
    # In the ellipse's rotated frame: (u/a)^2 + (v/b)^2 must equal 1.
    pts = flatten_primitive(e, n=64)
    ca, sa = math.cos(e.rotation), math.sin(e.rotation)
    for x, y in pts:
        dx, dy = x - e.center[0], y - e.center[1]
        u =  dx * ca + dy * sa     # component along the semi-major direction
        v = -dx * sa + dy * ca     # component along the semi-minor direction
        # this is a derivation check, not a sampling tolerance -- should be exact
        assert math.isclose((u / e.semi_major) ** 2 + (v / e.semi_minor) ** 2,
                            1.0, abs_tol=1e-9)


def test_flatten_axis_aligned_ellipse_extremes():
    # theta=0: extremes at (cx +/- a, cy) and (cx, cy +/- b)
    e = Ellipse(center=(64.0, 64.0), semi_major=20.0,
                semi_minor=10.0, rotation=0.0)
    pts = flatten_primitive(e, n=128)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert min(xs) == pytest.approx(44.0, abs=0.05)
    assert max(xs) == pytest.approx(84.0, abs=0.05)
    assert min(ys) == pytest.approx(54.0, abs=0.05)
    assert max(ys) == pytest.approx(74.0, abs=0.05)


# --- render contract ----------------------------------------------------------

def test_render_circle_and_equivalent_ellipse_are_visually_close():
    # An ellipse with a = b = r and theta = 0 IS a circle. Render output should
    # match a Circle of the same center/radius up to AA differences.
    r = 20.0
    circle = PrimitiveSet([Circle(center=(64.0, 64.0), radius=r)])
    ellipse = PrimitiveSet([Ellipse(center=(64.0, 64.0), semi_major=r,
                                    semi_minor=r, rotation=0.0)])
    img_c = render_primitives(circle, *CANVAS, line_width=2.0)
    img_e = render_primitives(ellipse, *CANVAS, line_width=2.0)
    diff = np.abs(img_c.astype(int) - img_e.astype(int))
    # tolerate small per-pixel AA differences; gross structural mismatch would
    # produce many large diffs.
    assert (diff > 64).sum() < 100, (
        f"too many divergent pixels ({(diff > 64).sum()}); circle/ellipse "
        "render shouldn't differ structurally")


def test_render_rotated_ellipse_has_axis_at_expected_angle():
    # A long thin ellipse rotated 45 degrees: its bounding box should be ~square.
    # An unrotated long thin ellipse: bounding box is markedly wider than tall.
    flat = PrimitiveSet([Ellipse(center=(64.0, 64.0), semi_major=40.0,
                                 semi_minor=4.0, rotation=0.0)])
    diag = PrimitiveSet([Ellipse(center=(64.0, 64.0), semi_major=40.0,
                                 semi_minor=4.0, rotation=math.pi / 4)])
    flat_img = render_primitives(flat, *CANVAS, line_width=2.0)
    diag_img = render_primitives(diag, *CANVAS, line_width=2.0)

    def _bbox_wh(img):
        ys, xs = np.where(img < 128)
        return xs.max() - xs.min(), ys.max() - ys.min()

    fw, fh = _bbox_wh(flat_img)
    dw, dh = _bbox_wh(diag_img)
    assert fw > 3 * fh                            # flat ellipse: wide
    assert abs(dw - dh) < 0.15 * max(dw, dh)      # diagonal ellipse: ~square


def test_render_does_not_crash_on_near_circular_ellipse():
    # the a == b case is mathematically degenerate (rotation is meaningless) but
    # must not blow up the renderer.
    e = PrimitiveSet([Ellipse(center=(64.0, 64.0), semi_major=15.0,
                              semi_minor=15.0, rotation=1.2)])
    img = render_primitives(e, *CANVAS, line_width=2.0)
    assert img.min() < 128                        # ink was drawn
    assert img.shape == CANVAS[::-1]              # (H, W)
