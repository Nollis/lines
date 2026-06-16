"""Tests for the line-art rasterizer (Unit 2)."""

import math

import numpy as np

from lines.primitives import Arc, Bezier, Circle, Line, PrimitiveSet
from lines.datagen.render import flatten_primitive, render_primitives


# --- flattening ---------------------------------------------------------------

def test_flatten_line_is_its_two_endpoints():
    pts = flatten_primitive(Line(p1=(0.0, 0.0), p2=(10.0, 5.0)))
    assert pts == [(0.0, 0.0), (10.0, 5.0)]


def test_flatten_straight_arc_is_its_two_endpoints():
    pts = flatten_primitive(Arc(p1=(0.0, 0.0), p2=(10.0, 0.0), bulge=0.0))
    assert pts == [(0.0, 0.0), (10.0, 0.0)]


def test_flatten_circle_points_lie_on_the_circle():
    pts = flatten_primitive(Circle(center=(50.0, 50.0), radius=20.0), n=64)
    for x, y in pts:
        assert math.hypot(x - 50.0, y - 50.0) == np.float64(20.0).item() or \
            math.isclose(math.hypot(x - 50.0, y - 50.0), 20.0, abs_tol=1e-6)


def test_flatten_arc_endpoints_match_primitive():
    arc = Arc(p1=(10.0, 30.0), p2=(70.0, 45.0), bulge=0.4)
    pts = flatten_primitive(arc, n=32)
    assert math.isclose(pts[0][0], 10.0, abs_tol=1e-6) and math.isclose(pts[0][1], 30.0, abs_tol=1e-6)
    assert math.isclose(pts[-1][0], 70.0, abs_tol=1e-6) and math.isclose(pts[-1][1], 45.0, abs_tol=1e-6)


def test_flatten_bezier_hits_first_and_last_control_point():
    bez = Bezier(control_points=((0.0, 0.0), (10.0, 30.0), (40.0, 30.0), (50.0, 0.0)))
    pts = flatten_primitive(bez, n=20)
    assert math.isclose(pts[0][0], 0.0, abs_tol=1e-6) and math.isclose(pts[0][1], 0.0, abs_tol=1e-6)
    assert math.isclose(pts[-1][0], 50.0, abs_tol=1e-6) and math.isclose(pts[-1][1], 0.0, abs_tol=1e-6)


# --- rasterization ------------------------------------------------------------

def test_render_has_expected_shape_and_white_background():
    img = render_primitives(PrimitiveSet([Circle(center=(50.0, 50.0), radius=20.0)]), 100, 100)
    assert img.shape == (100, 100)
    assert img.dtype == np.uint8
    assert img[0, 0] == 255  # untouched corner stays white


def test_render_draws_some_dark_pixels_for_a_circle():
    img = render_primitives(PrimitiveSet([Circle(center=(50.0, 50.0), radius=20.0)]), 100, 100)
    assert img.min() < 128   # ink present
    assert img.max() == 255  # background present


def test_horizontal_line_marks_its_row_only():
    img = render_primitives(PrimitiveSet([Line(p1=(10.0, 50.0), p2=(90.0, 50.0))]), 100, 100, line_width=2.0)
    assert img[50, 40] < 128       # on the line
    assert img[10, :].min() == 255  # a far row is untouched


def test_render_is_deterministic():
    pset = PrimitiveSet([Circle(center=(50.0, 50.0), radius=20.0), Line(p1=(0.0, 0.0), p2=(99.0, 99.0))])
    a = render_primitives(pset, 100, 100)
    b = render_primitives(pset, 100, 100)
    assert np.array_equal(a, b)


def test_straight_arc_renders_identically_to_the_equivalent_line():
    line = PrimitiveSet([Line(p1=(10.0, 20.0), p2=(80.0, 70.0))])
    arc = PrimitiveSet([Arc(p1=(10.0, 20.0), p2=(80.0, 70.0), bulge=0.0)])
    assert np.array_equal(render_primitives(line, 100, 100), render_primitives(arc, 100, 100))


def test_resolution_argument_is_respected():
    img = render_primitives(PrimitiveSet([Circle(center=(64.0, 64.0), radius=30.0)]), 128, 96)
    assert img.shape == (96, 128)  # (height, width)
