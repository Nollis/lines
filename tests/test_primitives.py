"""Tests for the canonical primitive schema (Unit 1)."""

import math

import pytest

from lines.primitives import Arc, Bezier, Circle, Line, PrimitiveSet


# --- serialization round-trips -------------------------------------------------

def test_line_roundtrips_through_dict():
    line = Line(p1=(10.0, 20.0), p2=(100.0, 50.0))
    assert Line.from_dict(line.to_dict()) == line


def test_arc_roundtrips_through_dict():
    arc = Arc(p1=(10.0, 20.0), p2=(100.0, 50.0), bulge=0.5)
    assert Arc.from_dict(arc.to_dict()) == arc


def test_circle_roundtrips_through_dict():
    circle = Circle(center=(50.0, 50.0), radius=25.0)
    assert Circle.from_dict(circle.to_dict()) == circle


def test_bezier_roundtrips_through_dict():
    bez = Bezier(control_points=((0.0, 0.0), (10.0, 30.0), (40.0, 30.0), (50.0, 0.0)))
    assert Bezier.from_dict(bez.to_dict()) == bez


def test_primitive_set_roundtrips_through_json():
    pset = PrimitiveSet([
        Line(p1=(0.0, 0.0), p2=(10.0, 10.0)),
        Circle(center=(50.0, 50.0), radius=5.0),
        Arc(p1=(1.0, 0.0), p2=(0.0, 1.0), bulge=0.41421356),
    ])
    restored = PrimitiveSet.from_json(pset.to_json())
    assert restored == pset


def test_from_dict_dispatches_on_type_tag():
    items = [
        Line(p1=(0.0, 0.0), p2=(1.0, 1.0)).to_dict(),
        Circle(center=(2.0, 2.0), radius=1.0).to_dict(),
    ]
    assert items[0]["type"] == "line"
    assert items[1]["type"] == "circle"


# --- normalization -------------------------------------------------------------

W, H = 200, 200  # square canvas keeps radius scaling well-defined


@pytest.mark.parametrize("prim", [
    Line(p1=(10.0, 20.0), p2=(100.0, 50.0)),
    Arc(p1=(10.0, 20.0), p2=(100.0, 50.0), bulge=-0.3),
    Circle(center=(50.0, 60.0), radius=25.0),
    Bezier(control_points=((0.0, 0.0), (10.0, 30.0), (40.0, 30.0), (50.0, 0.0))),
])
def test_normalize_then_denormalize_is_identity(prim):
    restored = prim.normalized(W, H).denormalized(W, H)
    assert restored.approx_equal(prim)


def test_normalized_coordinates_are_unit_scaled():
    line = Line(p1=(50.0, 100.0), p2=(150.0, 200.0)).normalized(200, 200)
    assert line.p1 == pytest.approx((0.25, 0.5))
    assert line.p2 == pytest.approx((0.75, 1.0))


def test_bulge_is_invariant_under_normalization():
    arc = Arc(p1=(20.0, 20.0), p2=(80.0, 20.0), bulge=0.37)
    assert arc.normalized(W, H).bulge == pytest.approx(0.37)


# --- validation ----------------------------------------------------------------

def test_zero_length_line_is_invalid():
    assert not Line(p1=(5.0, 5.0), p2=(5.0, 5.0)).is_valid()


def test_zero_radius_circle_is_invalid():
    assert not Circle(center=(5.0, 5.0), radius=0.0).is_valid()


def test_coincident_endpoint_arc_is_invalid():
    # a "zero-sweep" arc collapses its endpoints
    assert not Arc(p1=(5.0, 5.0), p2=(5.0, 5.0), bulge=0.5).is_valid()


def test_well_formed_primitives_are_valid():
    assert Line(p1=(0.0, 0.0), p2=(10.0, 0.0)).is_valid()
    assert Circle(center=(0.0, 0.0), radius=3.0).is_valid()
    assert Arc(p1=(0.0, 0.0), p2=(2.0, 0.0), bulge=1.0).is_valid()


# --- arc <-> center-parameter conversion --------------------------------------

def test_semicircle_arc_resolves_to_known_center_and_radius():
    # bulge=1 => included angle 180deg; chord of length 2 => radius 1, center at midpoint
    arc = Arc(p1=(0.0, 0.0), p2=(2.0, 0.0), bulge=1.0)
    center, radius, _start, _end = arc.to_center_params()
    assert center == pytest.approx((1.0, 0.0), abs=1e-9)
    assert radius == pytest.approx(1.0, abs=1e-9)


def test_quarter_circle_positive_bulge_centers_correctly():
    # endpoints on the unit circle, 90deg CCW sweep => bulge = tan(22.5deg)
    arc = Arc(p1=(1.0, 0.0), p2=(0.0, 1.0), bulge=math.tan(math.radians(22.5)))
    center, radius, _s, _e = arc.to_center_params()
    assert center == pytest.approx((0.0, 0.0), abs=1e-9)
    assert radius == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("bulge", [0.41421356, -0.41421356, 0.2, -0.85, 1.0])
def test_arc_endpoint_bulge_roundtrips_through_center_params(bulge):
    arc = Arc(p1=(10.0, 30.0), p2=(70.0, 45.0), bulge=bulge)
    center, radius, start, end = arc.to_center_params()
    restored = Arc.from_center_params(center, radius, start, end)
    assert restored.approx_equal(arc)


def test_from_center_params_roundtrips_for_negative_sweep():
    center, radius, start, end = (1.0, 1.0), 1.0, math.radians(-90), math.radians(-180)
    arc = Arc.from_center_params(center, radius, start, end)
    c2, r2, s2, e2 = arc.to_center_params()
    assert c2 == pytest.approx(center, abs=1e-9)
    assert r2 == pytest.approx(radius, abs=1e-9)
    assert (e2 - s2) == pytest.approx(end - start, abs=1e-9)


# --- bulge == 0 line/arc unification ------------------------------------------

def test_zero_bulge_arc_is_straight():
    arc = Arc(p1=(0.0, 0.0), p2=(10.0, 0.0), bulge=0.0)
    assert arc.is_straight()
    assert arc.is_valid()  # a straight segment is a valid renderable arc


def test_straight_arc_has_no_finite_center():
    arc = Arc(p1=(0.0, 0.0), p2=(10.0, 0.0), bulge=0.0)
    with pytest.raises(ValueError):
        arc.to_center_params()
