"""Tests for the Ellipse primitive (Stage 2 Unit E1).

Ellipse is the first new primitive added since v1, introduced to represent the
projected rim of a cylinder. Stored as (cx, cy, semi_major, semi_minor,
rotation) -- five floats, exactly matching the existing N_PARAMS = 5 so the
model's parameter head doesn't need to grow.

Two load-bearing properties:

1. **Canonical form** -- the same ellipse must always produce the same params:
   semi_major >= semi_minor (swap-and-rotate if not), and rotation reduced to
   [0, pi) since an ellipse is symmetric under rotation by pi.
2. **approx_equal respects the geometric symmetries** -- two ellipses that
   differ only by a +/-pi rotation, or by axis-swap-with-perpendicular-rotation,
   must compare equal.

If these aren't right, autoregressive token sequences for "the same" cylinder
rim won't be deterministic, and the model has nothing to learn.
"""

import math

import pytest

from lines.primitives import Ellipse, PrimitiveSet, primitive_from_dict


# --- dict round-trip ----------------------------------------------------------

def test_dict_round_trip():
    e = Ellipse(center=(50.0, 50.0), semi_major=20.0,
                semi_minor=10.0, rotation=0.3)
    assert Ellipse.from_dict(e.to_dict()) == e


def test_type_tag_in_dict():
    e = Ellipse(center=(50.0, 50.0), semi_major=20.0,
                semi_minor=10.0, rotation=0.3)
    assert e.to_dict()["type"] == "ellipse"


def test_primitive_from_dict_dispatches_ellipse():
    e = Ellipse(center=(50.0, 50.0), semi_major=20.0,
                semi_minor=10.0, rotation=0.3)
    restored = primitive_from_dict(e.to_dict())
    assert isinstance(restored, Ellipse)
    assert restored == e


def test_primitive_set_round_trips_with_ellipse():
    from lines.primitives import Line
    pset = PrimitiveSet([
        Line(p1=(0.0, 0.0), p2=(10.0, 10.0)),
        Ellipse(center=(50.0, 50.0), semi_major=15.0,
                semi_minor=8.0, rotation=0.5),
    ])
    restored = PrimitiveSet.from_json(pset.to_json())
    assert restored == pset


# --- normalize / denormalize --------------------------------------------------

def test_normalize_then_denormalize_is_identity():
    e = Ellipse(center=(50.0, 50.0), semi_major=20.0,
                semi_minor=10.0, rotation=0.3)
    canvas = 200
    assert e.normalized(canvas, canvas).denormalized(canvas, canvas).approx_equal(e)


def test_normalized_coords_scale_correctly():
    e = Ellipse(center=(100.0, 50.0), semi_major=40.0,
                semi_minor=20.0, rotation=0.5)
    n = e.normalized(200, 200)
    assert n.center == pytest.approx((0.5, 0.25))
    assert n.semi_major == pytest.approx(0.2)
    assert n.semi_minor == pytest.approx(0.1)


def test_rotation_is_invariant_under_normalization():
    e = Ellipse(center=(100.0, 50.0), semi_major=40.0,
                semi_minor=20.0, rotation=0.7)
    assert e.normalized(200, 200).rotation == pytest.approx(0.7)


# --- validation ---------------------------------------------------------------

def test_zero_semi_axis_is_invalid():
    assert not Ellipse(center=(0.0, 0.0), semi_major=0.0,
                       semi_minor=5.0, rotation=0.0).is_valid()
    assert not Ellipse(center=(0.0, 0.0), semi_major=5.0,
                       semi_minor=0.0, rotation=0.0).is_valid()


def test_well_formed_ellipse_is_valid():
    assert Ellipse(center=(0.0, 0.0), semi_major=10.0,
                   semi_minor=5.0, rotation=0.0).is_valid()
    # circle as a degenerate ellipse (a = b) is still a valid ellipse
    assert Ellipse(center=(0.0, 0.0), semi_major=5.0,
                   semi_minor=5.0, rotation=0.0).is_valid()


# --- canonical form (the load-bearing structural rule) ------------------------

def test_canonical_swaps_when_b_greater_than_a():
    # b > a: swap and rotate by pi/2 (then reduce mod pi)
    e = Ellipse(center=(50.0, 50.0), semi_major=5.0,
                semi_minor=10.0, rotation=0.1)
    c = e.canonical()
    assert c.semi_major == 10.0
    assert c.semi_minor == 5.0
    expected_theta = (0.1 + math.pi / 2) % math.pi
    assert math.isclose(c.rotation, expected_theta, abs_tol=1e-9)


def test_canonical_reduces_theta_mod_pi():
    e = Ellipse(center=(0.0, 0.0), semi_major=10.0,
                semi_minor=5.0, rotation=0.2 + math.pi)
    c = e.canonical()
    assert math.isclose(c.rotation, 0.2, abs_tol=1e-9)
    assert 0.0 <= c.rotation < math.pi


def test_canonical_handles_negative_theta():
    e = Ellipse(center=(0.0, 0.0), semi_major=10.0,
                semi_minor=5.0, rotation=-0.3)
    c = e.canonical()
    assert 0.0 <= c.rotation < math.pi


def test_canonical_is_idempotent():
    e = Ellipse(center=(0.0, 0.0), semi_major=10.0,
                semi_minor=5.0, rotation=0.4)
    once = e.canonical()
    twice = once.canonical()
    assert once == twice


def test_canonical_leaves_already_canonical_ellipse_unchanged():
    e = Ellipse(center=(50.0, 50.0), semi_major=10.0,
                semi_minor=5.0, rotation=0.3)
    c = e.canonical()
    assert c.center == e.center
    assert c.semi_major == e.semi_major
    assert c.semi_minor == e.semi_minor
    assert math.isclose(c.rotation, e.rotation, abs_tol=1e-9)


# --- approx_equal respects geometric symmetries -------------------------------

def test_approx_equal_under_theta_pi_symmetry():
    # rotating an ellipse by pi gives the same ellipse
    e1 = Ellipse(center=(50.0, 50.0), semi_major=10.0,
                 semi_minor=5.0, rotation=0.3)
    e2 = Ellipse(center=(50.0, 50.0), semi_major=10.0,
                 semi_minor=5.0, rotation=0.3 + math.pi)
    assert e1.approx_equal(e2)


def test_approx_equal_under_axis_swap_with_perpendicular_rotation():
    # (a=10, b=5, theta=0.3) is the SAME ellipse as (a=5, b=10, theta=0.3 - pi/2)
    e1 = Ellipse(center=(50.0, 50.0), semi_major=10.0,
                 semi_minor=5.0, rotation=0.3)
    e2 = Ellipse(center=(50.0, 50.0), semi_major=5.0,
                 semi_minor=10.0, rotation=0.3 - math.pi / 2)
    assert e1.approx_equal(e2)


def test_approx_equal_at_theta_boundary():
    # tiny positive theta and tiny negative theta are geometrically near-identical
    # (both ~horizontal major axis); canonical maps them to opposite ends of
    # [0, pi), so naive comparison would fail.
    e1 = Ellipse(center=(0.0, 0.0), semi_major=10.0,
                 semi_minor=5.0, rotation=0.001)
    e2 = Ellipse(center=(0.0, 0.0), semi_major=10.0,
                 semi_minor=5.0, rotation=-0.001)
    assert e1.approx_equal(e2, tol=1e-2)


def test_approx_equal_rejects_genuinely_different_ellipses():
    e1 = Ellipse(center=(50.0, 50.0), semi_major=10.0,
                 semi_minor=5.0, rotation=0.0)
    e2 = Ellipse(center=(50.0, 50.0), semi_major=10.0,
                 semi_minor=5.0, rotation=math.pi / 4)  # 45 degrees off
    assert not e1.approx_equal(e2)


def test_approx_equal_rejects_different_center():
    e1 = Ellipse(center=(50.0, 50.0), semi_major=10.0,
                 semi_minor=5.0, rotation=0.0)
    e2 = Ellipse(center=(60.0, 50.0), semi_major=10.0,
                 semi_minor=5.0, rotation=0.0)
    assert not e1.approx_equal(e2)


def test_approx_equal_rejects_different_semi_axis():
    e1 = Ellipse(center=(50.0, 50.0), semi_major=10.0,
                 semi_minor=5.0, rotation=0.0)
    e2 = Ellipse(center=(50.0, 50.0), semi_major=12.0,
                 semi_minor=5.0, rotation=0.0)
    assert not e1.approx_equal(e2)
