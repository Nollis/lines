"""Tests for differentiable-render refinement (Unit 6).

The refinement snaps model-predicted primitive parameters to the input ink by
gradient descent on a differentiable soft-render loss. Core claims:
a perturbed primitive converges toward the target, a correct primitive stays
put, and -- unlike the training SoftRenderer -- arcs are refined properly.
"""

import math

import numpy as np

from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas
from lines.primitives import Arc, Circle, Line, PrimitiveSet
from lines.refine.diffvg_refine import refine_primitives

CANVAS = Canvas(96, 96)


def _target(pset):
    return render_primitives(pset, CANVAS.width, CANVAS.height, line_width=2.0)


def test_refine_pulls_circle_radius_toward_target():
    target_set = PrimitiveSet([Circle(center=(48.0, 48.0), radius=24.0)])
    img = _target(target_set)
    start = PrimitiveSet([Circle(center=(48.0, 48.0), radius=17.0)])  # too small
    refined = refine_primitives(start, img, CANVAS, steps=60, lr=8e-3)
    r = refined.primitives[0].radius
    assert abs(r - 24.0) < abs(17.0 - 24.0)      # moved closer
    assert abs(r - 24.0) < 2.5                    # and got reasonably close


def test_refine_pulls_circle_center_toward_target():
    target_set = PrimitiveSet([Circle(center=(48.0, 48.0), radius=20.0)])
    img = _target(target_set)
    start = PrimitiveSet([Circle(center=(40.0, 54.0), radius=20.0)])  # offset center
    refined = refine_primitives(start, img, CANVAS, steps=60, lr=8e-3)
    cx, cy = refined.primitives[0].center
    assert math.hypot(cx - 48.0, cy - 48.0) < math.hypot(40.0 - 48.0, 54.0 - 48.0)


def test_refine_pulls_line_endpoints_toward_target():
    target_set = PrimitiveSet([Line(p1=(15.0, 20.0), p2=(80.0, 70.0))])
    img = _target(target_set)
    start = PrimitiveSet([Line(p1=(20.0, 26.0), p2=(74.0, 64.0))])  # both ends off
    refined = refine_primitives(start, img, CANVAS, steps=60, lr=8e-3)
    ln = refined.primitives[0]
    before = math.hypot(20 - 15, 26 - 20) + math.hypot(74 - 80, 64 - 70)
    after = math.hypot(ln.p1[0] - 15, ln.p1[1] - 20) + math.hypot(ln.p2[0] - 80, ln.p2[1] - 70)
    assert after < before


def test_refine_improves_arc_geometry():
    # the case the training SoftRenderer cannot do (it treats arcs as lines)
    target_set = PrimitiveSet([Arc(p1=(20.0, 70.0), p2=(76.0, 70.0), bulge=0.6)])
    img = _target(target_set)
    start = PrimitiveSet([Arc(p1=(20.0, 70.0), p2=(76.0, 70.0), bulge=0.35)])  # too shallow
    refined = refine_primitives(start, img, CANVAS, steps=80, lr=8e-3)
    b = refined.primitives[0].bulge
    assert abs(b - 0.6) < abs(0.35 - 0.6)     # bulge moved toward target


def test_refine_is_stable_on_already_correct_primitive():
    target_set = PrimitiveSet([Circle(center=(48.0, 48.0), radius=20.0)])
    img = _target(target_set)
    refined = refine_primitives(target_set, img, CANVAS, steps=40, lr=8e-3)
    r = refined.primitives[0].radius
    cx, cy = refined.primitives[0].center
    assert abs(r - 20.0) < 1.5
    assert math.hypot(cx - 48.0, cy - 48.0) < 1.5


def test_refine_empty_set_returns_empty():
    img = _target(PrimitiveSet([Circle(center=(48.0, 48.0), radius=20.0)]))
    refined = refine_primitives(PrimitiveSet([]), img, CANVAS, steps=10)
    assert refined.primitives == []


def test_refine_preserves_primitive_types():
    start = PrimitiveSet([
        Line(p1=(10.0, 10.0), p2=(50.0, 50.0)),
        Circle(center=(60.0, 60.0), radius=15.0),
    ])
    img = _target(start)
    refined = refine_primitives(start, img, CANVAS, steps=20)
    assert [p.type for p in refined.primitives] == ["line", "circle"]
