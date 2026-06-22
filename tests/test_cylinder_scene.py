"""Tests for the 3D cylinder scene generator (Stage 2 Unit E5).

Same generate-don't-extract insight as box scenes: project the rim's circle
plane to image coordinates analytically (closed-form SVD) and emit the
resulting :class:`Ellipse` directly. Silhouette lines come from the
``axis x forward`` bitangent. Back-face culling picks the visible rim.

Three load-bearing properties:

1. **Structure.** Every scene has 2 silhouette lines + 1 visible rim ellipse;
   primitives are valid and lie within the canvas.
2. **Determinism.** Same seed -> same primitive set.
3. **Geometric correctness on hand-checkable cases** -- the math has to be
   exactly right, not "approximately working".
"""

import math

import numpy as np
import pytest

from lines.datagen.cylinder_scene import (
    project_cylinder_to_primitives,
    sample_cylinder_scene,
)
from lines.datagen.projection import Camera
from lines.datagen.render import flatten_primitive
from lines.datagen.sampler2d import Canvas
from lines.primitives import Ellipse, Line


CANVAS = Canvas(128, 128)


def _within(prim, canvas):
    return all(0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height
               for x, y in flatten_primitive(prim, 96))


# --- structure ----------------------------------------------------------------

def test_scene_has_two_lines_and_one_ellipse():
    pset = sample_cylinder_scene(seed=7, canvas=CANVAS)
    types = sorted(p.type for p in pset.primitives)
    assert types == ["ellipse", "line", "line"]


def test_all_primitives_valid_and_in_canvas():
    for seed in range(60):
        pset = sample_cylinder_scene(seed=seed, canvas=CANVAS)
        for prim in pset.primitives:
            assert prim.is_valid()
            assert _within(prim, CANVAS)


def test_deterministic_for_a_seed():
    a = sample_cylinder_scene(seed=11, canvas=CANVAS)
    b = sample_cylinder_scene(seed=11, canvas=CANVAS)
    assert a.approx_equal(b, tol=1e-6)


def test_rim_ellipse_is_canonical():
    # Ellipse.canonical: semi_major >= semi_minor, rotation in [0, pi)
    for seed in range(30):
        pset = sample_cylinder_scene(seed=seed, canvas=CANVAS)
        rim = next(p for p in pset.primitives if isinstance(p, Ellipse))
        assert rim.semi_major >= rim.semi_minor
        assert 0.0 <= rim.rotation < math.pi


# --- geometric correctness on hand-checkable cases ----------------------------

def test_axis_parallel_to_image_normal_yields_circle_like_rim():
    # axis along +z, camera looking along -z -> rim circle projects to a CIRCLE
    # (an ellipse with semi_major == semi_minor)
    cam = Camera.looking_from(direction=(0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0))
    radius, height = 1.0, 2.0
    axis = np.array([0.0, 0.0, 1.0])
    prims = project_cylinder_to_primitives(axis, radius, height, cam,
                                            cylinder_center=np.zeros(3))
    rim = next(p for p in prims if isinstance(p, Ellipse))
    assert rim.semi_major == pytest.approx(radius, abs=1e-9)
    assert rim.semi_minor == pytest.approx(radius, abs=1e-9)


def test_oblique_rim_is_foreshortened_along_axis_direction():
    # Axis at 60 degrees from the view direction:
    #   semi_major == r (the rim's diameter perpendicular to the axis projection)
    #   semi_minor == r * |axis . forward| (the foreshortened diameter)
    cam = Camera.looking_from(direction=(0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0))
    radius, height = 1.0, 3.0
    angle = math.radians(60)
    axis = np.array([math.sin(angle), 0.0, math.cos(angle)])
    prims = project_cylinder_to_primitives(axis, radius, height, cam,
                                            cylinder_center=np.zeros(3))
    rim = next(p for p in prims if isinstance(p, Ellipse))
    assert rim.semi_major == pytest.approx(radius, abs=1e-9)
    assert rim.semi_minor == pytest.approx(radius * abs(math.cos(angle)), abs=1e-9)


def test_silhouette_lines_are_parallel_and_separated_by_2r():
    # For any non-degenerate view, the two silhouette lines should be parallel
    # in 2D and separated by 2*r * |bitangent in image| = 2*r (bitangent is unit
    # and perpendicular to forward, so its image-space length is 1).
    cam = Camera.looking_from(direction=(0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0))
    radius, height = 1.0, 3.0
    angle = math.radians(45)
    axis = np.array([math.sin(angle), 0.0, math.cos(angle)])
    prims = project_cylinder_to_primitives(axis, radius, height, cam,
                                            cylinder_center=np.zeros(3))
    lines = [p for p in prims if isinstance(p, Line)]
    assert len(lines) == 2

    def _dir(line):
        dx = line.p2[0] - line.p1[0]
        dy = line.p2[1] - line.p1[1]
        L = math.hypot(dx, dy)
        return (dx / L, dy / L)

    d1 = _dir(lines[0])
    d2 = _dir(lines[1])
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    assert abs(cross) < 1e-9          # parallel

    # perpendicular distance from one line's p1 to the other line
    # = |( (a.p1 - b.p1) . n )| where n is the unit normal to b
    a, b = lines[0], lines[1]
    nx, ny = -d2[1], d2[0]
    dx, dy = a.p1[0] - b.p1[0], a.p1[1] - b.p1[1]
    sep = abs(dx * nx + dy * ny)
    assert sep == pytest.approx(2 * radius, abs=1e-9)


def test_visible_rim_picks_the_camera_facing_one():
    # cylinder center at origin; top rim is at +axis*h/2.
    # For axis pointing toward camera (axis . forward > 0): visible = top.
    # For axis pointing away from camera (axis . forward < 0): visible = bottom.
    cam = Camera.looking_from(direction=(0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0))
    radius, height = 1.0, 3.0
    angle = math.radians(60)
    axis_up = np.array([math.sin(angle), 0.0, math.cos(angle)])    # tilts toward camera
    axis_dn = np.array([math.sin(angle), 0.0, -math.cos(angle)])   # tilts away

    prims_up = project_cylinder_to_primitives(axis_up, radius, height, cam,
                                               cylinder_center=np.zeros(3))
    prims_dn = project_cylinder_to_primitives(axis_dn, radius, height, cam,
                                               cylinder_center=np.zeros(3))

    rim_up = next(p for p in prims_up if isinstance(p, Ellipse))
    rim_dn = next(p for p in prims_dn if isinstance(p, Ellipse))

    # Camera basis: right=(1,0,0), up=(0,1,0), forward=(0,0,1).
    #
    # axis_up = (sin60, 0, cos60) tilts the cylinder toward +x AND +z. The
    # camera-facing rim has outward normal +axis_up, so its 3D center is
    # +axis_up*h/2 = (sin60*h/2, 0, cos60*h/2). Image-x is positive.
    #
    # axis_dn = (sin60, 0, -cos60) tilts toward +x AND -z. The camera-facing
    # rim is now the *bottom* rim (outward normal -axis_dn = (-sin60, 0, cos60)),
    # whose 3D center is -axis_dn*h/2 = (-sin60*h/2, 0, cos60*h/2). Image-x is
    # NEGATIVE.
    #
    # The opposite signs in image-x are exactly what tells us the right rim
    # was picked in each case -- a wrong-rim selection would put both at +x.
    assert rim_up.center[0] > 0.0
    assert rim_dn.center[0] < 0.0


# --- realistic sampling -------------------------------------------------------

def test_sampled_scenes_avoid_degenerate_views():
    # Near-end-on (axis ~ +/- forward) makes the rim near-degenerate;
    # near-side-on makes the rim very thin. Sampler should reject both.
    seen_ratios = []
    for seed in range(60):
        pset = sample_cylinder_scene(seed=seed, canvas=CANVAS)
        rim = next(p for p in pset.primitives if isinstance(p, Ellipse))
        seen_ratios.append(rim.semi_minor / rim.semi_major)
    # rim should never collapse: minimum ratio above some floor
    assert min(seen_ratios) > 0.10
    # and rim should never be a perfect circle every time (no diversity)
    assert max(seen_ratios) < 0.97
