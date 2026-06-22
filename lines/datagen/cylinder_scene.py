"""3D cylinder scenes -> 2 silhouette lines + 1 visible rim ellipse.

Stage 2's content unit. Same generate-don't-extract insight as
:mod:`lines.datagen.box_scene`: project the cylinder's rim circle to image
coordinates analytically (closed-form SVD of the 2x2 mapping matrix) and emit
the resulting :class:`Ellipse` directly -- no fitting, no rendering loop.

Geometry summary
----------------

For a cylinder with unit axis ``a`` (length 1), radius ``r``, height ``h``,
center ``c``, viewed orthographically through a Camera with right/up/forward:

* **Top rim center**    = ``c + (h/2) a`` ; bottom = ``c - (h/2) a``.
* **Silhouette tangent direction** = ``axis x forward``, normalized to
  ``n_hat``. Tangent points on each rim are at ``rim_center +/- r * n_hat``.
  The two silhouette lines connect (top + r*n_hat -> bottom + r*n_hat) and
  (top - r*n_hat -> bottom - r*n_hat).
* **Rim plane basis**: pick ``u`` perpendicular to ``axis`` (the projection of
  ``forward`` onto the rim plane, normalized) and ``v = axis x u``. The rim
  circle is ``rim_center + r*cos(t)*u + r*sin(t)*v`` for t in [0, 2*pi).
* **Projecting to image**: each rim point's image x,y is a linear function of
  ``(cos(t), sin(t))``, giving a 2x2 matrix M. The SVD ``M = U S V^T`` reads
  off the ellipse: ``semi_major = S[0]``, ``semi_minor = S[1]``, ``rotation
  = atan2(U[1,0], U[0,0])``.
* **Visible rim**: the one whose outward normal faces the camera (the same
  back-face cull rule we used for boxes).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from lines.datagen.projection import (
    Camera, fit_segments_to_canvas, random_rotation,
)
from lines.datagen.render import flatten_primitive
from lines.primitives import Ellipse, Line, PrimitiveSet


# --- analytic projection ------------------------------------------------------

def _project_point(p: np.ndarray, cam: Camera) -> Tuple[float, float]:
    return float(np.dot(p, cam.right)), float(np.dot(p, cam.up))


def _rim_ellipse(rim_center_3d: np.ndarray, axis: np.ndarray, radius: float,
                 cam: Camera) -> Ellipse:
    """Analytically project a rim circle to its image-space ellipse.

    Uses SVD of the 2x2 mapping (cos t, sin t) -> image (x, y) to extract the
    semi-axes and rotation in one closed-form step.
    """
    # in-rim-plane basis: u is `forward` projected onto the plane perpendicular
    # to `axis`, normalized. v = axis x u completes a right-handed frame.
    f = cam.forward
    u = f - np.dot(f, axis) * axis
    nu = np.linalg.norm(u)
    if nu < 1e-9:
        # forward is parallel to axis -> rim is camera-facing circle
        # pick any vector perpendicular to axis
        u = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = u - np.dot(u, axis) * axis
        u = u / np.linalg.norm(u)
    else:
        u = u / nu
    v = np.cross(axis, u)

    # 2x2 matrix that maps (cos t, sin t) -> (image x - cx, image y - cy)
    M = np.array([
        [radius * float(np.dot(u, cam.right)), radius * float(np.dot(v, cam.right))],
        [radius * float(np.dot(u, cam.up)),    radius * float(np.dot(v, cam.up))],
    ])

    U, sigma, _ = np.linalg.svd(M)
    # SVD gives U as a rotation/reflection. We want the rotation angle of the
    # first column (the semi-major direction). If det(U) < 0 it's a reflection;
    # negate the second column to make it a proper rotation (the second-axis
    # *direction* doesn't affect the ellipse shape).
    if np.linalg.det(U) < 0:
        U = U * np.array([[1.0, -1.0], [1.0, -1.0]])
    rotation = math.atan2(U[1, 0], U[0, 0])

    cx_img, cy_img = _project_point(rim_center_3d, cam)
    e = Ellipse(center=(cx_img, cy_img),
                semi_major=float(sigma[0]),
                semi_minor=float(sigma[1]),
                rotation=rotation)
    return e.canonical()


def project_cylinder_to_primitives(axis: np.ndarray, radius: float,
                                   height: float, cam: Camera,
                                   cylinder_center: np.ndarray) -> List:
    """Project one cylinder to its visible 2D primitives.

    Returns ``[Line, Line, Ellipse]``: the two silhouette lines connecting the
    two rim tangent points, and the visible (camera-facing) rim's ellipse.
    """
    axis = axis / np.linalg.norm(axis)

    top_center = cylinder_center + 0.5 * height * axis
    bot_center = cylinder_center - 0.5 * height * axis

    # silhouette bitangent (perpendicular to both axis and forward, image-plane)
    cross = np.cross(axis, cam.forward)
    n_norm = np.linalg.norm(cross)
    silhouette_lines: List[Line] = []
    if n_norm > 1e-9:
        n_hat = cross / n_norm
        for sign in (+1.0, -1.0):
            t_top = top_center + sign * radius * n_hat
            t_bot = bot_center + sign * radius * n_hat
            silhouette_lines.append(Line(p1=_project_point(t_top, cam),
                                         p2=_project_point(t_bot, cam)))

    # visible rim: outward normal faces the camera
    # top rim outward normal = +axis; bottom rim = -axis
    if float(np.dot(axis, cam.forward)) >= 0.0:
        visible_rim_center = top_center
        visible_axis = axis
    else:
        visible_rim_center = bot_center
        visible_axis = -axis

    rim = _rim_ellipse(visible_rim_center, visible_axis, radius, cam)
    return silhouette_lines + [rim]


# --- sampling -----------------------------------------------------------------

# Reject views where the rim aspect ratio gets too extreme. The ratio is
# |axis . forward| (foreshortening factor); accept only views where it's
# comfortably inside [0.15, 0.95] -- excludes near-side-on (rim ~= line) and
# near-end-on (cylinder ~= circle, silhouette ~= 0 length).
_MIN_FORESHORTEN = 0.15
_MAX_FORESHORTEN = 0.95
_MAX_TRIES = 64


def _fit_to_canvas(primitives, canvas, margin: float = 14.0):
    """Uniform scale + translate so all primitives sit within the canvas."""
    pts: List[Tuple[float, float]] = []
    for prim in primitives:
        pts.extend(flatten_primitive(prim, n=64))
    if not pts:
        return primitives
    arr = np.asarray(pts)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    avail = min(canvas.width, canvas.height) - 2 * margin
    scale = float(avail / extent.max())
    cx_img = canvas.width / 2 - (lo[0] + hi[0]) / 2 * scale
    cy_img = canvas.height / 2 - (lo[1] + hi[1]) / 2 * scale
    # y flip so the projection reads as image coords (y down)
    def _tx(p):
        x = p[0] * scale + cx_img
        y = canvas.height - (p[1] * scale + cy_img)
        return (float(x), float(y))

    out = []
    for prim in primitives:
        if isinstance(prim, Line):
            out.append(Line(p1=_tx(prim.p1), p2=_tx(prim.p2)))
        elif isinstance(prim, Ellipse):
            # center transforms; semi-axes scale; rotation is mirrored across
            # the y-flip (theta -> -theta), then re-canonicalized.
            new_center = _tx(prim.center)
            new_a = prim.semi_major * scale
            new_b = prim.semi_minor * scale
            new_rot = -prim.rotation
            out.append(Ellipse(center=new_center, semi_major=new_a,
                               semi_minor=new_b, rotation=new_rot).canonical())
        else:
            out.append(prim)
    return out


def sample_cylinder_scene(seed: int, canvas, *,
                          radius_range: Tuple[float, float] = (0.6, 1.4),
                          height_range: Tuple[float, float] = (1.6, 3.6),
                          margin: float = 14.0) -> PrimitiveSet:
    """Generate one randomly-oriented cylinder, project, fit to canvas."""
    rng = np.random.default_rng(seed)
    cam = Camera.looking_from(direction=(0.0, 0.0, -1.0),
                              up_hint=(0.0, 1.0, 0.0))

    for _ in range(_MAX_TRIES):
        radius = float(rng.uniform(*radius_range))
        height = float(rng.uniform(*height_range))
        # pick a random orientation; reject if the rim foreshortening is bad
        R = random_rotation(rng)
        axis = R @ np.array([0.0, 0.0, 1.0])
        cos_phi = abs(float(np.dot(axis, cam.forward)))
        if cos_phi < _MIN_FORESHORTEN or cos_phi > _MAX_FORESHORTEN:
            continue
        prims = project_cylinder_to_primitives(axis, radius, height, cam,
                                                cylinder_center=np.zeros(3))
        fitted = _fit_to_canvas(prims, canvas, margin=margin)
        if all(p.is_valid() for p in fitted):
            return PrimitiveSet(fitted)
    # fallback: an axis-aligned cylinder at a "nice" angle (should ~never hit)
    R = random_rotation(np.random.default_rng(seed + 1_000_000))
    axis = R @ np.array([0.0, 0.0, 1.0])
    prims = project_cylinder_to_primitives(axis, 1.0, 2.0, cam,
                                            cylinder_center=np.zeros(3))
    return PrimitiveSet(_fit_to_canvas(prims, canvas, margin=margin))
