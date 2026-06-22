"""Independent (OpenCV) rasterizer for the sim-to-real probe ONLY.

The training pipeline renders with Pillow + supersampled polylines
(:mod:`lines.datagen.render`). To honestly measure the sim-to-real gap we need
out-of-distribution images that share the *exact same ground-truth primitives*
but are drawn by a different rasterizer. OpenCV uses its own anti-aliased shape
primitives (native circle/ellipse/line algorithms), so any score drop on these
images isolates rasterization-domain shift -- not a difference in geometry.

A JPEG round-trip is also provided because the project's real inputs are jpgs;
compression artifacts are the most realistic domain shift of all.

NEVER use this to generate training data -- that would defeat the probe.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from lines.primitives import Arc, Circle, Ellipse, Line, PrimitiveSet


def render_cv2(pset: PrimitiveSet, width: int, height: int,
               line_width: int = 2, antialias: bool = True) -> np.ndarray:
    """Render a primitive set with OpenCV: white background, black ink."""
    img = np.full((height, width), 255, dtype=np.uint8)
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    for prim in pset.primitives:
        if isinstance(prim, Line):
            cv2.line(img, _ipt(prim.p1), _ipt(prim.p2), 0, line_width, line_type)
        elif isinstance(prim, Circle):
            cv2.circle(img, _ipt(prim.center), int(round(prim.radius)), 0,
                       line_width, line_type)
        elif isinstance(prim, Arc):
            _draw_arc(img, prim, line_width, line_type)
        elif isinstance(prim, Ellipse):
            # cv2.ellipse takes integer center + (semi_major, semi_minor) axes
            # and rotation in DEGREES. Stage-2 Ellipse stores rotation in radians.
            cv2.ellipse(img, _ipt(prim.center),
                        (int(round(prim.semi_major)), int(round(prim.semi_minor))),
                        math.degrees(prim.rotation), 0, 360, 0,
                        line_width, line_type)
        else:
            raise TypeError(f"probe renderer cannot draw {type(prim).__name__}")
    return img


def jpeg_roundtrip(img: np.ndarray, quality: int = 75) -> np.ndarray:
    """Encode to JPEG at ``quality`` and decode back -- introduces real
    compression artifacts (the project's inputs are jpgs)."""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)


def _ipt(p):
    return (int(round(p[0])), int(round(p[1])))


def _draw_arc(img, arc: Arc, line_width: int, line_type) -> None:
    if arc.is_straight():
        cv2.line(img, _ipt(arc.p1), _ipt(arc.p2), 0, line_width, line_type)
        return
    (cx, cy), r, start, end = arc.to_center_params()
    start_deg = math.degrees(start)
    end_deg = math.degrees(end)
    # cv2 draws the angle interval [min, max]; our |sweep| < 360 so this is the
    # same set of points as the signed start->end traversal.
    lo, hi = sorted((start_deg, end_deg))
    cv2.ellipse(img, (int(round(cx)), int(round(cy))),
                (int(round(r)), int(round(r))), 0, lo, hi, 0, line_width, line_type)
