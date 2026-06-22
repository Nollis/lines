"""Canonical geometric-primitive schema.

This is the single source of truth shared by the data generator, the classical
baseline, the model, the metric/eval harness, and the refiner. Every other
component imports these types; keep the representation stable.

Conventions
-----------
- Points are ``(x, y)`` tuples in canvas (pixel) coordinates.
- ``Arc`` is stored as ``endpoints + signed bulge`` (``bulge = tan(theta / 4)``,
  the DXF convention). Endpoints are directly localizable in an image and the
  bulge is a single signed scalar (sign = sweep direction) with no angle
  wraparound. ``bulge == 0`` degenerates to the straight segment between the
  endpoints, so lines and arcs unify smoothly.
- Full / near-full curvature is the ``Circle`` primitive, not an arc with an
  exploding bulge. The data generator caps arc sweep accordingly (Unit 2).
- Normalization assumes a square canvas so lengths (radius) have a single,
  well-defined scale factor. Bulge is a similarity invariant and is unchanged
  by normalization.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Tuple

Point = Tuple[float, float]

_EPS = 1e-9


def _approx_point(a: Point, b: Point, tol: float = 1e-6) -> bool:
    return math.isclose(a[0], b[0], abs_tol=tol) and math.isclose(a[1], b[1], abs_tol=tol)


# --- primitive types ----------------------------------------------------------

@dataclass(frozen=True)
class Line:
    p1: Point
    p2: Point
    type: str = "line"

    def to_dict(self) -> dict:
        return {"type": "line", "p1": list(self.p1), "p2": list(self.p2)}

    @staticmethod
    def from_dict(d: dict) -> "Line":
        return Line(p1=tuple(d["p1"]), p2=tuple(d["p2"]))

    def normalized(self, width: float, height: float) -> "Line":
        return Line(p1=_norm_pt(self.p1, width, height), p2=_norm_pt(self.p2, width, height))

    def denormalized(self, width: float, height: float) -> "Line":
        return Line(p1=_denorm_pt(self.p1, width, height), p2=_denorm_pt(self.p2, width, height))

    def is_valid(self) -> bool:
        return _dist(self.p1, self.p2) > _EPS

    def approx_equal(self, other: object, tol: float = 1e-6) -> bool:
        return (
            isinstance(other, Line)
            and _approx_point(self.p1, other.p1, tol)
            and _approx_point(self.p2, other.p2, tol)
        )


@dataclass(frozen=True)
class Arc:
    p1: Point
    p2: Point
    bulge: float
    type: str = "arc"

    def to_dict(self) -> dict:
        return {"type": "arc", "p1": list(self.p1), "p2": list(self.p2), "bulge": self.bulge}

    @staticmethod
    def from_dict(d: dict) -> "Arc":
        return Arc(p1=tuple(d["p1"]), p2=tuple(d["p2"]), bulge=float(d["bulge"]))

    def normalized(self, width: float, height: float) -> "Arc":
        # bulge is a similarity invariant -> unchanged
        return Arc(p1=_norm_pt(self.p1, width, height), p2=_norm_pt(self.p2, width, height), bulge=self.bulge)

    def denormalized(self, width: float, height: float) -> "Arc":
        return Arc(p1=_denorm_pt(self.p1, width, height), p2=_denorm_pt(self.p2, width, height), bulge=self.bulge)

    def is_straight(self) -> bool:
        return abs(self.bulge) <= _EPS

    def is_valid(self) -> bool:
        # endpoints must be distinct; bulge == 0 is a valid (straight) arc
        return _dist(self.p1, self.p2) > _EPS

    def to_center_params(self) -> Tuple[Point, float, float, float]:
        """Return ``(center, radius, start_angle, end_angle)`` (angles in radians).

        ``end_angle - start_angle`` is the signed sweep, so direction is preserved.
        Raises ``ValueError`` for a straight (``bulge == 0``) arc, which has no
        finite center.
        """
        if self.is_straight():
            raise ValueError("straight arc (bulge == 0) has no finite center")
        (x1, y1), (x2, y2) = self.p1, self.p2
        chord = _dist(self.p1, self.p2)
        if chord <= _EPS:
            raise ValueError("degenerate arc: coincident endpoints")
        theta = 4.0 * math.atan(self.bulge)          # signed included angle
        half = theta / 2.0
        r_signed = chord / (2.0 * math.sin(half))    # sign follows the sweep
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ux, uy = (x2 - x1) / chord, (y2 - y1) / chord
        nx, ny = -uy, ux                              # left normal to the chord
        apothem = r_signed * math.cos(half)
        cx, cy = mx + nx * apothem, my + ny * apothem
        start = math.atan2(y1 - cy, x1 - cx)
        end = start + theta
        return (cx, cy), abs(r_signed), start, end

    @classmethod
    def from_center_params(cls, center: Point, radius: float, start_angle: float, end_angle: float) -> "Arc":
        cx, cy = center
        p1 = (cx + radius * math.cos(start_angle), cy + radius * math.sin(start_angle))
        p2 = (cx + radius * math.cos(end_angle), cy + radius * math.sin(end_angle))
        bulge = math.tan((end_angle - start_angle) / 4.0)
        return cls(p1=p1, p2=p2, bulge=bulge)

    def approx_equal(self, other: object, tol: float = 1e-6) -> bool:
        return (
            isinstance(other, Arc)
            and _approx_point(self.p1, other.p1, tol)
            and _approx_point(self.p2, other.p2, tol)
            and math.isclose(self.bulge, other.bulge, abs_tol=tol)
        )


@dataclass(frozen=True)
class Circle:
    center: Point
    radius: float
    type: str = "circle"

    def to_dict(self) -> dict:
        return {"type": "circle", "center": list(self.center), "radius": self.radius}

    @staticmethod
    def from_dict(d: dict) -> "Circle":
        return Circle(center=tuple(d["center"]), radius=float(d["radius"]))

    def normalized(self, width: float, height: float) -> "Circle":
        return Circle(center=_norm_pt(self.center, width, height), radius=self.radius / width)

    def denormalized(self, width: float, height: float) -> "Circle":
        return Circle(center=_denorm_pt(self.center, width, height), radius=self.radius * width)

    def is_valid(self) -> bool:
        return self.radius > _EPS

    def approx_equal(self, other: object, tol: float = 1e-6) -> bool:
        return (
            isinstance(other, Circle)
            and _approx_point(self.center, other.center, tol)
            and math.isclose(self.radius, other.radius, abs_tol=tol)
        )


@dataclass(frozen=True)
class Bezier:
    control_points: Tuple[Point, ...]
    type: str = "bezier"

    def to_dict(self) -> dict:
        return {"type": "bezier", "control_points": [list(p) for p in self.control_points]}

    @staticmethod
    def from_dict(d: dict) -> "Bezier":
        return Bezier(control_points=tuple(tuple(p) for p in d["control_points"]))

    def normalized(self, width: float, height: float) -> "Bezier":
        return Bezier(control_points=tuple(_norm_pt(p, width, height) for p in self.control_points))

    def denormalized(self, width: float, height: float) -> "Bezier":
        return Bezier(control_points=tuple(_denorm_pt(p, width, height) for p in self.control_points))

    def is_valid(self) -> bool:
        # cubic Bezier; reject if every control point coincides (no extent)
        if len(self.control_points) != 4:
            return False
        first = self.control_points[0]
        return any(_dist(first, p) > _EPS for p in self.control_points[1:])

    def approx_equal(self, other: object, tol: float = 1e-6) -> bool:
        return (
            isinstance(other, Bezier)
            and len(self.control_points) == len(other.control_points)
            and all(_approx_point(a, b, tol) for a, b in zip(self.control_points, other.control_points))
        )


@dataclass(frozen=True)
class Ellipse:
    """Axis-aligned-and-rotated ellipse stored as (center, semi-axes, rotation).

    ``rotation`` is the angle (radians) of the semi-major axis from +x.
    Five floats matches the existing ``N_PARAMS = 5`` so the autoregressive
    model's parameter head doesn't need to grow when ellipse is added.

    The same geometric ellipse admits multiple parameterizations:

    * swapping (semi_major, semi_minor) and rotating by pi/2 yields the same
      ellipse;
    * rotating by pi yields the same ellipse (pi-symmetry).

    :meth:`canonical` collapses these to a unique form (``semi_major >=
    semi_minor`` and ``rotation in [0, pi)``) so the tokenizer always serializes
    the same shape identically -- load-bearing for the autoregressive bet, the
    same lesson the ``Arc`` endpoint-swap rule encoded.
    """

    center: Point
    semi_major: float
    semi_minor: float
    rotation: float                # radians; angle of the semi_major axis from +x
    type: str = "ellipse"

    def to_dict(self) -> dict:
        return {"type": "ellipse", "center": list(self.center),
                "semi_major": self.semi_major,
                "semi_minor": self.semi_minor,
                "rotation": self.rotation}

    @staticmethod
    def from_dict(d: dict) -> "Ellipse":
        return Ellipse(center=tuple(d["center"]),
                       semi_major=float(d["semi_major"]),
                       semi_minor=float(d["semi_minor"]),
                       rotation=float(d["rotation"]))

    def normalized(self, width: float, height: float) -> "Ellipse":
        # square-canvas assumption, consistent with Circle's radius normalization
        return Ellipse(center=_norm_pt(self.center, width, height),
                       semi_major=self.semi_major / width,
                       semi_minor=self.semi_minor / width,
                       rotation=self.rotation)

    def denormalized(self, width: float, height: float) -> "Ellipse":
        return Ellipse(center=_denorm_pt(self.center, width, height),
                       semi_major=self.semi_major * width,
                       semi_minor=self.semi_minor * width,
                       rotation=self.rotation)

    def is_valid(self) -> bool:
        return self.semi_major > _EPS and self.semi_minor > _EPS

    def canonical(self) -> "Ellipse":
        """Return the canonical representation: ``a >= b`` and ``rotation in [0, pi)``."""
        a, b, theta = self.semi_major, self.semi_minor, self.rotation
        if b > a:
            a, b = b, a
            theta = theta + math.pi / 2
        theta = theta % math.pi
        return Ellipse(center=self.center, semi_major=a, semi_minor=b, rotation=theta)

    def approx_equal(self, other: object, tol: float = 1e-6) -> bool:
        if not isinstance(other, Ellipse):
            return False
        a, b = self.canonical(), other.canonical()
        if not _approx_point(a.center, b.center, tol):
            return False
        if not math.isclose(a.semi_major, b.semi_major, abs_tol=tol):
            return False
        if not math.isclose(a.semi_minor, b.semi_minor, abs_tol=tol):
            return False
        # rotation lives on a circle of period pi: distance is min(d, pi - d)
        d_theta = abs(a.rotation - b.rotation) % math.pi
        return min(d_theta, math.pi - d_theta) < tol


_BY_TYPE = {"line": Line, "arc": Arc, "circle": Circle,
            "bezier": Bezier, "ellipse": Ellipse}


def primitive_from_dict(d: dict):
    try:
        cls = _BY_TYPE[d["type"]]
    except KeyError as exc:
        raise ValueError(f"unknown primitive type: {d.get('type')!r}") from exc
    return cls.from_dict(d)


# --- container -----------------------------------------------------------------

@dataclass
class PrimitiveSet:
    primitives: list

    def to_dict(self) -> dict:
        return {"primitives": [p.to_dict() for p in self.primitives]}

    @staticmethod
    def from_dict(d: dict) -> "PrimitiveSet":
        return PrimitiveSet([primitive_from_dict(p) for p in d["primitives"]])

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @staticmethod
    def from_json(s: str) -> "PrimitiveSet":
        return PrimitiveSet.from_dict(json.loads(s))

    def normalized(self, width: float, height: float) -> "PrimitiveSet":
        return PrimitiveSet([p.normalized(width, height) for p in self.primitives])

    def denormalized(self, width: float, height: float) -> "PrimitiveSet":
        return PrimitiveSet([p.denormalized(width, height) for p in self.primitives])

    def is_valid(self) -> bool:
        return all(p.is_valid() for p in self.primitives)

    def approx_equal(self, other: object, tol: float = 1e-6) -> bool:
        return (
            isinstance(other, PrimitiveSet)
            and len(self.primitives) == len(other.primitives)
            and all(a.approx_equal(b, tol) for a, b in zip(self.primitives, other.primitives))
        )


# --- helpers ------------------------------------------------------------------

def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _norm_pt(p: Point, width: float, height: float) -> Point:
    return (p[0] / width, p[1] / height)


def _denorm_pt(p: Point, width: float, height: float) -> Point:
    return (p[0] * width, p[1] * height)
