"""Convert between :class:`PrimitiveSet` and the flat tensor encoding used by
the set-prediction model.

Encoding (5 param slots + a type label):

* type ``0``: line   -> params = ``[p1.x, p1.y, p2.x,  p2.y, 0]``  (4 slots used)
* type ``1``: arc    -> params = ``[p1.x, p1.y, p2.x,  p2.y, bulge]`` (5)
* type ``2``: circle -> params = ``[cx,   cy,   r,     0,    0]``  (3)
* type ``3``: none / empty query

Coordinates are normalized to ``[0, 1]`` against the canvas. Bulge is stored
raw; the v1 sampler caps arc sweep at 270 degrees so the magnitude stays well
below the full-circle blow-up regime (rough range +/- 2.4).
"""

from __future__ import annotations

import numpy as np

from lines.primitives import Arc, Circle, Line, PrimitiveSet

N_PARAMS = 5
N_TYPES = 4  # line, arc, circle, none
TYPE_LINE = 0
TYPE_ARC = 1
TYPE_CIRCLE = 2
TYPE_NONE = 3

_TYPE_BY_NAME = {"line": TYPE_LINE, "arc": TYPE_ARC, "circle": TYPE_CIRCLE}
_NAME_BY_TYPE = {v: k for k, v in _TYPE_BY_NAME.items()}

# which param slots actually carry information per type -- everything else is
# slack the loss must ignore.
ACTIVE_SLOTS = {
    TYPE_LINE:   (0, 1, 2, 3),
    TYPE_ARC:    (0, 1, 2, 3, 4),
    TYPE_CIRCLE: (0, 1, 2),
}


def encode_primitive(prim, width: float, height: float):
    """Return ``(type_id, params[5])`` for one primitive."""
    params = np.zeros(N_PARAMS, dtype=np.float32)
    if isinstance(prim, Line):
        params[:4] = [prim.p1[0] / width, prim.p1[1] / height,
                      prim.p2[0] / width, prim.p2[1] / height]
        return TYPE_LINE, params
    if isinstance(prim, Arc):
        params[:5] = [prim.p1[0] / width, prim.p1[1] / height,
                      prim.p2[0] / width, prim.p2[1] / height, prim.bulge]
        return TYPE_ARC, params
    if isinstance(prim, Circle):
        params[:3] = [prim.center[0] / width, prim.center[1] / height,
                      prim.radius / width]
        return TYPE_CIRCLE, params
    raise TypeError(f"cannot encode primitive of type {type(prim).__name__}")


def encode_set(pset: PrimitiveSet, width: float, height: float):
    """Encode a primitive set as ``(types[k], params[k, 5])`` numpy arrays."""
    if not pset.primitives:
        return np.zeros((0,), dtype=np.int64), np.zeros((0, N_PARAMS), dtype=np.float32)
    types, params = [], []
    for prim in pset.primitives:
        t, p = encode_primitive(prim, width, height)
        types.append(t)
        params.append(p)
    return np.asarray(types, dtype=np.int64), np.stack(params).astype(np.float32)


def decode_primitive(type_id: int, params, width: float, height: float):
    """Inverse of :func:`encode_primitive`. ``params`` is length-5."""
    name = _NAME_BY_TYPE.get(int(type_id))
    if name is None:
        return None
    if name == "line":
        return Line(p1=(float(params[0]) * width, float(params[1]) * height),
                    p2=(float(params[2]) * width, float(params[3]) * height))
    if name == "arc":
        return Arc(p1=(float(params[0]) * width, float(params[1]) * height),
                   p2=(float(params[2]) * width, float(params[3]) * height),
                   bulge=float(params[4]))
    return Circle(center=(float(params[0]) * width, float(params[1]) * height),
                  radius=float(params[2]) * width)


def decode_set(types, params, width: float, height: float,
               drop_none: bool = True) -> PrimitiveSet:
    """Inverse of :func:`encode_set`. ``params`` has shape ``(k, 5)``."""
    prims = []
    for t, p in zip(types, params):
        if drop_none and int(t) == TYPE_NONE:
            continue
        prim = decode_primitive(int(t), p, width, height)
        if prim is not None and prim.is_valid():
            prims.append(prim)
    return PrimitiveSet(prims)
