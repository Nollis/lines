"""Tokenizer for the autoregressive prototype (plan Unit A2).

Serializes a :class:`PrimitiveSet` to a deterministic integer token sequence and
parses it back. This is the load-bearing piece of the autoregressive bet: if the
canonical ordering is non-deterministic, or shared corners don't land on the
same token, the model has nothing to learn.

Vocabulary
----------
* Specials: ``SOS``, ``EOS``, ``PAD``.
* Type tags: ``LINE``, ``ARC``, ``CIRCLE``.
* Coords: a single block of ``N_COORD`` tokens (reused for x and y), one per
  quantization bin spanning ``[0, canvas_side)``.
* Bulge: ``N_BULGE`` tokens spanning ``[-BULGE_RANGE, +BULGE_RANGE]``.

Sequence layout per primitive
-----------------------------
* ``LINE x1 y1 x2 y2``                (5 tokens)
* ``ARC  x1 y1 x2 y2 bulge``          (6 tokens)
* ``CIRCLE cx cy r``                  (4 tokens; r is encoded in coord bins)

The full sequence is ``[SOS, <prims in canonical order>, EOS]``.

Canonical ordering
------------------
Within a primitive, endpoint-symmetric types (Line, Arc) are normalized: the
"smaller" endpoint comes first under lex order, with the arc's bulge sign
flipped when its endpoints are swapped. Across primitives, sets are sorted by
``(type_id, *quantized_params)`` -- so two equal sets always emit identical
token sequences regardless of input order.
"""

from __future__ import annotations

from typing import List

from lines.primitives import Arc, Circle, Ellipse, Line, PrimitiveSet

# --- vocabulary ---------------------------------------------------------------

# 3 special tokens
SOS = 0
EOS = 1
PAD = 2

# 4 type tokens (Stage 2 adds ellipse)
TYPE_TOKENS = {"line": 3, "arc": 4, "circle": 5, "ellipse": 6}
_TYPE_BY_ID = {v: k for k, v in TYPE_TOKENS.items()}

# 64 coord bins by default (one bin per pixel at canvas_side=64)
N_COORD = 64
_COORD_BASE = 3 + len(TYPE_TOKENS)         # derived: skips specials + types
COORD_TOKENS = list(range(_COORD_BASE, _COORD_BASE + N_COORD))

# 64 bulge bins spanning [-BULGE_RANGE, BULGE_RANGE]. v1 sampler caps arc sweep
# at 270deg so |bulge| stays under ~2.4 (bulge = tan(theta/4), theta in radians).
N_BULGE = 64
BULGE_RANGE = 2.5
_BULGE_BASE = _COORD_BASE + N_COORD
_BULGE_TOKENS = list(range(_BULGE_BASE, _BULGE_BASE + N_BULGE))

# 64 ellipse-rotation bins spanning [0, THETA_RANGE) = [0, pi). Ellipse.canonical
# already maps rotation into this range, so quantization is straightforward.
import math as _math
N_THETA = 64
THETA_RANGE = _math.pi
_THETA_BASE = _BULGE_BASE + N_BULGE
_THETA_TOKENS = list(range(_THETA_BASE, _THETA_BASE + N_THETA))


def vocab_size() -> int:
    return 3 + len(TYPE_TOKENS) + N_COORD + N_BULGE + N_THETA


# --- tokenizer ----------------------------------------------------------------

class Tokenizer:
    def __init__(self, canvas_side: int = 64):
        self.canvas_side = canvas_side

    # -- coord quantization
    def quantize_coord(self, v: float) -> int:
        v = max(0.0, min(float(v), self.canvas_side - 1e-6))
        idx = int(v * N_COORD / self.canvas_side)
        idx = max(0, min(idx, N_COORD - 1))
        return COORD_TOKENS[idx]

    def dequantize_coord(self, token: int) -> float:
        idx = token - _COORD_BASE
        if idx < 0 or idx >= N_COORD:
            raise ValueError(f"token {token} is not a coord token")
        # bin center
        return (idx + 0.5) * self.canvas_side / N_COORD

    # -- bulge quantization
    def quantize_bulge(self, b: float) -> int:
        clamped = max(-BULGE_RANGE, min(float(b), BULGE_RANGE - 1e-9))
        idx = int((clamped + BULGE_RANGE) * N_BULGE / (2 * BULGE_RANGE))
        idx = max(0, min(idx, N_BULGE - 1))
        return _BULGE_TOKENS[idx]

    def dequantize_bulge(self, token: int) -> float:
        idx = token - _BULGE_BASE
        if idx < 0 or idx >= N_BULGE:
            raise ValueError(f"token {token} is not a bulge token")
        return -BULGE_RANGE + (idx + 0.5) * 2 * BULGE_RANGE / N_BULGE

    # -- theta quantization (ellipse rotation, range [0, pi))
    def quantize_theta(self, t: float) -> int:
        # Ellipse.canonical maps theta into [0, pi); clamp defensively
        clamped = float(t) % THETA_RANGE
        idx = int(clamped * N_THETA / THETA_RANGE)
        idx = max(0, min(idx, N_THETA - 1))
        return _THETA_TOKENS[idx]

    def dequantize_theta(self, token: int) -> float:
        idx = token - _THETA_BASE
        if idx < 0 or idx >= N_THETA:
            raise ValueError(f"token {token} is not a theta token")
        return (idx + 0.5) * THETA_RANGE / N_THETA

    # -- encode
    def encode(self, pset: PrimitiveSet) -> List[int]:
        canon = [self._canonicalize(p) for p in pset.primitives]
        # sort across primitives by (type_id, quantized params)
        def sort_key(prim):
            type_id = TYPE_TOKENS[prim.type]
            return (type_id,) + self._quantized_params(prim)
        canon.sort(key=sort_key)

        tokens = [SOS]
        for prim in canon:
            tokens.append(TYPE_TOKENS[prim.type])
            tokens.extend(self._param_tokens(prim))
        tokens.append(EOS)
        return tokens

    # -- decode
    def decode(self, tokens: List[int]) -> PrimitiveSet:
        if not tokens or tokens[0] != SOS:
            raise ValueError("token stream must start with SOS")
        i = 1
        prims = []
        while i < len(tokens) and tokens[i] != EOS:
            tok = tokens[i]
            kind = _TYPE_BY_ID.get(tok)
            if kind is None:
                # graceful skip on unexpected token (autoregressive may emit junk)
                i += 1
                continue
            try:
                if kind == "line":
                    x1, y1, x2, y2 = (self.dequantize_coord(t) for t in tokens[i + 1:i + 5])
                    prims.append(Line(p1=(x1, y1), p2=(x2, y2)))
                    i += 5
                elif kind == "arc":
                    x1, y1, x2, y2 = (self.dequantize_coord(t) for t in tokens[i + 1:i + 5])
                    bulge = self.dequantize_bulge(tokens[i + 5])
                    prims.append(Arc(p1=(x1, y1), p2=(x2, y2), bulge=bulge))
                    i += 6
                elif kind == "circle":
                    cx, cy, r = (self.dequantize_coord(t) for t in tokens[i + 1:i + 4])
                    prims.append(Circle(center=(cx, cy), radius=r))
                    i += 4
                else:  # ellipse: TYPE_ELLIPSE cx cy a b theta (6 tokens)
                    cx, cy, a, b = (self.dequantize_coord(t) for t in tokens[i + 1:i + 5])
                    theta = self.dequantize_theta(tokens[i + 5])
                    prims.append(Ellipse(center=(cx, cy), semi_major=a,
                                         semi_minor=b, rotation=theta))
                    i += 6
            except (ValueError, IndexError):
                i += 1   # malformed primitive -> skip
        return PrimitiveSet([p for p in prims if p.is_valid()])

    # -- helpers
    def _canonicalize(self, prim):
        if isinstance(prim, Line):
            return Line(*sorted([prim.p1, prim.p2]))
        if isinstance(prim, Arc):
            if prim.p2 < prim.p1:                # swap -> flip bulge
                return Arc(p1=prim.p2, p2=prim.p1, bulge=-prim.bulge)
            return prim
        if isinstance(prim, Ellipse):
            return prim.canonical()
        return prim

    def _param_tokens(self, prim) -> list:
        if isinstance(prim, Line):
            return [self.quantize_coord(prim.p1[0]), self.quantize_coord(prim.p1[1]),
                    self.quantize_coord(prim.p2[0]), self.quantize_coord(prim.p2[1])]
        if isinstance(prim, Arc):
            return [self.quantize_coord(prim.p1[0]), self.quantize_coord(prim.p1[1]),
                    self.quantize_coord(prim.p2[0]), self.quantize_coord(prim.p2[1]),
                    self.quantize_bulge(prim.bulge)]
        if isinstance(prim, Ellipse):
            return [self.quantize_coord(prim.center[0]), self.quantize_coord(prim.center[1]),
                    self.quantize_coord(prim.semi_major), self.quantize_coord(prim.semi_minor),
                    self.quantize_theta(prim.rotation)]
        return [self.quantize_coord(prim.center[0]), self.quantize_coord(prim.center[1]),
                self.quantize_coord(prim.radius)]

    def _quantized_params(self, prim) -> tuple:
        return tuple(self._param_tokens(prim))
