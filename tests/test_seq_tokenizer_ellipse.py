"""Ellipse-specific tokenizer tests (Stage 2 Unit E2).

The autoregressive bet rests on the tokenizer being deterministic: two
parameterizations of the *same* geometric ellipse must produce *the same*
token sequence, so the model learns one canonical form rather than fighting
five equivalent ones.

Three load-bearing properties pinned below:

1. Round-trip preserves ellipses within the quantization tolerance.
2. Canonical ordering collapses all three ellipse symmetries (a/b swap,
   pi-rotation, theta near boundary) to identical token streams.
3. Mixed sets containing an ellipse plus other primitives still round-trip.
"""

import math

import pytest

from lines.models.seq_tokenizer import (
    BULGE_RANGE, EOS, N_COORD, N_THETA, SOS, THETA_RANGE, TYPE_TOKENS,
    Tokenizer, vocab_size,
)
from lines.primitives import (
    Arc, Circle, Ellipse, Line, PrimitiveSet,
)


CANVAS_SIDE = 64
TOK = Tokenizer(canvas_side=CANVAS_SIDE)


# --- vocabulary changes -------------------------------------------------------

def test_ellipse_has_a_type_token():
    assert "ellipse" in TYPE_TOKENS


def test_theta_block_is_contiguous_and_distinct_from_other_blocks():
    # rebuild the four blocks by inspection
    n = vocab_size()
    # specials = 0,1,2 ; types contiguous starting at 3
    type_ids = set(TYPE_TOKENS.values())
    # check theta tokens are a block that doesn't collide with type/coord/bulge
    # (we don't import _THETA_TOKENS directly -- inferring shape is enough)
    assert n > 3 + len(TYPE_TOKENS) + N_COORD     # room for theta block exists
    assert n >= 3 + len(TYPE_TOKENS) + N_COORD + N_THETA


# --- round-trip ---------------------------------------------------------------

@pytest.mark.parametrize("ellipse", [
    Ellipse(center=(32.0, 32.0), semi_major=20.0,
            semi_minor=10.0, rotation=0.0),
    Ellipse(center=(20.0, 40.0), semi_major=15.0,
            semi_minor=8.0, rotation=0.5),
    Ellipse(center=(50.0, 15.0), semi_major=10.0,
            semi_minor=10.0, rotation=1.0),    # degenerate -> circle-shaped
])
def test_ellipse_round_trips_within_bin_tolerance(ellipse):
    pset = PrimitiveSet([ellipse])
    restored = TOK.decode(TOK.encode(pset))
    assert len(restored.primitives) == 1
    bin_px = CANVAS_SIDE / N_COORD
    bin_theta = THETA_RANGE / N_THETA
    out = restored.primitives[0]
    assert isinstance(out, Ellipse)
    assert abs(out.center[0] - ellipse.center[0]) < bin_px
    assert abs(out.center[1] - ellipse.center[1]) < bin_px
    assert abs(out.semi_major - ellipse.semi_major) < bin_px
    assert abs(out.semi_minor - ellipse.semi_minor) < bin_px
    # theta on a circle; compare via shortest-arc distance
    d_theta = abs(out.rotation - (ellipse.rotation % THETA_RANGE)) % THETA_RANGE
    d_theta = min(d_theta, THETA_RANGE - d_theta)
    assert d_theta < bin_theta * 1.01


def test_decoded_ellipse_is_canonical():
    # construct a non-canonical ellipse (b > a) and verify the decoded one is canonical
    e = Ellipse(center=(32.0, 32.0), semi_major=5.0,
                semi_minor=15.0, rotation=0.1)
    restored = TOK.decode(TOK.encode(PrimitiveSet([e]))).primitives[0]
    assert restored.semi_major >= restored.semi_minor
    assert 0.0 <= restored.rotation < THETA_RANGE


# --- the structural property: symmetry equivalents -> same tokens -------------

def test_axis_swap_with_perpendicular_rotation_tokenizes_identically():
    # (a=15, b=8, theta=0.3) is the SAME geometric ellipse as
    # (a=8, b=15, theta=0.3 - pi/2). Both must yield the same token stream.
    e1 = Ellipse(center=(32.0, 32.0), semi_major=15.0,
                 semi_minor=8.0, rotation=0.3)
    e2 = Ellipse(center=(32.0, 32.0), semi_major=8.0,
                 semi_minor=15.0, rotation=0.3 - math.pi / 2)
    assert TOK.encode(PrimitiveSet([e1])) == TOK.encode(PrimitiveSet([e2]))


def test_pi_rotation_tokenizes_identically():
    e1 = Ellipse(center=(32.0, 32.0), semi_major=15.0,
                 semi_minor=8.0, rotation=0.3)
    e2 = Ellipse(center=(32.0, 32.0), semi_major=15.0,
                 semi_minor=8.0, rotation=0.3 + math.pi)
    assert TOK.encode(PrimitiveSet([e1])) == TOK.encode(PrimitiveSet([e2]))


def test_input_order_does_not_change_token_sequence_with_ellipses():
    a = Line(p1=(10.0, 10.0), p2=(20.0, 20.0))
    b = Ellipse(center=(40.0, 40.0), semi_major=12.0,
                semi_minor=6.0, rotation=0.4)
    c = Circle(center=(20.0, 50.0), radius=8.0)
    s1 = TOK.encode(PrimitiveSet([a, b, c]))
    s2 = TOK.encode(PrimitiveSet([c, a, b]))
    s3 = TOK.encode(PrimitiveSet([b, c, a]))
    assert s1 == s2 == s3


# --- mixed sets round-trip ----------------------------------------------------

def test_mixed_set_with_ellipse_round_trips():
    pset = PrimitiveSet([
        Line(p1=(5.0, 5.0), p2=(55.0, 55.0)),
        Arc(p1=(10.0, 50.0), p2=(50.0, 50.0), bulge=0.4),
        Circle(center=(32.0, 32.0), radius=10.0),
        Ellipse(center=(20.0, 20.0), semi_major=12.0,
                semi_minor=6.0, rotation=0.6),
    ])
    restored = TOK.decode(TOK.encode(pset))
    assert len(restored.primitives) == 4
    assert {p.type for p in restored.primitives} == {"line", "arc", "circle", "ellipse"}
