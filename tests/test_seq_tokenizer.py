"""Tests for the autoregressive-prototype tokenizer (A2).

Load-bearing properties under TDD:

1. Round-trip is exact -- serialize -> parse reconstructs the primitive set
   within the quantization tolerance.
2. Canonical ordering is deterministic -- the same primitive set always
   serializes to identical tokens, regardless of input order or endpoint
   swap.
3. Shared coordinates land in the same coord token -- a corner used by two
   lines is the *same* token both times (the structural property the whole
   autoregressive bet rests on).
4. Token vocabulary is consistent and bounded.

These are all CPU-cheap, pure-Python checks. The full autoregressive model is
worthless if these are wrong, so they get nailed down first.
"""

import pytest

from lines.models.seq_tokenizer import (
    BULGE_RANGE, COORD_TOKENS, EOS, N_BULGE, N_COORD,
    PAD, SOS, TYPE_TOKENS, Tokenizer, vocab_size,
)
from lines.primitives import Arc, Circle, Line, PrimitiveSet


CANVAS_SIDE = 64
TOK = Tokenizer(canvas_side=CANVAS_SIDE)


def test_special_and_type_tokens_are_distinct():
    ids = [SOS, EOS, PAD] + list(TYPE_TOKENS.values())
    assert len(set(ids)) == len(ids)


def test_vocab_size_accounts_for_every_class_of_token():
    n = vocab_size()
    # 3 special + 3 type + N_COORD coord + N_BULGE bulge
    assert n == 3 + len(TYPE_TOKENS) + N_COORD + N_BULGE


def test_coord_tokens_form_a_contiguous_block():
    block = sorted(COORD_TOKENS)
    assert block == list(range(block[0], block[0] + N_COORD))


# --- round-trips --------------------------------------------------------------

@pytest.mark.parametrize("prim", [
    Line(p1=(10.0, 10.0), p2=(50.0, 50.0)),
    Circle(center=(32.0, 32.0), radius=12.0),
    Arc(p1=(8.0, 32.0), p2=(56.0, 32.0), bulge=0.4),
])
def test_single_primitive_round_trips(prim):
    pset = PrimitiveSet([prim])
    tokens = TOK.encode(pset)
    restored = TOK.decode(tokens)
    assert len(restored.primitives) == 1
    bin_px = CANVAS_SIDE / N_COORD
    assert restored.primitives[0].approx_equal(prim, tol=bin_px * 1.01)


def test_mixed_primitive_set_round_trips():
    pset = PrimitiveSet([
        Line(p1=(8.0, 8.0), p2=(56.0, 8.0)),
        Line(p1=(56.0, 8.0), p2=(56.0, 56.0)),
        Circle(center=(32.0, 32.0), radius=10.0),
        Arc(p1=(10.0, 30.0), p2=(30.0, 10.0), bulge=0.3),
    ])
    restored = TOK.decode(TOK.encode(pset))
    assert len(restored.primitives) == 4
    assert {p.type for p in restored.primitives} == {"line", "arc", "circle"}


def test_encoding_starts_with_sos_and_ends_with_eos():
    tokens = TOK.encode(PrimitiveSet([Line(p1=(10.0, 10.0), p2=(50.0, 50.0))]))
    assert tokens[0] == SOS
    assert tokens[-1] == EOS


def test_empty_set_round_trips():
    tokens = TOK.encode(PrimitiveSet([]))
    assert tokens == [SOS, EOS]
    assert TOK.decode(tokens).primitives == []


# --- canonical ordering -------------------------------------------------------

def test_input_order_does_not_change_token_sequence():
    a = Line(p1=(10.0, 10.0), p2=(20.0, 20.0))
    b = Circle(center=(40.0, 40.0), radius=8.0)
    c = Line(p1=(30.0, 5.0), p2=(50.0, 50.0))
    s1 = TOK.encode(PrimitiveSet([a, b, c]))
    s2 = TOK.encode(PrimitiveSet([c, a, b]))
    s3 = TOK.encode(PrimitiveSet([b, c, a]))
    assert s1 == s2 == s3


def test_swapped_endpoints_canonicalize_to_same_tokens():
    a = Line(p1=(10.0, 10.0), p2=(50.0, 30.0))
    b = Line(p1=(50.0, 30.0), p2=(10.0, 10.0))   # same line, swapped
    assert TOK.encode(PrimitiveSet([a])) == TOK.encode(PrimitiveSet([b]))


# --- the structural property (what the autoregressive bet rests on) -----------

def test_shared_corner_uses_the_same_coord_tokens_twice():
    # two lines that share an endpoint at (40, 40) must emit (40, 40) as the
    # SAME token in both places -- this is what lets an autoregressive model
    # "remember" the corner and re-emit it exactly.
    shared = (40.0, 40.0)
    a = Line(p1=(10.0, 10.0), p2=shared)
    b = Line(p1=shared, p2=(50.0, 8.0))
    tokens = TOK.encode(PrimitiveSet([a, b]))
    # the quantized version of `shared` must appear at least twice in the token
    # stream (the two endpoints referring to the same corner)
    x_tok = TOK.quantize_coord(shared[0])
    y_tok = TOK.quantize_coord(shared[1])
    assert tokens.count(x_tok) >= 2
    assert tokens.count(y_tok) >= 2


def test_two_decoded_lines_at_a_shared_corner_actually_share_their_endpoint():
    shared = (40.0, 40.0)
    a = Line(p1=(10.0, 10.0), p2=shared)
    b = Line(p1=shared, p2=(50.0, 8.0))
    restored = TOK.decode(TOK.encode(PrimitiveSet([a, b])))
    # find the two lines and the endpoint each one has near `shared`
    near = []
    for line in restored.primitives:
        for end in (line.p1, line.p2):
            if abs(end[0] - shared[0]) < 2 and abs(end[1] - shared[1]) < 2:
                near.append(end)
    assert len(near) == 2
    # the two corner endpoints must be the SAME point post-decode
    assert near[0] == near[1]


# --- bulge --------------------------------------------------------------------

def test_bulge_round_trip_is_within_one_bin():
    bulge = 0.55
    pset = PrimitiveSet([Arc(p1=(10.0, 32.0), p2=(54.0, 32.0), bulge=bulge)])
    restored = TOK.decode(TOK.encode(pset))
    bin_w = 2 * BULGE_RANGE / N_BULGE
    assert abs(restored.primitives[0].bulge - bulge) < bin_w * 1.01


# --- coord quantization ------------------------------------------------------

def test_quantize_then_dequantize_is_within_half_a_bin():
    bin_px = CANVAS_SIDE / N_COORD
    for v in (0.0, 10.0, 31.5, 50.0, float(CANVAS_SIDE - 0.01)):
        tok = TOK.quantize_coord(v)
        back = TOK.dequantize_coord(tok)
        assert abs(back - v) <= bin_px / 2 + 1e-6


def test_quantize_clamps_out_of_range():
    # values outside [0, canvas) must clamp rather than throw
    TOK.quantize_coord(-5.0)
    TOK.quantize_coord(CANVAS_SIDE + 5.0)
