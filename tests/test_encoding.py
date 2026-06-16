"""Tests for the model encoding/decoding (Unit 5)."""

import numpy as np
import pytest

from lines.models.encoding import (
    ACTIVE_SLOTS, N_PARAMS, N_TYPES, TYPE_ARC, TYPE_CIRCLE, TYPE_LINE, TYPE_NONE,
    decode_primitive, decode_set, encode_primitive, encode_set,
)
from lines.primitives import Arc, Circle, Line, PrimitiveSet

W, H = 100, 100


def test_type_constants_are_contiguous_and_match_n_types():
    assert {TYPE_LINE, TYPE_ARC, TYPE_CIRCLE, TYPE_NONE} == set(range(N_TYPES))


def test_line_encodes_and_decodes():
    line = Line(p1=(10.0, 20.0), p2=(80.0, 70.0))
    t, p = encode_primitive(line, W, H)
    assert t == TYPE_LINE
    assert p.shape == (N_PARAMS,)
    assert p[:4].tolist() == pytest.approx([0.1, 0.2, 0.8, 0.7], abs=1e-6)
    assert decode_primitive(t, p, W, H).approx_equal(line, tol=1e-4)


def test_arc_round_trips_with_bulge():
    arc = Arc(p1=(10.0, 20.0), p2=(80.0, 70.0), bulge=-0.45)
    t, p = encode_primitive(arc, W, H)
    assert t == TYPE_ARC
    assert p[4] == np.float32(-0.45)
    assert decode_primitive(t, p, W, H).approx_equal(arc, tol=1e-4)


def test_circle_round_trips():
    circle = Circle(center=(40.0, 60.0), radius=15.0)
    t, p = encode_primitive(circle, W, H)
    assert t == TYPE_CIRCLE
    assert p[:3].tolist() == pytest.approx([0.4, 0.6, 0.15], abs=1e-6)
    assert decode_primitive(t, p, W, H).approx_equal(circle, tol=1e-4)


def test_encode_set_round_trip_through_decode():
    pset = PrimitiveSet([
        Line(p1=(0.0, 0.0), p2=(50.0, 50.0)),
        Circle(center=(25.0, 75.0), radius=10.0),
        Arc(p1=(60.0, 60.0), p2=(90.0, 30.0), bulge=0.3),
    ])
    types, params = encode_set(pset, W, H)
    restored = decode_set(types, params, W, H)
    assert restored.approx_equal(pset, tol=1e-4)


def test_active_slots_table_covers_every_concrete_type():
    assert TYPE_LINE in ACTIVE_SLOTS
    assert TYPE_ARC in ACTIVE_SLOTS
    assert TYPE_CIRCLE in ACTIVE_SLOTS
    assert TYPE_NONE not in ACTIVE_SLOTS
    for slots in ACTIVE_SLOTS.values():
        assert all(0 <= s < N_PARAMS for s in slots)


def test_decoding_none_yields_nothing():
    assert decode_primitive(TYPE_NONE, np.zeros(N_PARAMS, dtype=np.float32), W, H) is None


def test_decode_set_drops_none_queries():
    types = np.array([TYPE_LINE, TYPE_NONE, TYPE_CIRCLE], dtype=np.int64)
    params = np.zeros((3, N_PARAMS), dtype=np.float32)
    params[0, :4] = [0.1, 0.1, 0.5, 0.5]
    params[2, :3] = [0.5, 0.5, 0.1]
    out = decode_set(types, params, W, H)
    assert len(out.primitives) == 2
    assert {p.type for p in out.primitives} == {"line", "circle"}
