"""Tests for the 3D projection core (Stage 0).

Pure-geometry foundation: project a 3D mesh orthographically and determine which
edges are visible. No model, no rendering -- just exact projected segments that
will become 2D primitives in later stages.
"""

import math

import numpy as np

from lines.datagen.projection import (
    Camera, Mesh, box_mesh, edge_face_adjacency, face_normals, front_facing,
    fit_segments_to_canvas, project_visible_edges, random_rotation, rotate_mesh,
    visible_edges,
)
from lines.datagen.sampler2d import Canvas


# --- mesh ---------------------------------------------------------------------

def test_box_has_expected_counts():
    m = box_mesh(2.0, 2.0, 2.0)
    assert m.vertices.shape == (8, 3)
    assert len(m.edges) == 12
    assert len(m.faces) == 6


def test_box_face_normals_point_outward():
    m = box_mesh(2.0, 3.0, 4.0)
    normals = face_normals(m)
    for face, n in zip(m.faces, normals):
        centroid = m.vertices[list(face)].mean(axis=0)
        assert np.dot(n, centroid) > 0   # outward = same side as centroid from center


def test_every_box_edge_borders_exactly_two_faces():
    m = box_mesh(2.0, 2.0, 2.0)
    adj = edge_face_adjacency(m)
    assert len(adj) == 12
    assert all(len(faces) == 2 for faces in adj.values())


# --- camera / projection ------------------------------------------------------

def test_orthographic_projection_drops_depth_axis():
    cam = Camera.looking_from(direction=(0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0))
    p = np.array([[3.0, 4.0, 9.0]])
    xy = cam.project(p)
    assert np.allclose(xy[0], [3.0, 4.0])
    assert np.isclose(cam.depth(p)[0], 9.0)   # depth along the camera axis


def test_basis_is_orthonormal():
    cam = Camera.looking_from(direction=(-1.0, -2.0, -0.5))
    B = np.stack([cam.right, cam.up, cam.forward])
    assert np.allclose(B @ B.T, np.eye(3), atol=1e-9)


# --- visibility ---------------------------------------------------------------

def test_oblique_view_of_box_shows_nine_edges():
    # camera at (1,1,1): faces +x,+y,+z are front-facing -> 9 unique visible edges
    m = box_mesh(2.0, 2.0, 2.0)
    cam = Camera.looking_from(direction=(-1.0, -1.0, -1.0))
    ff = front_facing(m, cam)
    assert ff.sum() == 3
    assert len(visible_edges(m, cam)) == 9


def test_face_on_view_shows_four_edges():
    # looking straight down -z: only the +z face is front-facing -> its 4 edges
    m = box_mesh(2.0, 2.0, 2.0)
    cam = Camera.looking_from(direction=(0.0, 0.0, -1.0), up_hint=(0.0, 1.0, 0.0))
    assert front_facing(m, cam).sum() == 1
    assert len(visible_edges(m, cam)) == 4


def test_hidden_edges_excluded():
    m = box_mesh(2.0, 2.0, 2.0)
    cam = Camera.looking_from(direction=(-1.0, -1.0, -1.0))
    vis = set(map(frozenset, visible_edges(m, cam)))
    # the far corner is vertex (-,-,-) = index 0; its three edges are all hidden
    hidden_corner_edges = [frozenset((0, 1)), frozenset((0, 3)), frozenset((0, 4))]
    assert all(e not in vis for e in hidden_corner_edges)


def test_project_visible_edges_returns_segments():
    m = box_mesh(2.0, 2.0, 2.0)
    cam = Camera.looking_from(direction=(-1.0, -1.0, -1.0))
    segs = project_visible_edges(m, cam)
    assert len(segs) == 9
    for p1, p2 in segs:
        assert p1.shape == (2,) and p2.shape == (2,)


# --- rotation -----------------------------------------------------------------

def test_random_rotation_is_proper_orthogonal():
    R = random_rotation(np.random.default_rng(0))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotate_mesh_preserves_topology_and_shape():
    m = box_mesh(2.0, 2.0, 2.0)
    R = random_rotation(np.random.default_rng(1))
    rm = rotate_mesh(m, R)
    assert rm.vertices.shape == m.vertices.shape
    assert rm.edges == m.edges and rm.faces == m.faces
    # edge lengths are preserved under rotation
    a = np.linalg.norm(m.vertices[0] - m.vertices[1])
    b = np.linalg.norm(rm.vertices[0] - rm.vertices[1])
    assert np.isclose(a, b)


# --- fitting ------------------------------------------------------------------

def test_fit_segments_stays_within_canvas():
    m = box_mesh(2.0, 2.0, 2.0)
    cam = Camera.looking_from(direction=(-1.0, -1.3, -0.7))
    segs = project_visible_edges(m, cam)
    canvas = Canvas(128, 128)
    fitted = fit_segments_to_canvas(segs, canvas, margin=8.0)
    for p1, p2 in fitted:
        for x, y in (p1, p2):
            assert 0.0 <= x <= canvas.width and 0.0 <= y <= canvas.height
