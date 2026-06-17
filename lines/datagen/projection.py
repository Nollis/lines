"""3D projection core (3D roadmap, Stage 0).

Pure-geometry foundation for the 3D data generator. We control the 3D model, so
we project it analytically and determine edge visibility directly -- no renderer
in the loop -- which yields exact 2D ground-truth segments for later stages.

Scope: orthographic projection + back-face-culling visibility for *convex*
solids (boxes now, cylinders later). General hidden-line removal for CSG parts
is a later stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Mesh:
    vertices: np.ndarray          # (V, 3)
    edges: List[Tuple[int, int]]  # undirected vertex-index pairs
    faces: List[Tuple[int, ...]]  # vertex-index loops, CCW seen from outside


@dataclass
class Camera:
    right: np.ndarray     # (3,) image +x
    up: np.ndarray        # (3,) image +y
    forward: np.ndarray   # (3,) unit vector pointing from the object toward the camera

    @classmethod
    def looking_from(cls, direction, up_hint=(0.0, 0.0, 1.0)) -> "Camera":
        """``direction`` is the view direction (camera -> object). The camera
        sits opposite it and looks back along ``-direction``."""
        d = np.asarray(direction, dtype=float)
        forward = -d / np.linalg.norm(d)
        up_hint = np.asarray(up_hint, dtype=float)
        if abs(np.dot(up_hint, forward)) > 0.99:        # nearly parallel -> pick another
            up_hint = np.array([0.0, 1.0, 0.0])
        right = np.cross(up_hint, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        return cls(right=right, up=up, forward=forward)

    def project(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return np.stack([pts @ self.right, pts @ self.up], axis=-1)

    def depth(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return pts @ self.forward      # larger = closer to the camera


# --- mesh construction --------------------------------------------------------

def box_mesh(w: float, h: float, d: float, center=(0.0, 0.0, 0.0)) -> Mesh:
    cx, cy, cz = center
    hw, hh, hd = w / 2, h / 2, d / 2
    verts = np.array([
        [cx - hw, cy - hh, cz - hd],  # 0 (-,-,-)
        [cx + hw, cy - hh, cz - hd],  # 1 (+,-,-)
        [cx + hw, cy + hh, cz - hd],  # 2 (+,+,-)
        [cx - hw, cy + hh, cz - hd],  # 3 (-,+,-)
        [cx - hw, cy - hh, cz + hd],  # 4 (-,-,+)
        [cx + hw, cy - hh, cz + hd],  # 5 (+,-,+)
        [cx + hw, cy + hh, cz + hd],  # 6 (+,+,+)
        [cx - hw, cy + hh, cz + hd],  # 7 (-,+,+)
    ], dtype=float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0),     # bottom (z-)
             (4, 5), (5, 6), (6, 7), (7, 4),     # top (z+)
             (0, 4), (1, 5), (2, 6), (3, 7)]     # verticals
    faces = [
        (0, 3, 2, 1),   # z-  (normal -z)
        (4, 5, 6, 7),   # z+  (normal +z)
        (0, 1, 5, 4),   # y-  (normal -y)
        (3, 7, 6, 2),   # y+  (normal +y)
        (0, 4, 7, 3),   # x-  (normal -x)
        (1, 2, 6, 5),   # x+  (normal +x)
    ]
    return Mesh(vertices=verts, edges=edges, faces=faces)


# --- geometry -----------------------------------------------------------------

def face_normals(mesh: Mesh) -> np.ndarray:
    """Outward unit normal per face via Newell's method (winding-consistent)."""
    normals = []
    for face in mesh.faces:
        loop = mesh.vertices[list(face)]
        n = np.zeros(3)
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            n[0] += (a[1] - b[1]) * (a[2] + b[2])
            n[1] += (a[2] - b[2]) * (a[0] + b[0])
            n[2] += (a[0] - b[0]) * (a[1] + b[1])
        normals.append(n / np.linalg.norm(n))
    return np.array(normals)


def edge_face_adjacency(mesh: Mesh) -> Dict[FrozenSet[int], List[int]]:
    adj: Dict[FrozenSet[int], List[int]] = {frozenset(e): [] for e in mesh.edges}
    for fi, face in enumerate(mesh.faces):
        for i in range(len(face)):
            key = frozenset((face[i], face[(i + 1) % len(face)]))
            if key in adj:
                adj[key].append(fi)
    return adj


def front_facing(mesh: Mesh, camera: Camera) -> np.ndarray:
    """Boolean per face: True if its outward normal faces the camera."""
    return face_normals(mesh) @ camera.forward > 1e-9


def visible_edges(mesh: Mesh, camera: Camera) -> List[Tuple[int, int]]:
    """Edges adjacent to at least one front-facing face (convex-solid rule)."""
    ff = front_facing(mesh, camera)
    adj = edge_face_adjacency(mesh)
    out = []
    for edge in mesh.edges:
        faces = adj[frozenset(edge)]
        if any(ff[fi] for fi in faces):
            out.append(edge)
    return out


def silhouette_edges(mesh: Mesh, camera: Camera) -> List[Tuple[int, int]]:
    """Edges between exactly one front- and one back-facing face (the outline)."""
    ff = front_facing(mesh, camera)
    adj = edge_face_adjacency(mesh)
    out = []
    for edge in mesh.edges:
        faces = adj[frozenset(edge)]
        if len(faces) == 2 and (ff[faces[0]] != ff[faces[1]]):
            out.append(edge)
    return out


def project_visible_edges(mesh: Mesh, camera: Camera) -> List[Tuple[np.ndarray, np.ndarray]]:
    segs = []
    for i, j in visible_edges(mesh, camera):
        p1 = camera.project(mesh.vertices[i])[0]
        p2 = camera.project(mesh.vertices[j])[0]
        segs.append((p1, p2))
    return segs


# --- transforms ---------------------------------------------------------------

def random_rotation(rng) -> np.ndarray:
    return Rotation.random(random_state=rng).as_matrix()


def rotate_mesh(mesh: Mesh, R: np.ndarray) -> Mesh:
    return Mesh(vertices=mesh.vertices @ R.T, edges=mesh.edges, faces=mesh.faces)


def fit_segments_to_canvas(segments, canvas, margin: float = 8.0):
    """Scale + center projected segments into the canvas (uniform scale, y down)."""
    pts = np.array([p for seg in segments for p in seg])
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    extent = np.maximum(hi - lo, 1e-6)
    avail = min(canvas.width, canvas.height) - 2 * margin
    scale = avail / extent.max()
    offset = np.array([canvas.width, canvas.height]) / 2 - (lo + hi) / 2 * scale

    out = []
    for p1, p2 in segments:
        q1 = p1 * scale + offset
        q2 = p2 * scale + offset
        # flip y so the projection reads naturally in image coordinates (y down)
        q1 = np.array([q1[0], canvas.height - q1[1]])
        q2 = np.array([q2[0], canvas.height - q2[1]])
        out.append((q1, q2))
    return out
