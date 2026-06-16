"""Domain-randomization knobs for rendering.

v1 randomizes line weight (and keeps supersampling fixed for clean AA). Unit 7
expands this with resolution, contrast, and mild noise to widen the appearance
distribution and shrink the sim-to-real gap.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderParams:
    line_width: float
    supersample: int = 4


def sample_render_params(
    rng,
    line_width_range=(1.5, 3.0),
    supersample: int = 4,
) -> RenderParams:
    lo, hi = line_width_range
    return RenderParams(line_width=float(rng.uniform(lo, hi)), supersample=supersample)


def default_render_params() -> RenderParams:
    return RenderParams(line_width=2.0, supersample=4)
