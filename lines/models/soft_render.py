"""Soft (differentiable-ish) renderer for set-prediction models.

This is a 'geometric' soft renderer that computes pixel intensities by
calculating the distance from each pixel to the nearest point on the
predicted primitives. It's used as an auxiliary loss term to force the
model to put 'ink' in the right places.

The distance functions are written in PyTorch to allow gradients to flow
back to the primitive parameters (endpoints, radius, bulge).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lines.models.encoding import TYPE_LINE, TYPE_ARC, TYPE_CIRCLE, TYPE_NONE


def soft_render_lines(
    coords: torch.Tensor,     # (B, N, 4) -> [x1, y1, x2, y2]
    grid: torch.Tensor,       # (1, 1, H, W, 2)
    sigma: float = 0.02,      # blur radius
) -> torch.Tensor:            # (B, N, H, W)
    # Extract endpoints
    p1 = coords[..., :2].unsqueeze(-2).unsqueeze(-2)  # (B, N, 1, 1, 2)
    p2 = coords[..., 2:4].unsqueeze(-2).unsqueeze(-2) # (B, N, 1, 1, 2)
    
    # Vector from p1 to p2
    v = p2 - p1
    l2 = (v * v).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    
    # Vector from p1 to pixel
    w = grid - p1
    
    # Projection factor clamped to [0, 1] for segment
    t = ((w * v).sum(dim=-1, keepdim=True) / l2).clamp(0, 1)
    
    # Closest point on segment
    closest = p1 + t * v
    
    # Squared distance to closest point
    dist2 = ((grid - closest) ** 2).sum(dim=-1)
    
    # Soft mask (Gaussian-like)
    return torch.exp(-dist2 / (2 * sigma**2))


def soft_render_circles(
    coords: torch.Tensor,     # (B, N, 3) -> [cx, cy, r]
    grid: torch.Tensor,       # (1, 1, H, W, 2)
    sigma: float = 0.02,
) -> torch.Tensor:
    center = coords[..., :2].unsqueeze(-2).unsqueeze(-2) # (B, N, 1, 1, 2)
    radius = coords[..., 2:3].unsqueeze(-2).unsqueeze(-2) # (B, N, 1, 1, 1)
    
    # Distance from center to pixel
    d = torch.sqrt(((grid - center) ** 2).sum(dim=-1, keepdim=True) + 1e-8)
    
    # Distance to circle boundary
    dist2 = (d - radius) ** 2
    dist2 = dist2.squeeze(-1)
    
    return torch.exp(-dist2 / (2 * sigma**2))


class SoftRenderer(torch.nn.Module):
    def __init__(self, canvas_size: int = 64, sigma: float = 0.02):
        super().__init__()
        self.canvas_size = canvas_size
        self.sigma = sigma
        
        # Create a coordinate grid [0, 1]
        # grid shape: (H, W, 2)
        y, x = torch.meshgrid(
            torch.linspace(0.5/canvas_size, 1.0 - 0.5/canvas_size, canvas_size),
            torch.linspace(0.5/canvas_size, 1.0 - 0.5/canvas_size, canvas_size),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1)
        self.register_buffer("grid", grid.unsqueeze(0).unsqueeze(0)) # (1, 1, H, W, 2)

    def forward(
        self,
        pred_logits: torch.Tensor,  # (B, N, K)
        pred_params: torch.Tensor,  # (B, N, P)
    ) -> torch.Tensor:              # (B, 1, H, W)
        B, N, K = pred_logits.shape
        probs = pred_logits.softmax(dim=-1) # (B, N, K)
        
        # We only render lines and circles for now (arcs are harder to do differentiably)
        # but we can approximate arcs as lines for the rendering loss or just skip them.
        # Since we want a general coverage loss, even a line approximation helps.
        
        # Render lines (and arcs-as-lines for now)
        line_mask = probs[..., [TYPE_LINE, TYPE_ARC]].sum(dim=-1) # (B, N)
        line_imgs = soft_render_lines(pred_params[..., :4], self.grid, self.sigma) # (B, N, H, W)
        line_contribution = (line_imgs * line_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        
        # Render circles
        circle_mask = probs[..., TYPE_CIRCLE] # (B, N)
        circle_imgs = soft_render_circles(pred_params[..., :3], self.grid, self.sigma)
        circle_contribution = (circle_imgs * circle_mask.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)
        
        # Final image is essentially a soft-OR (clamped sum)
        return (line_contribution + circle_contribution).clamp(0, 1)
