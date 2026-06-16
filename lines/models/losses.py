"""DETR-style set-prediction loss.

For each image:

1. The Hungarian matcher assigns N predicted queries to M ground-truth
   primitives (M can be 0).
2. Type cross-entropy is computed for *every* query: matched queries get the
   GT type; unmatched queries get the "none" label so the model learns to
   predict emptiness on unused slots. A class weight downweights "none" so
   the easy majority class doesn't drown out the rare-but-important matched
   queries.
3. Parameter L1 loss is computed only on matched queries, only on the slots
   the matched GT type actually uses (``ACTIVE_SLOTS``). Padded slots are
   ignored so they cannot be tuned away from zero by an artificially low loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lines.models.encoding import ACTIVE_SLOTS, N_PARAMS, N_TYPES, TYPE_NONE, TYPE_ARC
from lines.models.matcher import HungarianMatcher
from lines.models.soft_render import SoftRenderer


@dataclass
class LossOutputs:
    loss: torch.Tensor
    loss_class: torch.Tensor
    loss_param: torch.Tensor
    loss_render: torch.Tensor
    n_matched: int


class SetPredictionLoss(nn.Module):
    def __init__(
        self,
        matcher: HungarianMatcher | None = None,
        class_weight_none: float = 0.1,
        param_weight: float = 5.0,
        render_weight: float = 2.0,
        canvas_size: int = 64,
        render_canvas_size: int | None = None,
    ):
        super().__init__()
        self.matcher = matcher or HungarianMatcher()
        self.param_weight = param_weight
        self.render_weight = render_weight
        self.render_canvas_size = render_canvas_size or canvas_size
        self.renderer = SoftRenderer(canvas_size=self.render_canvas_size)
        weights = torch.ones(N_TYPES)
        weights[TYPE_NONE] = class_weight_none
        self.register_buffer("class_weights", weights)

    def forward(
        self,
        pred_logits: torch.Tensor,    # (B, N, K)
        pred_params: torch.Tensor,    # (B, N, P)
        gt_types_list,                # list of (M_b,) long tensors
        gt_params_list,               # list of (M_b, P) float tensors
        gt_images: torch.Tensor | None = None, # (B, 1, H, W)
    ) -> LossOutputs:
        B, N, _ = pred_logits.shape
        device = pred_logits.device

        target_types = torch.full((B, N), TYPE_NONE, dtype=torch.long, device=device)
        matched_pred_slots = []
        matched_gt_params = []
        matched_gt_types = []
        total_matched = 0

        for b in range(B):
            gt_t = gt_types_list[b].to(device)
            gt_p = gt_params_list[b].to(device)
            pred_idx, gt_idx = self.matcher(pred_logits[b], pred_params[b], gt_t, gt_p)
            if pred_idx.numel() == 0:
                continue
            target_types[b, pred_idx] = gt_t[gt_idx]
            matched_pred_slots.append(pred_params[b, pred_idx])      # (m, P)
            matched_gt_params.append(gt_p[gt_idx])                    # (m, P)
            matched_gt_types.append(gt_t[gt_idx])                     # (m,)
            total_matched += pred_idx.numel()

        loss_class = F.cross_entropy(
            pred_logits.reshape(-1, N_TYPES),
            target_types.reshape(-1),
            weight=self.class_weights.to(device),
        )

        if total_matched == 0:
            loss_param = pred_params.sum() * 0.0   # keeps autograd graph alive
        else:
            preds = torch.cat(matched_pred_slots, dim=0)               # (M_all, P)
            gts = torch.cat(matched_gt_params, dim=0)                  # (M_all, P)
            types = torch.cat(matched_gt_types, dim=0)                 # (M_all,)

            # Symmetrized loss calculation
            # For lines and arcs, [p1x, p1y, p2x, p2y] can be swapped.
            # For arcs, swapping endpoints requires negating the bulge (index 4).
            gts_flipped = gts.clone()
            gts_flipped[:, [0, 1, 2, 3]] = gts[:, [2, 3, 0, 1]]
            
            # Negate bulge for arcs when flipped
            arc_mask = (types == TYPE_ARC)
            gts_flipped[arc_mask, 4] = -gts[arc_mask, 4]

            dist_orig = (preds - gts).abs()
            dist_flipped = (preds - gts_flipped).abs()
            
            # Determine which orientation is closer for each match
            # We only consider the slots that are actually active for the given type
            mask = torch.zeros_like(preds)
            for i, t in enumerate(types.tolist()):
                slots = ACTIVE_SLOTS.get(int(t))
                if slots:
                    mask[i, list(slots)] = 1.0
            
            # We use the sum of absolute differences over active slots to decide orientation
            error_orig = (dist_orig * mask).sum(dim=-1)
            error_flipped = (dist_flipped * mask).sum(dim=-1)
            use_flipped = (error_flipped < error_orig).float().unsqueeze(-1)
            
            best_dist = dist_orig * (1.0 - use_flipped) + dist_flipped * use_flipped
            loss_param = (best_dist * mask).sum() / mask.sum().clamp_min(1.0)

        # Auxiliary rendering loss
        loss_render = torch.tensor(0.0, device=device)
        if self.render_weight > 0 and gt_images is not None:
            # Our input image is 0=fg, 255=bg (inverted). SoftRenderer produces 1=fg, 0=bg.
            # Convert gt_images to [0, 1] where 1 is fg.
            gt_fg = 1.0 - gt_images
            if self.render_canvas_size != gt_fg.shape[-1]:
                gt_fg = F.interpolate(gt_fg, size=(self.render_canvas_size, self.render_canvas_size), mode='area')
            pred_render = self.renderer(pred_logits, pred_params)
            loss_render = F.mse_loss(pred_render, gt_fg)

        loss = loss_class + self.param_weight * loss_param + self.render_weight * loss_render
        return LossOutputs(loss=loss, loss_class=loss_class, loss_param=loss_param,
                           loss_render=loss_render, n_matched=total_matched)
