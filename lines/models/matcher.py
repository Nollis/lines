"""Hungarian matcher: assign predicted queries to ground-truth primitives.

The matcher's role is to decide WHICH query learns WHICH primitive on each
training example. Without it, set prediction degenerates because the loss has
no canonical ordering across the unordered ground-truth set.

Cost is the sum of two terms (DETR-style):

* ``-prob(gt_type)`` for each (query, gt) pair -- prefer queries already
  inclined toward the right type.
* L1 distance between the predicted params and the GT params, restricted to
  the slots the GT type actually uses (``ACTIVE_SLOTS``). Padded slots are
  ignored so they cannot fake an artificially good match.

The assignment is found with ``scipy.optimize.linear_sum_assignment``; the
returned index tensors say "pred query ``pred_idx[k]`` was matched to gt
primitive ``gt_idx[k]``".
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from lines.models.encoding import ACTIVE_SLOTS


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1.0, cost_param: float = 5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_param = cost_param

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,   # (N, K)
        pred_params: torch.Tensor,   # (N, P)
        gt_types: torch.Tensor,      # (M,)
        gt_params: torch.Tensor,     # (M, P)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N = pred_logits.shape[0]
        M = gt_types.shape[0]
        if M == 0 or N == 0:
            return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))

        probs = pred_logits.softmax(dim=-1)            # (N, K)
        class_cost = -probs[:, gt_types]               # (N, M)

        param_cost = torch.zeros(N, M, device=pred_params.device)
        for j in range(M):
            t = int(gt_types[j].item())
            slots = ACTIVE_SLOTS.get(t)
            if not slots:
                continue
            diff = (pred_params[:, list(slots)] - gt_params[j, list(slots)]).abs()
            param_cost[:, j] = diff.sum(dim=-1)

        cost = self.cost_class * class_cost + self.cost_param * param_cost
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        return torch.as_tensor(rows, dtype=torch.long), torch.as_tensor(cols, dtype=torch.long)
