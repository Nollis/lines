"""Tests for the DETR-style Hungarian matcher (Unit 5)."""

import numpy as np
import torch

from lines.models.encoding import N_PARAMS, N_TYPES, TYPE_CIRCLE, TYPE_LINE, TYPE_NONE
from lines.models.matcher import HungarianMatcher

torch.manual_seed(0)


def _logits_for(types):
    """One-hot logits matching the given target types (with strong margin)."""
    logits = torch.full((len(types), N_TYPES), -5.0)
    for i, t in enumerate(types):
        logits[i, t] = 5.0
    return logits


def test_matcher_picks_obvious_assignment():
    # 3 queries, 2 GT primitives. Query 0 already predicts the line's params,
    # query 2 already predicts the circle's params, query 1 is junk.
    pred_logits = _logits_for([TYPE_LINE, TYPE_NONE, TYPE_CIRCLE])
    pred_params = torch.zeros(3, N_PARAMS)
    pred_params[0, :4] = torch.tensor([0.1, 0.1, 0.5, 0.5])
    pred_params[2, :3] = torch.tensor([0.5, 0.5, 0.1])

    gt_types = torch.tensor([TYPE_LINE, TYPE_CIRCLE])
    gt_params = torch.zeros(2, N_PARAMS)
    gt_params[0, :4] = torch.tensor([0.1, 0.1, 0.5, 0.5])
    gt_params[1, :3] = torch.tensor([0.5, 0.5, 0.1])

    matcher = HungarianMatcher()
    pred_idx, gt_idx = matcher(pred_logits, pred_params, gt_types, gt_params)

    pairs = dict(zip(pred_idx.tolist(), gt_idx.tolist()))
    assert pairs == {0: 0, 2: 1}


def test_matcher_handles_more_queries_than_gt():
    pred_logits = torch.zeros(8, N_TYPES)
    pred_params = torch.rand(8, N_PARAMS)
    gt_types = torch.tensor([TYPE_LINE])
    gt_params = torch.zeros(1, N_PARAMS)
    gt_params[0, :4] = torch.tensor([0.2, 0.2, 0.4, 0.4])

    matcher = HungarianMatcher()
    pred_idx, gt_idx = matcher(pred_logits, pred_params, gt_types, gt_params)
    assert len(pred_idx) == 1 and len(gt_idx) == 1
    assert 0 <= pred_idx.item() < 8 and gt_idx.item() == 0


def test_matcher_returns_empty_when_no_gt():
    pred_logits = torch.zeros(4, N_TYPES)
    pred_params = torch.zeros(4, N_PARAMS)
    gt_types = torch.empty(0, dtype=torch.long)
    gt_params = torch.zeros(0, N_PARAMS)
    matcher = HungarianMatcher()
    pred_idx, gt_idx = matcher(pred_logits, pred_params, gt_types, gt_params)
    assert pred_idx.numel() == 0 and gt_idx.numel() == 0
