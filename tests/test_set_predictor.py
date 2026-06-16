"""Tests for the set-prediction model and loss (Unit 5).

The keystone test is ``test_overfits_tiny_batch``: a small model must drive
loss to near-zero on a fixed 4-image batch. If it cannot, the matcher / loss /
heads are wired wrong and no amount of real training will help.
"""

import numpy as np
import torch

from lines.datagen.dataset import write_dataset, Dataset
from lines.datagen.sampler2d import Canvas
from lines.models.encoding import N_PARAMS, N_TYPES, TYPE_NONE, encode_set
from lines.models.losses import SetPredictionLoss
from lines.models.matcher import HungarianMatcher
from lines.models.set_predictor import SetPredictor, required_feature_size

torch.manual_seed(0)
np.random.seed(0)


# --- shape / wiring -----------------------------------------------------------

def test_forward_returns_expected_shapes():
    m = SetPredictor(n_queries=8, d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=required_feature_size(64))
    x = torch.zeros(2, 1, 64, 64)
    logits, params = m(x)
    assert logits.shape == (2, 8, N_TYPES)
    assert params.shape == (2, 8, N_PARAMS)


def test_initial_type_predictions_lean_toward_none():
    m = SetPredictor(n_queries=8, d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=required_feature_size(64))
    with torch.no_grad():
        logits, _ = m(torch.zeros(1, 1, 64, 64))
        types = logits.argmax(-1).flatten().tolist()
    assert all(t == TYPE_NONE for t in types)


def test_coords_are_in_unit_interval():
    m = SetPredictor(n_queries=8, d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=required_feature_size(64))
    with torch.no_grad():
        _logits, params = m(torch.rand(2, 1, 64, 64))
    assert torch.all(params[..., :4] >= 0) and torch.all(params[..., :4] <= 1)


def test_loss_with_empty_gt_runs_and_is_finite():
    m = SetPredictor(n_queries=8, d_model=64, n_heads=4, n_decoder_layers=2,
                     feature_size=required_feature_size(64))
    crit = SetPredictionLoss(matcher=HungarianMatcher())
    logits, params = m(torch.zeros(1, 1, 64, 64))
    out = crit(logits, params, [torch.empty(0, dtype=torch.long)],
               [torch.zeros(0, N_PARAMS)])
    assert torch.isfinite(out.loss)
    assert out.n_matched == 0


# --- the keystone sanity check ------------------------------------------------

def test_overfits_tiny_batch(tmp_path):
    """A small model must drive loss to ~0 on a fixed 4-image batch."""
    canvas = Canvas(64, 64)
    write_dataset(tmp_path, n_samples=4, seed=0, canvas=canvas, randomize=False)
    ds = Dataset(tmp_path)

    images, gt_types_list, gt_params_list = [], [], []
    for i in range(len(ds)):
        img, pset = ds[i]
        images.append(torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0))
        t, p = encode_set(pset, canvas.width, canvas.height)
        gt_types_list.append(torch.from_numpy(t))
        gt_params_list.append(torch.from_numpy(p))
    batch = torch.stack(images, dim=0)

    model = SetPredictor(n_queries=8, d_model=64, n_heads=4, n_decoder_layers=2,
                         feature_size=required_feature_size(64))
    crit = SetPredictionLoss(matcher=HungarianMatcher(),
                             class_weight_none=0.5, param_weight=5.0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)

    losses = []
    for step in range(400):
        logits, params = model(batch)
        out = crit(logits, params, gt_types_list, gt_params_list)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())

    assert losses[-1] < losses[0] * 0.3, f"loss did not drop enough: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert losses[-1] < 0.55, f"final loss too high: {losses[-1]:.3f}"
