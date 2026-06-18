"""Tests for the autoregressive prototype model (A2)."""

import numpy as np
import torch

from lines.datagen.dataset import write_dataset
from lines.datagen.sampler2d import Canvas
from lines.models.autoregressive import (
    AutoregressiveModel, greedy_sample, teacher_forced_loss,
)
from lines.models.seq_tokenizer import EOS, SOS, Tokenizer, vocab_size

torch.manual_seed(0)


def _tiny_model(canvas_side=64):
    return AutoregressiveModel(canvas_side=canvas_side, d_model=64,
                               n_heads=4, n_decoder_layers=2)


def test_forward_returns_logits_over_vocab():
    m = _tiny_model()
    images = torch.zeros(2, 1, 64, 64)
    targets = torch.tensor([[SOS, 10, 20, EOS], [SOS, 30, 40, EOS]], dtype=torch.long)
    logits = m(images, targets[:, :-1])
    assert logits.shape == (2, 3, vocab_size())   # (B, T-1, V)


def test_loss_is_finite_on_random_input():
    m = _tiny_model()
    images = torch.zeros(1, 1, 64, 64)
    targets = torch.tensor([[SOS, 10, 20, 30, EOS]], dtype=torch.long)
    loss = teacher_forced_loss(m, images, targets)
    assert torch.isfinite(loss)


def test_greedy_sample_returns_a_sequence_ending_in_eos_or_max_len():
    m = _tiny_model()
    images = torch.zeros(1, 1, 64, 64)
    tokens = greedy_sample(m, images, max_len=20)
    assert tokens[0] == SOS
    assert tokens[-1] == EOS or len(tokens) == 20


def test_overfits_a_tiny_box_batch(tmp_path):
    """The capacity test: a tiny model must drive teacher-forced loss
    to near-zero on a fixed handful of box scenes. If it cannot, the
    architecture cannot represent the structure."""
    from lines.datagen.box_scene import sample_box_scene

    canvas = Canvas(64, 64)
    tok = Tokenizer(canvas_side=64)
    images, target_lists = [], []
    for i in range(8):
        pset = sample_box_scene(i, canvas)
        # build a 64x64 image deterministically (just a uniform gradient is fine
        # for representation capacity -- we are testing that the SEQUENCE can
        # be predicted, even if the image-to-sequence mapping isn't realistic)
        from lines.datagen.render import render_primitives
        img = render_primitives(pset, 64, 64, line_width=2.0)
        images.append(torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0))
        target_lists.append(tok.encode(pset))

    max_len = max(len(t) for t in target_lists)
    targets = torch.full((len(target_lists), max_len), EOS, dtype=torch.long)
    for i, seq in enumerate(target_lists):
        targets[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
    batch_img = torch.stack(images, dim=0)

    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    losses = []
    for _ in range(300):
        loss = teacher_forced_loss(model, batch_img, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.2, f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
    assert losses[-1] < 0.5, f"final loss too high: {losses[-1]:.3f}"
