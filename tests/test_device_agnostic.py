"""Device-agnostic contract tests (A3 Unit P1).

The CPU-side tests pin the public contract: every device-touching path accepts
an explicit device kwarg or honors the device of the tensors it's given. A
CUDA mirror runs only when CUDA is available, and asserts that the same input
produces the same output (within floating-point tolerance) on CPU and CUDA.
"""

import numpy as np
import pytest
import torch

from lines.datagen.box_scene import sample_box_scene
from lines.datagen.render import render_primitives
from lines.datagen.sampler2d import Canvas
from lines.models.autoregressive import (
    AutoregressiveModel, greedy_sample, teacher_forced_loss,
)
from lines.models.seq_tokenizer import EOS, Tokenizer
from lines.primitives import Circle, Line, PrimitiveSet
from lines.refine.diffvg_refine import refine_primitives


CANVAS = Canvas(64, 64)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --- diffvg refinement: must accept a device and produce equivalent output ---

def test_refine_accepts_device_kwarg_on_cpu():
    target = PrimitiveSet([Circle(center=(32.0, 32.0), radius=12.0)])
    img = render_primitives(target, 64, 64, line_width=2.0)
    start = PrimitiveSet([Circle(center=(32.0, 32.0), radius=8.0)])
    out = refine_primitives(start, img, CANVAS, steps=20, lr=8e-3, device="cpu")
    assert abs(out.primitives[0].radius - 12.0) < abs(8.0 - 12.0)


@cuda_only
def test_refine_runs_on_cuda_and_matches_cpu():
    target = PrimitiveSet([Circle(center=(32.0, 32.0), radius=12.0)])
    img = render_primitives(target, 64, 64, line_width=2.0)
    start = PrimitiveSet([Circle(center=(32.0, 32.0), radius=8.0)])
    torch.manual_seed(0)
    cpu_out = refine_primitives(start, img, CANVAS, steps=10, lr=8e-3, device="cpu")
    torch.manual_seed(0)
    gpu_out = refine_primitives(start, img, CANVAS, steps=10, lr=8e-3, device="cuda")
    # not bit-exact (different kernels); same direction + magnitude
    assert abs(cpu_out.primitives[0].radius - gpu_out.primitives[0].radius) < 0.5


# --- autoregressive: device follows model parameters --------------------------

def test_autoregressive_forward_on_cpu():
    m = AutoregressiveModel(canvas_side=64, d_model=64, n_heads=4, n_decoder_layers=2)
    images = torch.zeros(1, 1, 64, 64)
    targets = torch.tensor([[0, 10, 20, 1]], dtype=torch.long)
    logits = m(images, targets[:, :-1])
    assert logits.device.type == "cpu"


@cuda_only
def test_autoregressive_forward_on_cuda():
    m = AutoregressiveModel(canvas_side=64, d_model=64, n_heads=4, n_decoder_layers=2).cuda()
    images = torch.zeros(1, 1, 64, 64, device="cuda")
    targets = torch.tensor([[0, 10, 20, 1]], dtype=torch.long, device="cuda")
    logits = m(images, targets[:, :-1])
    assert logits.device.type == "cuda"


@cuda_only
def test_autoregressive_overfits_a_tiny_batch_on_cuda():
    tok = Tokenizer(canvas_side=64)
    images, seqs = [], []
    for i in range(4):
        pset = sample_box_scene(i, CANVAS)
        img = render_primitives(pset, 64, 64, line_width=2.0)
        images.append(torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0))
        seqs.append(tok.encode(pset))
    max_len = max(len(s) for s in seqs)
    targets = torch.full((4, max_len), EOS, dtype=torch.long)
    for i, s in enumerate(seqs):
        targets[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    batch_img = torch.stack(images, dim=0).cuda()
    targets = targets.cuda()

    model = AutoregressiveModel(canvas_side=64, d_model=64, n_heads=4,
                                n_decoder_layers=2, max_seq_len=max_len + 4).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    losses = []
    for _ in range(150):
        loss = teacher_forced_loss(model, batch_img, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5


@cuda_only
def test_greedy_sample_works_on_cuda():
    m = AutoregressiveModel(canvas_side=64, d_model=64, n_heads=4, n_decoder_layers=2).cuda()
    images = torch.zeros(1, 1, 64, 64, device="cuda")
    tokens = greedy_sample(m, images, max_len=10)
    assert isinstance(tokens, list) and tokens[0] == 0      # SOS
