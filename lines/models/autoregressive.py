"""Autoregressive (CAD-as-language) prototype for A2.

A small image-conditioned GPT-style decoder over the tokenizer vocabulary. The
job of *this* prototype is to answer one question: "can an autoregressive model
*represent* clean structured output?" -- i.e. overfit a tiny set of boxes and
concentric circles to high accuracy. Full-scale training is A3, after the
choice.

Architecture:

* CNN encoder (same shape as `SetPredictor`'s plain encoder) turns a
  ``HxW`` grayscale image into a ``H/16 x W/16`` token grid.
* A learned 2D positional embedding is added.
* A causal transformer decoder consumes a teacher-forced token sequence,
  cross-attending to the encoded image tokens.
* A linear head projects each decoder position to vocabulary logits.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lines.models.seq_tokenizer import EOS, PAD, SOS, vocab_size


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class AutoregressiveModel(nn.Module):
    def __init__(self, canvas_side: int = 64, d_model: int = 128,
                 n_heads: int = 4, n_decoder_layers: int = 2,
                 max_seq_len: int = 256):
        super().__init__()
        if canvas_side % 16:
            raise ValueError(f"canvas_side {canvas_side} must be a multiple of 16")
        self.canvas_side = canvas_side
        self.d_model = d_model
        self.feature_size = canvas_side // 16
        self.vocab_size = vocab_size()
        self.max_seq_len = max_seq_len

        # image encoder: 1 -> 32 -> 64 -> 96 -> d_model (each block /2)
        self.encoder = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, 96),
            _conv_block(96, d_model),
        )
        self.img_pos = nn.Parameter(
            torch.zeros(self.feature_size * self.feature_size, d_model))
        nn.init.trunc_normal_(self.img_pos, std=0.02)

        # token side
        self.tok_embed = nn.Embedding(self.vocab_size, d_model, padding_idx=PAD)
        self.tok_pos = nn.Parameter(torch.zeros(max_seq_len, d_model))
        nn.init.trunc_normal_(self.tok_pos, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)
        self.head = nn.Linear(d_model, self.vocab_size)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """``image``: (B, 1, H, W) float -> memory (B, S, d_model)."""
        feat = self.encoder(image)                            # (B, d, h, w)
        tokens = feat.flatten(2).transpose(1, 2)              # (B, hw, d)
        return tokens + self.img_pos                          # broadcast over batch

    def forward(self, image: torch.Tensor, in_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass. Returns (B, T, V) logits."""
        memory = self.encode_image(image)
        B, T = in_tokens.shape
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} > max_seq_len {self.max_seq_len}")
        tgt = self.tok_embed(in_tokens) + self.tok_pos[:T]    # (B, T, d)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=tgt.device), diagonal=1)
        decoded = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
        return self.head(decoded)


def teacher_forced_loss(model: AutoregressiveModel,
                        image: torch.Tensor,
                        targets: torch.Tensor,
                        ignore_index: int = PAD) -> torch.Tensor:
    """``targets`` includes SOS..EOS. Shifted next-token cross-entropy."""
    in_tokens = targets[:, :-1]
    out_tokens = targets[:, 1:]
    logits = model(image, in_tokens)                           # (B, T-1, V)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           out_tokens.reshape(-1),
                           ignore_index=ignore_index)


@torch.no_grad()
def greedy_sample(model: AutoregressiveModel, image: torch.Tensor,
                  max_len: int = 128) -> List[int]:
    """Greedy autoregressive decoding from one image. Returns a token list."""
    model.eval()
    device = next(model.parameters()).device
    memory = model.encode_image(image.to(device))             # (1, S, d)
    tokens = [SOS]
    for _ in range(max_len - 1):
        in_tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        T = in_tokens.size(1)
        tgt = model.tok_embed(in_tokens) + model.tok_pos[:T]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
        decoded = model.decoder(tgt=tgt, memory=memory, tgt_mask=causal)
        next_token = int(model.head(decoded[:, -1]).argmax(dim=-1).item())
        tokens.append(next_token)
        if next_token == EOS:
            break
    return tokens
