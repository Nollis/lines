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
def beam_sample(model: AutoregressiveModel, image: torch.Tensor,
                max_len: int = 128, beam_size: int = 3,
                length_norm_alpha: float = 0.7) -> List[int]:
    """Beam-search decoding from one image.

    Returns the highest-scoring complete sequence (ends in EOS) or the highest-
    scoring partial sequence if none finishes within ``max_len``. ``beam_size=1``
    is exactly equivalent to greedy decoding.

    Scoring is sum of log-probs, normalized by ``len ** alpha`` (Wu et al.); this
    prevents the well-known bias toward short / early-EOS sequences when
    beam_size > 1.
    """
    if beam_size < 1:
        raise ValueError("beam_size must be >= 1")
    model.eval()
    device = next(model.parameters()).device
    memory = model.encode_image(image.to(device))                # (1, S, d)

    # each beam: (tokens, cum_logp, finished)
    beams = [( [SOS], 0.0, False )]
    finished: List[tuple] = []

    for _ in range(max_len - 1):
        # gather still-active beams; freeze the finished ones for the survivors
        live = [b for b in beams if not b[2]]
        if not live:
            break

        # batched forward over all live beams (one decoder pass per step)
        max_t = max(len(b[0]) for b in live)
        # PAD-aware batching is overkill for small beams; just pad with EOS
        # (causal mask makes trailing tokens irrelevant for the last position)
        in_tokens = torch.full((len(live), max_t), EOS, dtype=torch.long, device=device)
        lengths = []
        for i, (toks, _, _) in enumerate(live):
            in_tokens[i, :len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
            lengths.append(len(toks))
        T = max_t
        tgt = model.tok_embed(in_tokens) + model.tok_pos[:T]
        causal = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
        # cross-attend each beam to the same image memory
        mem = memory.expand(len(live), -1, -1)
        decoded = model.decoder(tgt=tgt, memory=mem, tgt_mask=causal)

        candidates = []
        for i, (toks, cum_lp, _) in enumerate(live):
            last = lengths[i] - 1
            logp = model.head(decoded[i, last]).log_softmax(dim=-1)        # (V,)
            topv, topi = logp.topk(beam_size)
            for v, idx in zip(topv.tolist(), topi.tolist()):
                candidates.append((toks + [idx], cum_lp + v, idx == EOS))

        # finished beams from this step go to the holding pool; pick top-K live for next step
        new_live = []
        for toks, cum_lp, is_eos in candidates:
            if is_eos:
                finished.append((toks, cum_lp))
            else:
                new_live.append((toks, cum_lp, False))
        new_live.sort(key=lambda b: -b[1])
        beams = new_live[:beam_size]
        if not beams:
            break

    def _score(toks, cum_lp):
        return cum_lp / max(1, len(toks)) ** length_norm_alpha

    if finished:
        best = max(finished, key=lambda b: _score(b[0], b[1]))
        return best[0]
    # no beam ended naturally -- return the best surviving partial
    best = max(beams, key=lambda b: _score(b[0], b[1]))
    return best[0]


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
