"""DETR-style set-prediction model: image -> primitive set.

Architecture is deliberately small so v1 trains on CPU:

* a small convolutional encoder (4 strided blocks) turning a ``HxW`` grayscale
  image into an ``H/16 x W/16`` feature map;
* a learned 2D positional encoding added to the features;
* a transformer decoder with ``N`` learned query embeddings, cross-attending
  to the flattened encoder features;
* two linear heads on each query embedding: a type logit head and a 5-slot
  parameter head with a sigmoid on the first four slots (coordinates in
  ``[0, 1]``) and a free range for slot 4 (bulge).

The shapes are the contract -- ``forward`` returns ``(logits, params)`` of
shape ``(B, N, K)`` and ``(B, N, P)``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from lines.models.encoding import N_PARAMS, N_TYPES


def _plain_conv_block(c_in: int, c_out: int) -> nn.Sequential:
    """Legacy encoder block: two strided convs, no residual.

    Retained so checkpoints saved before the residual switch can be loaded.
    State-dict keys for this block are ``encoder.{i}.{0,1,3,4}.*``.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
        nn.GroupNorm(8, c_out),
        nn.GELU(),
    )


class ResidualBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, c_out)
        
        self.shortcut = nn.Sequential()
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
                nn.GroupNorm(8, c_out)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gelu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.gelu(out)



class SetPredictor(nn.Module):
    def __init__(
        self,
        n_queries: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_decoder_layers: int = 3,
        feature_size: int = 8,   # input HxW / 16 (e.g. 128 -> 8)
        encoder_type: str = "residual",  # "residual" (current) or "plain" (legacy compat)
    ):
        super().__init__()
        self.n_queries = n_queries
        self.d_model = d_model
        self.feature_size = feature_size
        self.encoder_type = encoder_type

        # encoder: 1 -> 32 -> 64 -> 96 -> d_model (each block /2)
        if encoder_type == "residual":
            self.encoder = nn.Sequential(
                ResidualBlock(1, 32),
                ResidualBlock(32, 64),
                ResidualBlock(64, 96),
                ResidualBlock(96, d_model),
            )
        elif encoder_type == "plain":
            self.encoder = nn.Sequential(
                _plain_conv_block(1, 32),
                _plain_conv_block(32, 64),
                _plain_conv_block(64, 96),
                _plain_conv_block(96, d_model),
            )
        else:
            raise ValueError(f"unknown encoder_type: {encoder_type!r}")

        # learned 2D positional embedding for the feature map tokens
        self.pos_embed = nn.Parameter(torch.zeros(feature_size * feature_size, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # learned query embeddings, one per primitive slot
        self.queries = nn.Parameter(torch.zeros(n_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        self.head_type = nn.Linear(d_model, N_TYPES)
        self.head_params = nn.Linear(d_model, N_PARAMS)
        # bias the type head toward "none" at init so the model starts predicting
        # empty -- a stable starting point for set prediction
        with torch.no_grad():
            self.head_type.bias.zero_()
            self.head_type.bias[-1] = 2.0  # TYPE_NONE

    def forward(self, image: torch.Tensor):
        """``image``: (B, 1, H, W) float in ``[0, 1]``. Returns (logits, params).

        ``logits``: (B, N, K)   pre-softmax type scores
        ``params``: (B, N, P)   first 4 slots in (0,1) via sigmoid, slot 4 free
        """
        B = image.shape[0]
        feat = self.encoder(image)                                  # (B, d, h, w)
        h, w = feat.shape[-2:]
        assert h == self.feature_size and w == self.feature_size, \
            f"encoder produced {h}x{w}, expected {self.feature_size}x{self.feature_size}"
        tokens = feat.flatten(2).transpose(1, 2)                    # (B, hw, d)
        tokens = tokens + self.pos_embed                            # broadcast

        queries = self.queries.unsqueeze(0).expand(B, -1, -1)       # (B, N, d)
        decoded = self.decoder(tgt=queries, memory=tokens)          # (B, N, d)

        logits = self.head_type(decoded)                            # (B, N, K)
        raw = self.head_params(decoded)
        coords = torch.sigmoid(raw[..., :4])
        free = raw[..., 4:5]
        params = torch.cat([coords, free], dim=-1)                  # (B, N, P)
        return logits, params


def detect_encoder_type(state_dict: dict) -> str:
    """Sniff a state dict to decide which encoder shape produced it.

    The plain (legacy) encoder stores ``encoder.0.0.weight`` (the first Conv2d
    inside an ``nn.Sequential``). The residual encoder stores
    ``encoder.0.conv1.weight``.
    """
    if "encoder.0.conv1.weight" in state_dict:
        return "residual"
    if "encoder.0.0.weight" in state_dict:
        return "plain"
    raise ValueError("state dict has no recognized encoder keys")


def load_set_predictor(checkpoint_path, map_location: str = "cpu"):
    """Instantiate a :class:`SetPredictor` matching a saved checkpoint.

    Auto-detects ``encoder_type`` from the state dict so checkpoints saved
    before the residual switch still load. Returns ``(model, cfg)``.
    """
    ck = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    state = ck["model"]
    cfg = dict(ck.get("cfg", {}))
    cfg.setdefault("encoder_type", detect_encoder_type(state))

    canvas_side = int(cfg.get("canvas_side", 64))
    model = SetPredictor(
        n_queries=int(cfg.get("n_queries", 16)),
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        n_decoder_layers=int(cfg.get("n_decoder_layers", 3)),
        feature_size=required_feature_size(canvas_side),
        encoder_type=cfg["encoder_type"],
    )
    model.load_state_dict(state)
    return model, cfg


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def required_feature_size(image_hw: int) -> int:
    """Encoder downsamples by 16 (four stride-2 blocks)."""
    if image_hw % 16:
        raise ValueError(f"image side {image_hw} must be a multiple of 16")
    return image_hw // 16
