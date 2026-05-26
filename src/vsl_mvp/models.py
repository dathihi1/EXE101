from __future__ import annotations

import math

import torch
from torch import nn


class GRUClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder(x)
        return self.head(out.mean(dim=1))


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        model_dim: int = 128,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, model_dim)
        self.pos = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.proj(x))
        x = self.encoder(x)
        return self.head(x.mean(dim=1))


class AttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(x * weights, dim=1)


class LiteTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        model_dim: int = 96,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos = PositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=model_dim * 3,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool = AttentionPooling(model_dim)
        self.head = nn.Sequential(nn.LayerNorm(model_dim), nn.Dropout(dropout), nn.Linear(model_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos(self.proj(x))
        x = self.encoder(x)
        return self.head(self.pool(x))


def build_model(name: str, input_dim: int, num_classes: int) -> nn.Module:
    if name == "gru":
        return GRUClassifier(input_dim=input_dim, num_classes=num_classes)
    if name == "transformer":
        return TransformerClassifier(input_dim=input_dim, num_classes=num_classes)
    if name == "lite_transformer":
        return LiteTransformerClassifier(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
