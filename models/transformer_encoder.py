"""
TRATA - Transformer Temporal Encoder (Layer 04, Stage 1 of Hybrid Engine)
---------------------------------------------------------------------------
Matches the architecture diagram: "Transformers capture long-range
dependencies" -> feeds into Random Forest for "nonlinear interactions" and
"robust prediction".

This module is trained first (supervised, with a temporary linear head) so
that it learns a meaningful pooled embedding of the 48-timestep x 21-feature
solar wind / geomagnetic sequence. After training, the head is discarded and
get_embedding() is used to produce fixed-length vectors that get concatenated
with the current physics feature vector and fed into the Random Forest
(see models/random_forest_head.py + train.py).
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding so the Transformer knows the
    temporal order of the 48 timesteps (4 hours of history)."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerForecastEncoder(nn.Module):
    """
    Physics-informed Transformer encoder.

    Input:  (batch, seq_len=48, n_features=21)
    Output: pooled embedding (batch, d_model) via get_embedding()
            OR full multi-horizon prediction (batch, n_horizons) via forward()
            (the prediction head is only used during pretraining of the
            embedding; downstream forecasting uses the Random Forest head)
    """

    def __init__(self, n_features: int, n_horizons: int = 3, d_model: int = 32,
                 n_heads: int = 2, n_layers: int = 2, dim_feedforward: int = 64,
                 dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        # Pretraining head (discarded after embedding extraction; RF replaces it)
        self.pretrain_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_horizons),
        )

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the pooled temporal embedding used as Random Forest input."""
        x = self.input_proj(x)                 # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)          # (batch, seq_len, d_model)
        x = self.norm(x)
        pooled = x.mean(dim=1)                   # mean-pool over time -> (batch, d_model)
        return pooled

    def get_embedding_mc_dropout(self, x: torch.Tensor, n_samples: int = 30) -> torch.Tensor:
        """MC-Dropout embedding sampling: keeps dropout layers ACTIVE across
        n_samples stochastic forward passes through the Transformer encoder.
        Returns (n_samples, batch, d_model). Used at inference time by the
        Random Forest head to produce mean + std (confidence) predictions
        through the full hybrid pipeline, restoring the MC-Dropout
        uncertainty approach on top of the Transformer + Random Forest
        architecture.
        """
        self.train()  # activates dropout inside TransformerEncoderLayer
        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(self.get_embedding(x).unsqueeze(0))
        self.eval()
        return torch.cat(samples, dim=0)  # (n_samples, batch, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Used only during Transformer pretraining (supervised embedding learning)."""
        emb = self.get_embedding(x)
        return self.pretrain_head(emb)
