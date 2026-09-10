"""
The self-model. Two jobs:

1. Gate: read a summary of a layer's current fast-weight delta plus the pooled input of the
   current chunk, and decide how much of this chunk's proposed self-modification to apply.
   Trained end to end by the language-model loss flowing back through the fast-weight update.

2. Predict: read the summaries of every fast layer after a chunk's update, plus the pooled final
   hidden state, and predict the loss the model will incur on the next chunk. Trained against the
   loss that actually occurs. At inference this prediction drives rollback.

Both share one summary encoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfModel(nn.Module):
    def __init__(self, cfg, delta_rows: int, delta_cols: int, n_fast_layers: int):
        super().__init__()
        self.cfg = cfg
        H = cfg.fast_heads
        self.H = H
        self.n_fast = n_fast_layers
        flat = delta_rows * delta_cols
        # fixed random projection of each head's flattened delta; a stable, cheap fingerprint
        proj = torch.randn(flat, cfg.summary_dim) / flat ** 0.5
        self.register_buffer("proj", proj)
        stats = 3  # norm, mean, std
        per_head = stats + cfg.summary_dim
        hid = cfg.selfmodel_hidden
        self.layer_emb = nn.Embedding(n_fast_layers, hid)
        self.encoder = nn.Sequential(
            nn.Linear(H * per_head, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU()
        )
        self.gate = nn.Sequential(
            nn.Linear(hid + cfg.d_model, hid), nn.GELU(), nn.Linear(hid, H)
        )
        self.predict = nn.Sequential(
            nn.Linear(n_fast_layers * hid + cfg.d_model, hid), nn.GELU(), nn.Linear(hid, 1)
        )
        # open gate at init so training starts as a plain fast-weight model
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, 2.0)

    def summarize(self, delta: torch.Tensor) -> torch.Tensor:
        """delta: (B, H, R, d) -> (B, H*(3+summary_dim))"""
        B, H, R, d = delta.shape
        flat = delta.reshape(B, H, R * d)
        norm = flat.norm(dim=-1, keepdim=True)
        mean = flat.mean(dim=-1, keepdim=True)
        std = flat.std(dim=-1, keepdim=True)
        fp = flat @ self.proj
        return torch.cat([norm, mean, std, fp], dim=-1).reshape(B, -1)

    def encode(self, delta: torch.Tensor, fast_index: int) -> torch.Tensor:
        """(B, hid) encoding of one layer's fast state."""
        s = self.summarize(delta)
        idx = torch.full((delta.shape[0],), fast_index, device=delta.device, dtype=torch.long)
        return self.encoder(s) + self.layer_emb(idx)

    def gate_from_encoding(self, enc: torch.Tensor, pooled_x: torch.Tensor) -> torch.Tensor:
        """(B, H, 1, 1) multiplier in (0,1) on this chunk's proposed update."""
        g = torch.sigmoid(self.gate(torch.cat([enc, pooled_x], dim=-1)))
        return g[:, :, None, None]

    def predict_loss(self, encs: torch.Tensor, pooled_h: torch.Tensor) -> torch.Tensor:
        """
        encs: (B, C, n_fast, hid) encodings of each fast layer's state at the start of chunk c
        pooled_h: (B, C, d_model) pooled final hidden state of the chunk before
        returns (B, C) predicted mean loss for chunk c
        """
        B, C = encs.shape[:2]
        x = torch.cat([encs.reshape(B, C, -1), pooled_h], dim=-1)
        return F.softplus(self.predict(x)).squeeze(-1)
