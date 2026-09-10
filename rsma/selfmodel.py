"""
The self-model. Two jobs, sharing one encoder of the fast-weight state:

1. Gates. From a summary of a layer's fast-weight delta plus the pooled input of the current
   chunk, decide per head how much of the old memory to keep and how much of this chunk's
   proposed modification to write. Trained end to end by the language-model loss flowing back
   through the fast-weight update.

2. Forecast. From the summaries of every fast layer at the start of a chunk plus the pooled
   hidden state of the chunk before, predict the BENEFIT of the carried fast state on that
   chunk: loss with the state reset minus loss with the state carried. Positive means the
   memory helps. Trained against the benefit actually measured (a second forward pass with the
   state reset). At inference the forecast drives rollback: a modification is discarded when
   the forecast benefit of the new state is lower than that of the state before it.

Predicting benefit rather than absolute loss separates the effect of the modification from the
difficulty of the text, which is what gating and rollback actually need.
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
            nn.Linear(hid + cfg.d_model, hid), nn.GELU(), nn.Linear(hid, 2 * H)
        )
        self.predict = nn.Sequential(
            nn.Linear(n_fast_layers * hid + cfg.d_model, hid), nn.GELU(), nn.Linear(hid, hid), nn.GELU(), nn.Linear(hid, 1)
        )
        self.reset_special_init()

    def reset_special_init(self):
        # start with the write gate open (0.88) and the keep gate nearly closed to forgetting (0.98)
        nn.init.zeros_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].bias[: self.H].fill_(2.0)
            self.gate[-1].bias[self.H:].fill_(4.0)
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)

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

    def gates_from_encoding(self, enc: torch.Tensor, pooled_x: torch.Tensor):
        """returns (write, keep), each (B, H, 1, 1) in (0,1)."""
        g = torch.sigmoid(self.gate(torch.cat([enc, pooled_x], dim=-1)))
        w, k = g[:, : self.H], g[:, self.H:]
        return w[:, :, None, None], k[:, :, None, None]

    def predict_benefit(self, encs: torch.Tensor, pooled_h: torch.Tensor) -> torch.Tensor:
        """
        encs: (B, C, n_fast, hid) encodings of each fast layer's state at the start of chunk c
        pooled_h: (B, C, d_model) pooled final hidden state of the chunk before
        returns (B, C): predicted (loss with state reset - loss with state carried) for chunk c
        """
        B, C = encs.shape[:2]
        x = torch.cat([encs.reshape(B, C, -1), pooled_h], dim=-1)
        return self.predict(x).squeeze(-1)
