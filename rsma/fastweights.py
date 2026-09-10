"""
Self-referential fast weights (after Irie, Schlag, Csordas, Schmidhuber 2022).

Each head owns a matrix W of shape (R, d) with R = 3d + 1. Given input x:

    [_, q, k, beta] = W x          phi(z) = z / |z|
    v     = W phi(q)               the read: what the matrix currently associates with the query
    v_bar = W phi(k)
    W    <- W + gate * sigmoid(beta) * (v - v_bar) phi(k)^T
    y     = v[:d]                  the output is the first d rows of the retrieved vector

The matrix rewrites itself so that it maps phi(k) to what it currently maps phi(q) to. Every
quantity, including the query, key and learning rate, is produced by W itself, and the rows
that produce them are rewritten too. W = W0 + delta: W0 is a trained slow parameter, delta is
the fast state. Reading through a unit-norm query is what makes stored associations
retrievable; reading with the raw input does not, because a layer-normed input is orthogonal
to near-uniform keys.

Updates are applied per chunk: within a chunk W is frozen, the chunk's updates are summed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfReferentialFastWeights(nn.Module):
    def __init__(self, cfg, fast_index: int):
        super().__init__()
        self.cfg = cfg
        self.fast_index = fast_index
        D, H = cfg.d_model, cfg.fast_heads
        assert D % H == 0
        self.H, self.d = H, D // H
        self.R = 3 * self.d + 1
        self.W0 = nn.Parameter(torch.zeros(H, self.R, self.d))
        nn.init.normal_(self.W0, std=cfg.fast_w0_std)
        with torch.no_grad():
            self.W0[:, -1, :].zero_()  # beta row starts at zero; rate set by beta_bias
        self.beta_bias = nn.Parameter(torch.full((H,), cfg.fast_beta_init))
        self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, D)
        self.reset_special_init()

    def reset_special_init(self):
        nn.init.normal_(self.out.weight, std=0.01)  # small so the sublayer starts near identity
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.W0[:, -1, :].zero_()

    def init_state(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.H, self.R, self.d, device=device)

    def _clip(self, delta: torch.Tensor) -> torch.Tensor:
        m = self.cfg.fast_max_norm
        n = delta.flatten(2).norm(dim=-1)[:, :, None, None]
        return delta * torch.clamp(m / (n + 1e-6), max=1.0)

    def forward(self, x: torch.Tensor, delta: torch.Tensor, selfmodel=None):
        """
        x: (B, T, D) residual stream
        delta: (B, H, R, d) fast state entering this sequence
        returns y (B, T, D), new delta, aux dict with per-chunk encodings and gates
        """
        B, T, D = x.shape
        C = self.cfg.chunk_size
        assert T % C == 0, "sequence length must be a multiple of chunk_size"
        n_chunks = T // C
        xn = self.ln(x).view(B, T, self.H, self.d)
        d = self.d
        ys, encs, gates = [], [], []
        for c in range(n_chunks):
            xc = xn[:, c * C:(c + 1) * C]                     # (B, C, H, d)
            W = self.W0.unsqueeze(0) + delta                   # (B, H, R, d)
            o = torch.einsum("bhrd,bthd->bthr", W, xc)         # (B, C, H, R)
            _, q, k, beta = torch.split(o, [d, d, d, 1], dim=-1)
            phi_q = F.normalize(q, dim=-1)
            phi_k = F.normalize(k, dim=-1)
            v = torch.einsum("bhrd,bthd->bthr", W, phi_q)      # read
            v_bar = torch.einsum("bhrd,bthd->bthr", W, phi_k)
            ys.append(v[..., :d])
            lr = torch.sigmoid(beta + self.beta_bias[None, None, :, None]) * self.cfg.fast_lr_scale
            upd = torch.einsum("bthr,bthd->bhrd", lr * (v - v_bar), phi_k)
            if selfmodel is not None and self.cfg.use_gate:
                enc = selfmodel.encode(delta, self.fast_index)
                pooled = x[:, c * C:(c + 1) * C].mean(dim=1)
                g = selfmodel.gate_from_encoding(enc, pooled)
            else:
                enc = selfmodel.encode(delta, self.fast_index) if selfmodel is not None else None
                g = torch.ones(B, self.H, 1, 1, device=x.device)
            encs.append(enc)
            gates.append(g.flatten(1))
            delta = self._clip(delta + g * upd)
        if selfmodel is not None:
            encs.append(selfmodel.encode(delta, self.fast_index))  # state after the last chunk
        y = torch.cat(ys, dim=1).reshape(B, T, D)
        aux = {
            "encs": torch.stack(encs, dim=1) if encs[0] is not None else None,  # (B, n_chunks+1, hid)
            "gates": torch.stack(gates, dim=1),                                  # (B, n_chunks, H)
            "delta_norm": delta.flatten(2).norm(dim=-1).mean(),
        }
        return self.out(y), delta, aux
