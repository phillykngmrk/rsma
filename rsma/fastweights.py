"""
Self-referential fast weights (after Irie, Schlag, Csordas, Schmidhuber 2022), with two
additions: a direct input-to-value path in the write target, and a learned forget rate.

Each head owns a matrix W of shape (R, d) with R = 3d + 1. Given input x:

    [_, q, k, beta] = W x          phi(z) = z / |z|
    v     = W phi(q)               the read: what the matrix currently associates with the query
    v_bar = W phi(k)
    t     = v + Wv x               the write target: the read plus a direct function of the input
    W    <- keep * W + write * sigmoid(beta) * (t - v_bar) phi(k)^T
    y     = v[:d]                  the output is the first d rows of the retrieved vector

Every quantity in the update, including the query, key and learning rate, is produced by W
itself, and the rows that produce them are rewritten too. W = W0 + delta: W0 is a trained
slow parameter, delta is the fast state. `keep` and `write` come from the self-model, which
reads a summary of delta: it decides per chunk how much of the old memory to keep and how much
of the proposed modification to apply. Wv is a slow parameter that learns what is worth storing.

Updates are applied per chunk in exact sequential (WY) form: q, k, beta and the direct target
are computed from the chunk-start matrix, and the within-chunk updates are solved as a
triangular system so that each token's write sees the writes before it.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _solve_unit_lower(A, rhs):
    """
    Solve (I + A) U = rhs for A strictly lower triangular of size C.
    N = -A is nilpotent (N^C = 0), so (I - N)^-1 = (I + N)(I + N^2)(I + N^4)... exactly.
    Pure matmuls, so it runs on any backend.
    """
    C = A.shape[-1]
    I = torch.eye(C, device=A.device, dtype=A.dtype)
    N = -A
    inv = I + N
    P = N @ N
    steps = 2
    while steps < C:
        inv = inv @ (I + P)
        P = P @ P
        steps *= 2
    return inv @ rhs


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
        self.Wv = nn.Parameter(torch.zeros(H, self.R, self.d)) if cfg.fast_direct_value else None
        if self.Wv is not None:
            nn.init.normal_(self.Wv, std=cfg.fast_w0_std)
        self.beta_bias = nn.Parameter(torch.full((H,), cfg.fast_beta_init))
        self.ln = nn.LayerNorm(D)
        self.out = nn.Linear(D, D)
        self.reset_special_init()

    def reset_special_init(self):
        nn.init.normal_(self.out.weight, std=0.01)  # small so the sublayer starts near identity
        nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            self.W0[:, -1, :].zero_()  # beta row starts at zero; rate set by beta_bias
            if self.Wv is not None:
                self.Wv[:, self.d:, :].zero_()  # direct path writes only the output rows

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
            # self-model looks at the state and this chunk's input, decides keep and write rates
            if selfmodel is not None:
                enc = selfmodel.encode(delta, self.fast_index)
                if self.cfg.use_gate:
                    pooled = x[:, c * C:(c + 1) * C].mean(dim=1)
                    g_write, g_keep = selfmodel.gates_from_encoding(enc, pooled)  # (B, H, 1, 1) each
                else:
                    g_write = g_keep = torch.ones(B, self.H, 1, 1, device=x.device)
            else:
                enc = None
                g_write = g_keep = torch.ones(B, self.H, 1, 1, device=x.device)
            delta = g_keep * delta                             # forget first
            W = self.W0.unsqueeze(0) + delta                   # (B, H, R, d)
            o = torch.einsum("bhrd,bthd->bthr", W, xc)         # (B, C, H, R)
            _, q, k, beta = torch.split(o, [d, d, d, 1], dim=-1)
            pq = F.normalize(q, dim=-1).transpose(1, 2)        # (B, H, C, d)
            pk = F.normalize(k, dim=-1).transpose(1, 2)
            V = torch.einsum("bhrd,bhcd->bhcr", W, pq)         # read at chunk start
            Vb = torch.einsum("bhrd,bhcd->bhcr", W, pk)
            target = V
            if self.Wv is not None:
                target = target + torch.einsum("hrd,bthd->bhtr", self.Wv, xc)
            lr = torch.sigmoid(beta + self.beta_bias[None, None, :, None]) * self.cfg.fast_lr_scale
            lr = lr.transpose(1, 2) * g_write                  # (B, H, C, 1)
            # Exact sequential delta rule within the chunk (WY form). With u_i the update
            # vector for token i and W_i = W + sum_{j<i} u_j k_j^T:
            #   u_i = lr_i [ (W_i q_i + Wv x_i) - (W_i k_i) ]
            #       = lr_i [ (target_i - Vb_i) + sum_{j<i} u_j (k_j.q_i - k_j.k_i) ]
            # which is the triangular system (I + A) U = lr (target - Vb), A_ij = lr_i (k_j.k_i - k_j.q_i), j < i.
            KK = pk @ pk.transpose(-1, -2)                     # [i, j] = k_i . k_j
            QK = pq @ pk.transpose(-1, -2)                     # [i, j] = q_i . k_j
            A = torch.tril(lr * (KK - QK), diagonal=-1)        # (B, H, C, C)
            U = _solve_unit_lower(A, lr * (target - Vb))       # (B, H, C, R)
            Y = V + torch.tril(QK, diagonal=-1) @ U            # read with within-chunk updates applied
            ys.append(Y[..., :d].transpose(1, 2))              # (B, C, H, d)
            encs.append(enc)
            gates.append(torch.cat([g_write.flatten(1), g_keep.flatten(1)], dim=1))
            delta = self._clip(delta + torch.einsum("bhcr,bhcd->bhrd", U, pk))
        if selfmodel is not None:
            encs.append(selfmodel.encode(delta, self.fast_index))  # state after the last chunk
        y = torch.cat(ys, dim=1).reshape(B, T, D)
        aux = {
            "encs": torch.stack(encs, dim=1) if encs[0] is not None else None,  # (B, n_chunks+1, hid)
            "gates": torch.stack(gates, dim=1),                                  # (B, n_chunks, 2H): write then keep
            "delta_norm": delta.flatten(2).norm(dim=-1).mean(),
        }
        return self.out(y), delta, aux
