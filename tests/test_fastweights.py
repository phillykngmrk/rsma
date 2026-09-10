import torch
import torch.nn.functional as F
from rsma.config import RSMAConfig
from rsma.fastweights import SelfReferentialFastWeights, _solve_unit_lower


def test_solve_unit_lower():
    A = torch.tril(torch.randn(2, 3, 16, 16), diagonal=-1)
    rhs = torch.randn(2, 3, 16, 5)
    U = _solve_unit_lower(A, rhs)
    ref = torch.linalg.solve(torch.eye(16) + A, rhs)
    assert torch.allclose(U, ref, atol=1e-4)


def test_chunked_matches_sequential_delta_rule():
    """Within a chunk, the WY update must equal applying the delta rule token by token
    (with q, k, beta computed from the chunk-start matrix)."""
    torch.manual_seed(0)
    cfg = RSMAConfig(d_model=32, fast_heads=2, chunk_size=8, fast_max_norm=1e9, use_gate=False)
    _check_sequential(cfg)
    _check_sequential(RSMAConfig(d_model=32, fast_heads=2, chunk_size=8, fast_max_norm=1e9, use_gate=False, fast_direct_value=False))


def _check_sequential(cfg):
    fl = SelfReferentialFastWeights(cfg, 0).eval()
    x = torch.randn(1, 8, 32)
    delta0 = torch.randn(1, 2, fl.R, fl.d) * 0.05
    y_chunk, delta_chunk, _ = fl(x, delta0, None)
    # sequential reference
    d = fl.d
    xn = fl.ln(x).view(1, 8, 2, d)
    W = fl.W0.unsqueeze(0) + delta0
    o = torch.einsum("bhrd,bthd->bthr", W, xn)
    _, q, k, beta = torch.split(o, [d, d, d, 1], dim=-1)
    pq, pk = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    lr = torch.sigmoid(beta + fl.beta_bias[None, None, :, None]) * cfg.fast_lr_scale
    Wt = W.clone()
    ys = []
    for t in range(8):
        v = torch.einsum("bhrd,bhd->bhr", Wt, pq[:, t])
        vb = torch.einsum("bhrd,bhd->bhr", Wt, pk[:, t])
        ys.append(v[..., :d])
        tgt = v
        if fl.Wv is not None:
            tgt = tgt + torch.einsum("hrd,bhd->bhr", fl.Wv, xn[:, t])
        u = lr[:, t] * (tgt - vb)                                 # (B, H, R)
        Wt = Wt + torch.einsum("bhr,bhd->bhrd", u, pk[:, t])
    y_seq = fl.out(torch.stack(ys, dim=1).reshape(1, 8, 32))
    assert torch.allclose(y_chunk, y_seq, atol=1e-4), (y_chunk - y_seq).abs().max()
    assert torch.allclose(delta_chunk, Wt - fl.W0.unsqueeze(0), atol=1e-4)


def test_delta_is_not_rank_one():
    torch.manual_seed(0)
    cfg = RSMAConfig(vocab_size=32)
    from rsma.model import RSMA
    m = RSMA(cfg).eval()
    x = torch.randint(0, 32, (1, 256))
    with torch.no_grad():
        _, st, _ = m(x)
    s = torch.linalg.svdvals(st[0][0, 0])
    assert s[1] / s[0] > 0.1, s[:4]
