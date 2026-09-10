import torch
from rsma import RSMA, RSMAConfig


def small_cfg(**kw):
    base = dict(vocab_size=16, d_model=32, n_layers=2, n_heads=2, seq_len=32, chunk_size=8,
                fast_layers=(1,), fast_heads=2, summary_dim=8, selfmodel_hidden=16)
    base.update(kw)
    return RSMAConfig(**base)


def test_shapes_and_loss():
    cfg = small_cfg()
    m = RSMA(cfg)
    x = torch.randint(0, 16, (3, 32))
    logits, state, aux = m(x, targets=x)
    assert logits.shape == (3, 32, 16)
    assert len(state) == 1 and state[0].shape == (3, 2, 3 * 16 + 1, 16)
    assert aux["chunk_loss"].shape == (3, 4)
    assert aux["pred_loss"].shape == (3, 5)
    assert torch.isfinite(aux["loss"])
    aux["loss"].backward()


def test_causal():
    cfg = small_cfg()
    m = RSMA(cfg).eval()
    x = torch.randint(0, 16, (1, 32))
    x2 = x.clone()
    x2[0, 20:] = torch.randint(0, 16, (12,))
    l1, _, _ = m(x)
    l2, _, _ = m(x2)
    assert torch.allclose(l1[0, :20], l2[0, :20], atol=1e-5)


def test_state_changes_with_input_and_persists():
    cfg = small_cfg()
    m = RSMA(cfg).eval()
    x = torch.randint(0, 16, (2, 32))
    _, s1, _ = m(x)
    assert s1[0].abs().sum() > 0
    _, s2, _ = m(x, s1)
    assert not torch.allclose(s1[0], s2[0])


def test_norm_clip():
    cfg = small_cfg(fast_max_norm=0.5, fast_beta_init=5.0)
    m = RSMA(cfg).eval()
    x = torch.randint(0, 16, (2, 32))
    state = None
    for _ in range(5):
        _, state, _ = m(x, state)
    assert state[0].flatten(2).norm(dim=-1).max() <= 0.5 + 1e-4


def test_no_fast_baseline():
    cfg = small_cfg(use_fast=False)
    m = RSMA(cfg)
    x = torch.randint(0, 16, (2, 32))
    logits, state, aux = m(x, targets=x)
    assert state == [] and "pred_loss" not in aux
    aux["loss"].backward()


def test_consolidate_and_runtime():
    from rsma.runtime import SelfModifyingRuntime
    cfg = small_cfg(tier=3)
    m = RSMA(cfg)
    hold = [(torch.randint(0, 16, (2, 32)),) * 2 for _ in range(2)]
    rt = SelfModifyingRuntime(m, holdout_batches=hold, consolidate_every=3)
    for _ in range(6):
        rec = rt.step(*hold[0])
    assert len(rt.consolidations) == 2
    assert rt.state[0].abs().sum() == 0  # reset after consolidation


def test_synthetic_stream_windows():
    from rsma.data.synthetic import RuleSwitchMarkov
    d = RuleSwitchMarkov(vocab=8, seq_len=32, seed=1)
    wins = list(d.stream(2, 4))
    assert len(wins) == 4
    x0, y0, sw0 = wins[0]
    assert x0.shape == (2, 32) and torch.equal(x0[:, 1:], y0[:, :-1])
    x1, _, sw1 = wins[1]
    assert torch.equal(wins[0][1][:, -1], x1[:, 0])  # windows are contiguous
    assert torch.equal(sw1, sw0 - 32)
