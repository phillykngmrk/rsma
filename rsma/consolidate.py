"""
Tier 3: consolidation. Merge accumulated fast deltas into the slow W0 of each fast layer, then
verify on a held-out buffer. If the merged model regresses beyond tolerance, revert.

This is the step where the model permanently rewrites its own parameters.
"""
import torch


@torch.no_grad()
def holdout_loss(model, batches, state=None):
    model.eval()
    tot, n = 0.0, 0
    for x, y in batches:
        _, _, aux = model(x, state, targets=y)
        tot += aux["lm_loss"].item()
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def consolidate(model, state, holdout_batches, eta=1.0, tol=0.0, etas=(1.0, 0.5, 0.25), conv_batches=None):
    """
    Merge the fast deltas into the slow W0 of each fast layer, verified on held-out data.
    state: list of fast deltas (B, H, R, d); the batch dimension is averaged before merging.
    etas: merge strengths to try, largest first; the first that does not regress is kept.
    conv_batches: optional held-out batches of recent conversation. When given, the verified
      loss is the mean of corpus loss and conversation loss, so a merge that helps the model
      remember the conversation can be accepted even if corpus loss is unchanged.
    The held-out sets are fixed, so their loss is deterministic and the default tolerance is
    zero: a merge is kept only if it does not make the model worse. A positive tolerance lets
    small regressions accumulate across repeated merges.
    """
    fast_layers = model.fast_layers

    def verify():
        l = holdout_loss(model, holdout_batches, None)
        if conv_batches:
            return 0.5 * l + 0.5 * holdout_loss(model, conv_batches, None), l
        return l, l

    before, before_corpus = verify()
    backup = [fl.W0.detach().clone() for fl in fast_layers]
    means = [d.mean(0) for d in state]
    tried = []
    accepted, used_eta, after, after_corpus = False, None, before, before_corpus
    for e in ([eta] if etas is None else etas):
        for fl, w, dm in zip(fast_layers, backup, means):
            fl.W0.copy_(w + e * dm)
        a, ac = verify()
        tried.append((e, round(a, 4)))
        if a <= before * (1.0 + tol):
            accepted, used_eta, after, after_corpus = True, e, a, ac
            break
    if not accepted:
        for fl, w in zip(fast_layers, backup):
            fl.W0.copy_(w)
    fresh = [torch.zeros_like(d) for d in state]
    merged_norm = sum(dm.norm().item() for dm in means)
    return {"before": before, "after": after, "before_corpus": before_corpus, "after_corpus": after_corpus,
            "accepted": accepted, "eta": used_eta, "tried": tried, "merged_norm": merged_norm, "state": fresh}


def sleep(model, recent_batches, replay_batches, holdout_batches, steps=20, lr=2e-5, tol=0.0):
    """
    Gradient consolidation. Fine-tune the slow weights on recent experience mixed with replay
    from the original corpus, then verify on held-out data and revert if it regressed.
    recent_batches / replay_batches: lists of (x, y). Interleaved round-robin.
    """
    if not recent_batches:
        return {"skipped": True}
    before = holdout_loss(model, holdout_batches, None)
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    losses = []
    for i in range(steps):
        src = recent_batches if (i % 2 == 0 or not replay_batches) else replay_batches
        x, y = src[(i // 2) % len(src)]
        _, _, aux = model(x, None, targets=y)
        opt.zero_grad(set_to_none=True)
        aux["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(aux["lm_loss"].item())
    model.eval()
    after = holdout_loss(model, holdout_batches, None)
    accepted = after <= before * (1.0 + tol)
    if not accepted:
        model.load_state_dict(backup)
    return {"before": before, "after": after, "accepted": accepted, "steps": steps,
            "train_loss_first": losses[0], "train_loss_last": losses[-1]}
