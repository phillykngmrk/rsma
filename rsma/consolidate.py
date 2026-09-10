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
def consolidate(model, state, holdout_batches, eta=1.0, tol=0.0):
    """
    state: list of fast deltas (B, H, R, d). The batch dimension is averaged before merging.
    Returns dict with before/after losses, accepted flag, and the reset state.

    The held-out set is fixed, so its loss is deterministic and the default tolerance is zero:
    a merge is kept only if it does not make the model worse. A positive tolerance lets small
    regressions accumulate across repeated merges, which is a ratchet in the wrong direction.
    """
    fast_layers = model.fast_layers
    before = holdout_loss(model, holdout_batches, None)
    backup = [fl.W0.detach().clone() for fl in fast_layers]
    for fl, delta in zip(fast_layers, state):
        fl.W0.add_(eta * delta.mean(0))
    after = holdout_loss(model, holdout_batches, None)
    accepted = after <= before * (1.0 + tol)
    if not accepted:
        for fl, w in zip(fast_layers, backup):
            fl.W0.copy_(w)
    fresh = [torch.zeros_like(d) for d in state]
    merged_norm = sum(d.mean(0).norm().item() for d in state)
    return {"before": before, "after": after, "accepted": accepted, "merged_norm": merged_norm, "state": fresh}


def sleep(model, recent_batches, replay_batches, holdout_batches, steps=20, lr=1e-4, tol=0.0):
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
