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
def consolidate(model, state, holdout_batches, eta=1.0, tol=0.02):
    """
    state: list of fast deltas (B, H, R, d). The batch dimension is averaged before merging.
    Returns dict with before/after losses, accepted flag, and the reset state.
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
