"""
Tier 2 and 3 runtime. Feeds a stream of sequences through the model with persistent fast
weights, uses the self-model's forecast to decide on rollback, and at tier 3 periodically
consolidates fast weights into slow weights.
"""
import torch
from .consolidate import consolidate


class SelfModifyingRuntime:
    def __init__(self, model, holdout_batches=None, tol=0.15, ema=0.95,
                 consolidate_every=50, consolidate_eta=1.0, consolidate_tol=0.02, device="cpu"):
        self.model = model
        self.cfg = model.cfg
        self.tier = model.cfg.tier
        self.holdout = holdout_batches or []
        self.tol = tol
        self.ema_decay = ema
        self.consolidate_every = consolidate_every
        self.consolidate_eta = consolidate_eta
        self.consolidate_tol = consolidate_tol
        self.device = device
        self.state = None
        self.ema_loss = None
        self.step_count = 0
        self.rollbacks = 0
        self.consolidations = []
        self.log = []

    @torch.no_grad()
    def step(self, x, y):
        self.model.eval()
        B = x.shape[0]
        if self.state is None or self.tier == 1:
            self.state = self.model.init_state(B, x.device)
        snapshot = self.model.clone_state(self.state)
        _, new_state, aux = self.model(x, self.state, targets=y)
        actual = aux["lm_loss"].item()
        forecast = aux["pred_loss"][:, -1].mean().item() if "pred_loss" in aux else actual
        rolled_back = False
        if self.ema_loss is None:
            self.ema_loss = actual
        elif self.tier >= 2 and forecast > self.ema_loss * (1.0 + self.tol):
            # the self-model expects this modification to hurt: discard it
            new_state = snapshot
            rolled_back = True
            self.rollbacks += 1
        self.ema_loss = self.ema_decay * self.ema_loss + (1 - self.ema_decay) * actual
        self.state = new_state
        self.step_count += 1
        rec = {"step": self.step_count, "loss": actual, "forecast": forecast, "ema": self.ema_loss,
               "rolled_back": rolled_back, "delta_norm": aux["delta_norms"].mean().item() if "delta_norms" in aux else 0.0}
        if self.tier == 3 and self.step_count % self.consolidate_every == 0 and self.holdout:
            res = consolidate(self.model, self.state, self.holdout, self.consolidate_eta, self.consolidate_tol)
            self.state = res.pop("state")
            res["step"] = self.step_count
            self.consolidations.append(res)
            rec["consolidated"] = res["accepted"]
        self.log.append(rec)
        return rec
