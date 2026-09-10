"""Load a checkpoint into a model built from its config, tolerating parameters that have
changed shape since it was trained (the self-model and fast-weight layers have evolved).
Mismatched or missing parameters keep their fresh initialization and are reported."""
import os
import torch

from .config import RSMAConfig
from .model import RSMA


def load_run(run, device):
    ck = torch.load(os.path.join("runs", run, "ckpt.pt"), map_location=device)
    cfg = RSMAConfig.from_dict(ck["cfg"])
    sd = ck["model"]
    if cfg.use_fast and not any(k.endswith(".fast.Wv") for k in sd):
        cfg.fast_direct_value = False  # trained before the direct value path existed
    model = RSMA(cfg).to(device)
    own = model.state_dict()
    keep, dropped = {}, []
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            keep[k] = v
        else:
            dropped.append(k)
    missing = [k for k in own if k not in keep]
    model.load_state_dict(keep, strict=False)
    if dropped or missing:
        print(f"[checkpoint runs/{run}: {len(dropped)} parameters dropped, {len(missing)} freshly initialized "
              f"(self-model or fast-weight layout changed since training)]")
    model.eval()
    return model, cfg, ck
