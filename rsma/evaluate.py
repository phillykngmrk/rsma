"""
Evaluation of a trained RSMA.

  python -m rsma.evaluate --run fast --baseline nofast          # synthetic adaptation curves
  python -m rsma.evaluate --run text_fast --stream 40 --tier 3  # tier 2/3 runtime on a text stream

Reports:
  adaptation:   loss in windows after a rule switch, fast model vs frozen baseline
  self-model:   correlation between predicted and actual chunk loss
  runtime:      loss over a persistent stream, rollbacks, consolidation accept/reject
"""
import argparse
import json
import os

import torch

from .config import RSMAConfig
from .model import RSMA
from .data.synthetic import RuleSwitchMarkov
from .data.text import CharText
from .runtime import SelfModifyingRuntime
from .train import get_device, stream_eval


def load(run, device):
    ck = torch.load(os.path.join("runs", run, "ckpt.pt"), map_location=device)
    cfg = RSMAConfig.from_dict(ck["cfg"])
    m = RSMA(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    meta = json.load(open(os.path.join("runs", run, "config.json")))
    args = meta["args"]
    args["_vocab_chars"] = meta.get("vocab_chars")
    return m, cfg, args


@torch.no_grad()
def adaptation_curve(model, data, batches=8, batch=32, windows=((0, 8), (8, 32), (32, 96)), tier1=True):
    """Mean loss by position relative to the first switch, and per window."""
    T = data.seq_len
    sums = torch.zeros(2 * T)
    cnts = torch.zeros(2 * T)
    total = 0.0
    for _ in range(batches):
        x, y, sw = data.batch(batch)
        _, _, aux = model(x, targets=y)
        per = aux["per_token_loss"].cpu()
        total += per.mean().item()
        rel = torch.arange(T)[None] - sw[:, :1].cpu() + T  # index 0..2T
        sums.index_add_(0, rel.flatten(), per.flatten())
        cnts.index_add_(0, rel.flatten(), torch.ones(rel.numel()))
    curve = sums / cnts.clamp(min=1)
    out = {"mean_loss": total / batches}
    for a, b in windows:
        out[f"after_switch[{a},{b})"] = curve[T + a:T + b].mean().item()
    out["before_switch[-32,0)"] = curve[T - 32:T].mean().item()
    return out, curve


@torch.no_grad()
def selfmodel_calibration(model, get_batch, n=8):
    ps, as_ = [], []
    for _ in range(n):
        x, y = get_batch()
        _, _, aux = model(x, targets=y)
        if "pred_loss" not in aux:
            return None
        C = aux["chunk_loss"].shape[1]
        ps.append(aux["pred_loss"][:, :C].flatten().cpu())
        as_.append(aux["chunk_loss"].flatten().cpu())
    p, a = torch.cat(ps), torch.cat(as_)
    corr = torch.corrcoef(torch.stack([p, a]))[0, 1].item()
    return {"corr": corr, "mae": (p - a).abs().mean().item(), "pred_mean": p.mean().item(), "actual_mean": a.mean().item()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--stream", type=int, default=0, help="number of consecutive sequences for tier 2/3 runtime")
    ap.add_argument("--tier", type=int, default=None)
    ap.add_argument("--consolidate-every", type=int, default=10)
    ap.add_argument("--rollback-tol", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or get_device()

    model, cfg, targs = load(args.run, device)
    if args.tier is not None:
        model.cfg.tier = args.tier
    task = targs["task"]
    report = {"run": args.run, "task": task}

    class A:  # minimal args shim for stream_eval
        pass
    sa = A()
    sa.task, sa.batch, sa.stream = task, 32, max(2, targs.get("stream", 1))

    if task == "synthetic":
        data = RuleSwitchMarkov(vocab=targs["vocab"], seq_len=targs["seq_len"], n_switches=targs["n_switches"], seed=4242, device=device)
        report["adaptation"], curve = adaptation_curve(model, data)
        torch.save(curve, os.path.join("runs", args.run, "adaptation_curve.pt"))
        report["selfmodel"] = selfmodel_calibration(model, lambda: data.batch(32)[:2])
        report["memory"] = stream_eval(model, data, sa, n_streams=8)
        if args.baseline:
            base, _, _ = load(args.baseline, device)
            data_b = RuleSwitchMarkov(vocab=targs["vocab"], seq_len=targs["seq_len"], n_switches=targs["n_switches"], seed=4242, device=device)
            report["baseline_adaptation"], curve_b = adaptation_curve(base, data_b)
            report["baseline_memory"] = stream_eval(base, data_b, sa, n_streams=8)
            torch.save(curve_b, os.path.join("runs", args.baseline, "adaptation_curve.pt"))
        def stream_src(n):
            for _ in range(n):
                x, y, _ = data.batch(1)
                yield x, y
        holdout = [data.batch(16)[:2] for _ in range(2)]
    else:
        ds = CharText(seq_len=targs["seq_len"], seed=7, device=device, corpus=targs.get("corpus"), vocab_chars=targs.get("_vocab_chars"))
        report["selfmodel"] = selfmodel_calibration(model, lambda: ds.batch(32, "val"))
        report["memory"] = stream_eval(model, ds, sa, n_streams=8)
        report["val_loss"] = selfmodel_calibration(model, lambda: ds.batch(32, "val"))["actual_mean"] if report["selfmodel"] else None
        stream_src = lambda n: ds.stream(1, n, "val")
        holdout = [ds.batch(16, "val") for _ in range(2)]

    if args.stream > 0 and cfg.use_fast:
        rt = SelfModifyingRuntime(model, holdout_batches=holdout, tol=args.rollback_tol,
                                  consolidate_every=args.consolidate_every, device=device)
        # frozen comparison: same stream, no persistent state
        losses, frozen = [], []
        for x, y in stream_src(args.stream):
            rec = rt.step(x, y)
            losses.append(rec["loss"])
            with torch.no_grad():
                _, _, aux = model(x, None, targets=y)
            frozen.append(aux["lm_loss"].item())
        n = len(losses)
        h = n // 2
        report["runtime"] = {
            "tier": model.cfg.tier, "sequences": n,
            "persistent_loss_first_half": sum(losses[:h]) / h, "persistent_loss_second_half": sum(losses[h:]) / (n - h),
            "reset_loss_first_half": sum(frozen[:h]) / h, "reset_loss_second_half": sum(frozen[h:]) / (n - h),
            "rollbacks": rt.rollbacks,
            "consolidations": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in rt.consolidations],
        }
        json.dump(rt.log, open(os.path.join("runs", args.run, f"runtime_tier{model.cfg.tier}.json"), "w"), indent=1)

    print(json.dumps(report, indent=1))
    json.dump(report, open(os.path.join("runs", args.run, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
