"""
Meta-training. Slow weights, the self-model, and the fast-weight init W0 are trained by
backpropagating through the fast-weight updates inside each sequence.

  python -m rsma.train --task synthetic --name fast
  python -m rsma.train --task synthetic --name nofast --no-fast
  python -m rsma.train --task text --name text_fast
"""
import argparse
import json
import math
import os
import time

import torch

from .config import RSMAConfig
from .model import RSMA
from .data.synthetic import RuleSwitchMarkov
from .data.text import CharText


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_data(args, device):
    if args.task == "synthetic":
        train = RuleSwitchMarkov(vocab=args.vocab, seq_len=args.seq_len, n_switches=args.n_switches, seed=args.seed, device=device)
        val = RuleSwitchMarkov(vocab=args.vocab, seq_len=args.seq_len, n_switches=args.n_switches, seed=args.seed + 1000, device=device)
        vocab = args.vocab
        return train, val, vocab
    ds = CharText(seq_len=args.seq_len, seed=args.seed, device=device)
    return ds, ds, ds.vocab


def next_batch(data, args, split="train"):
    if args.task == "synthetic":
        x, y, _ = data.batch(args.batch)
        return x, y
    return data.batch(args.batch, split)


@torch.no_grad()
def evaluate(model, data, args, n=8):
    model.eval()
    lm, sm, corr = 0.0, 0.0, 0.0
    for _ in range(n):
        x, y = next_batch(data, args, "val")
        _, _, aux = model(x, targets=y)
        lm += aux["lm_loss"].item()
        if "sm_loss" in aux:
            sm += aux["sm_loss"].item()
            p = aux["pred_loss"][:, :aux["chunk_loss"].shape[1]].flatten()
            a = aux["chunk_loss"].flatten()
            if p.std() > 0 and a.std() > 0:
                corr += torch.corrcoef(torch.stack([p, a]))[0, 1].item()
    model.train()
    return {"val_lm": lm / n, "val_sm": sm / n, "val_selfmodel_corr": corr / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="synthetic", choices=["synthetic", "text"])
    ap.add_argument("--name", default="run")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--fast-layers", default="1,3,5")
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--n-switches", type=int, default=1)
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--no-fast", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--sm-weight", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or get_device()
    torch.manual_seed(args.seed)
    train_data, val_data, vocab = make_data(args, device)
    cfg = RSMAConfig(
        vocab_size=vocab, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
        seq_len=args.seq_len, chunk_size=args.chunk,
        fast_layers=tuple(int(i) for i in args.fast_layers.split(",") if i != ""),
        use_fast=not args.no_fast, use_gate=not args.no_gate,
        selfmodel_loss_weight=args.sm_weight, tier=args.tier,
    )
    model = RSMA(cfg).to(device)
    print(f"device={device} params={model.n_params()/1e6:.2f}M fast_layers={model.fast_layer_ids}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    run_dir = os.path.join("runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    json.dump({"cfg": cfg.to_dict(), "args": vars(args)}, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    log = open(os.path.join(run_dir, "log.jsonl"), "w")

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = next_batch(train_data, args)
        _, _, aux = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        aux["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rec = {"step": step, "lm": aux["lm_loss"].item(), "lr": lr_at(step)}
        if "sm_loss" in aux:
            rec["sm"] = aux["sm_loss"].item()
            rec["gate"] = torch.stack([g.mean() for g in aux["gates"]]).mean().item()
            rec["delta_norm"] = aux["delta_norms"].mean().item()
        if step % 20 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step:5d} lm {rec['lm']:.4f}" + (f" sm {rec['sm']:.4f} gate {rec['gate']:.3f} |d| {rec['delta_norm']:.2f}" if "sm" in rec else "") + f"  {el/step*1000:.0f}ms/step")
        if step % args.eval_every == 0 or step == args.steps:
            rec.update(evaluate(model, val_data, args))
            print(f"  eval step {step}: " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k.startswith("val")))
            torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step}, os.path.join(run_dir, "ckpt.pt"))
        log.write(json.dumps(rec) + "\n")
        log.flush()
    log.close()


if __name__ == "__main__":
    main()
