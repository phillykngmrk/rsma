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
    ds = CharText(seq_len=args.seq_len, seed=args.seed, device=device, corpus=args.corpus)
    return ds, ds, ds.vocab


def next_batch(data, args, split="train"):
    if args.task == "synthetic":
        x, y, _ = data.batch(args.batch)
        return x, y
    return data.batch(args.batch, split)


def stream_windows(data, args, split="train"):
    """(x, y) windows of one stream of args.stream consecutive sequences"""
    if args.task == "synthetic":
        for x, y, _ in data.stream(args.batch, args.stream):
            yield x, y
    else:
        yield from data.stream(args.batch, args.stream, split)


def window_iter(data, args, split="train"):
    """infinite iterator of (x, y, first_in_stream)"""
    while True:
        if args.stream <= 1:
            x, y = next_batch(data, args, split)
            yield x, y, True
        else:
            for i, (x, y) in enumerate(stream_windows(data, args, split)):
                yield x, y, i == 0


@torch.no_grad()
def stream_eval(model, data, args, n_streams=4, early=32):
    """
    Memory across the attention window. For each stream, run windows with fast state carried
    (persistent) and with fast state reset (reset). Loss on the first `early` tokens of windows
    after the first is where carried-over knowledge shows up.
    """
    model.eval()
    acc = {"w0": [], "later_persist": [], "later_reset": [], "early_persist": [], "early_reset": []}
    for _ in range(n_streams):
        state = None
        for i, (x, y) in enumerate(stream_windows(data, args, "val")):
            _, state, aux = model(x, state, targets=y)
            _, _, aux0 = model(x, None, targets=y)
            if i == 0:
                acc["w0"].append(aux["lm_loss"].item())
            else:
                acc["later_persist"].append(aux["lm_loss"].item())
                acc["later_reset"].append(aux0["lm_loss"].item())
                acc["early_persist"].append(aux["per_token_loss"][:, :early].mean().item())
                acc["early_reset"].append(aux0["per_token_loss"][:, :early].mean().item())
    model.train()
    return {f"stream_{k}": (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}


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
    out = {"val_lm": lm / n, "val_sm": sm / n, "val_selfmodel_corr": corr / n}
    if args.stream > 1:
        out.update(stream_eval(model, data, args))
    return out


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
    ap.add_argument("--corpus", default=None, help="path to a UTF-8 text file for --task text")
    ap.add_argument("--n-switches", type=int, default=1)
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--stream", type=int, default=1, help="consecutive windows per stream; fast state carries across them")
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
        selfmodel_loss_weight=args.sm_weight, tier=args.tier, stream_len=args.stream,
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
    meta = {"cfg": cfg.to_dict(), "args": vars(args)}
    if args.task == "text":
        meta["vocab_chars"] = train_data.itos
    json.dump(meta, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    log = open(os.path.join(run_dir, "log.jsonl"), "w")

    model.train()
    t0 = time.time()
    windows = window_iter(train_data, args)
    state = None
    for step in range(1, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y, first = next(windows)
        if first:
            state = None
        _, new_state, aux = model(x, state, targets=y)
        state = model.clone_state(new_state) if new_state else None
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
            print(f"  eval step {step}: " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k.startswith(("val", "stream"))))
            torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step}, os.path.join(run_dir, "ckpt.pt"))
        log.write(json.dumps(rec) + "\n")
        log.flush()
    log.close()


if __name__ == "__main__":
    main()
