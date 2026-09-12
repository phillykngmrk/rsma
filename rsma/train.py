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
from .data.tokens import TokenText
from .data.figures import FigureText


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
    if args.task == "figures":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.graft)
        ds = FigureText(tok, seq_len=args.seq_len, seed=args.seed, device=device)
        if args.fact_streams > 0:
            ds.with_fact_streams(args.fact_streams, seed=args.seed)
        print("figures:", ds.summary(), "| fact streams:", args.fact_streams)
        return ds, ds, ds.vocab
    if args.task == "tokens":
        sources = {k: float(v) for k, v in (kv.split("=") for kv in args.sources.split(","))} if args.sources else None
        ds = TokenText(seq_len=args.seq_len, sources=sources, seed=args.seed, device=device)
        print("token sources:", {n: f"{ds.sizes[n]/1e6:.1f}M" for n in ds.names}, "weights", dict(zip(ds.names, ds.weights.round(3))))
        return ds, ds, ds.vocab
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
    preds, actual = [], []
    for _ in range(n_streams):
        state = None
        for i, (x, y) in enumerate(stream_windows(data, args, "val")):
            _, _, aux0 = model(x, None, targets=y)
            _, state, aux = model(x, state, targets=y, ref_chunk_loss=aux0["chunk_loss"])
            if i == 0:
                acc["w0"].append(aux["lm_loss"].item())
            else:
                acc["later_persist"].append(aux["lm_loss"].item())
                acc["later_reset"].append(aux0["lm_loss"].item())
                acc["early_persist"].append(aux["per_token_loss"][:, :early].mean().item())
                acc["early_reset"].append(aux0["per_token_loss"][:, :early].mean().item())
                if "benefit" in aux:
                    C = aux["benefit"].shape[1]
                    preds.append(aux["pred_benefit"][:, :C].flatten().cpu())
                    actual.append(aux["benefit"].flatten().cpu())
    model.train()
    out = {f"stream_{k}": (sum(v) / len(v) if v else float("nan")) for k, v in acc.items()}
    if preds:
        p, a = torch.cat(preds), torch.cat(actual)
        out["stream_selfmodel_corr"] = torch.corrcoef(torch.stack([p, a]))[0, 1].item() if p.std() > 0 and a.std() > 0 else 0.0
        out["stream_selfmodel_mae"] = (p - a).abs().mean().item()
        out["stream_benefit_mean"] = a.mean().item()
    return out


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
    model.train()
    out = {"val_lm": lm / n, "val_sm": sm / n}
    if args.stream > 1:
        out.update(stream_eval(model, data, args))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="synthetic", choices=["synthetic", "text", "tokens", "figures"])
    ap.add_argument("--graft", default=None, help="Hugging Face base model to graft RSMA onto (frozen base + adapters)")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--base-dtype", default="fp32", choices=["fp32", "bf16"], help="dtype of the frozen base (bf16 halves memory for larger bases)")
    ap.add_argument("--fact-streams", type=float, default=0.0, help="fraction of training streams that are tell-then-ask memory dialogues")
    ap.add_argument("--lora-lr", type=float, default=None, help="separate learning rate for the base adapters (default: same as --lr)")
    ap.add_argument("--keep-ckpts", action="store_true", help="also keep a copy of the checkpoint at every evaluation")
    ap.add_argument("--sources", default=None, help='token source weights, e.g. "wikitext=0.6,gutenberg=0.2,malcolmx=0.2"')
    ap.add_argument("--init-from", default=None, help="run name whose checkpoint initializes the model (fine-tuning)")
    ap.add_argument("--resume", action="store_true", help="continue this run from its checkpoint (optimizer state is not restored)")
    ap.add_argument("--fast-heads", type=int, default=4)
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
    cfg.fast_heads = args.fast_heads
    if args.graft:
        from .graft import GraftedRSMA
        model = GraftedRSMA(args.graft, cfg, lora_r=args.lora_r, dtype=torch.bfloat16 if args.base_dtype == "bf16" else torch.float32).to(device)
        cfg = model.cfg
    else:
        model = RSMA(cfg).to(device)
    start_step = 1
    if args.resume and os.path.exists(os.path.join("runs", args.name, "ckpt.pt")):
        ck = torch.load(os.path.join("runs", args.name, "ckpt.pt"), map_location=device)
        model.load_state_dict(ck["model"] if "model" in ck else ck["trainable"], strict=False)
        start_step = int(ck.get("step", 0)) + 1
        print(f"resumed runs/{args.name} at step {start_step}")
    if args.init_from:
        ck = torch.load(os.path.join("runs", args.init_from, "ckpt.pt"), map_location=device)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        print(f"initialized from runs/{args.init_from} (missing {len(missing)}, unexpected {len(unexpected)})")
    print(f"device={device} params={model.n_params()/1e6:.2f}M trainable={model.n_params(True)/1e6:.2f}M fast_layers={model.fast_layer_ids}")
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad and (n.endswith(".A") or n.endswith(".B"))]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and not (n.endswith(".A") or n.endswith(".B"))]
    groups = [{"params": other_params, "lr": args.lr, "base_lr": args.lr}]
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lora_lr or args.lr, "base_lr": args.lora_lr or args.lr})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.1)

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
    if args.task == "tokens":
        meta["tokenizer"] = "data_cache/tokens/tokenizer.json"
    json.dump(meta, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    log = open(os.path.join(run_dir, "log.jsonl"), "a" if args.resume else "w")

    model.train()
    t0 = time.time()
    for step in range(start_step, args.steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g["base_lr"] / args.lr
        opt.zero_grad(set_to_none=True)
        if args.stream > 1:
            # one step = one stream. Fast state is carried across windows WITH gradient, so the
            # loss on later windows teaches the model what to write into its weights earlier.
            state, total, auxs = None, 0.0, []
            for wi, (x, y) in enumerate(stream_windows(train_data, args)):
                ref = None
                if wi > 0 and model.selfmodel is not None:
                    with torch.no_grad():  # what the loss would be without the carried state
                        _, _, raux = model(x, None, targets=y)
                    ref = raux["chunk_loss"]
                _, state, aux = model(x, state, targets=y, ref_chunk_loss=ref)
                total = total + aux["loss"]
                auxs.append(aux)
            (total / len(auxs)).backward()
            aux = {"lm_loss": torch.stack([a["lm_loss"] for a in auxs]).mean()}
            if "sm_loss" in auxs[0]:
                aux["sm_loss"] = torch.stack([a["sm_loss"] for a in auxs]).mean()
                aux["gates"] = auxs[-1]["gates"]
                aux["delta_norms"] = auxs[-1]["delta_norms"]
        else:
            x, y = next_batch(train_data, args)
            _, _, aux = model(x, targets=y)
            aux["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        rec = {"step": step, "lm": aux["lm_loss"].item(), "lr": lr_at(step)}
        if args.stream > 1:
            rec["windows"] = step * args.stream
        if "sm_loss" in aux:
            rec["sm"] = aux["sm_loss"].item()
            H = model.cfg.fast_heads
            rec["gate"] = torch.stack([g[..., :H].mean() for g in aux["gates"]]).mean().item()
            rec["keep"] = torch.stack([g[..., H:].mean() for g in aux["gates"]]).mean().item()
            rec["delta_norm"] = aux["delta_norms"].mean().item()
        if step % 20 == 0 or step == 1:
            el = time.time() - t0
            print(f"step {step:5d} lm {rec['lm']:.4f}" + (f" sm {rec['sm']:.4f} write {rec['gate']:.3f} keep {rec['keep']:.3f} |d| {rec['delta_norm']:.2f}" if "sm" in rec else "") + f"  {el/max(1, step - start_step + 1)*1000:.0f}ms/step")
        if step % args.eval_every == 0 or step == args.steps:
            rec.update(evaluate(model, val_data, args))
            print(f"  eval step {step}: " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if k.startswith(("val", "stream"))))
            if args.graft:
                model.save(os.path.join(run_dir, "ckpt.pt"), extra={"step": step, "system_prompt": getattr(train_data, "system_prompt", None), "persona_name": getattr(train_data, "persona_name", None)})
                if args.keep_ckpts:
                    import shutil
                    shutil.copy(os.path.join(run_dir, "ckpt.pt"), os.path.join(run_dir, f"ckpt_step{step}.pt"))
            else:
                torch.save({"cfg": cfg.to_dict(), "model": model.state_dict(), "step": step}, os.path.join(run_dir, "ckpt.pt"))
        log.write(json.dumps(rec) + "\n")
        log.flush()
    log.close()


if __name__ == "__main__":
    main()
