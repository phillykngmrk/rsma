"""
Interactive loop with a self-modifying model.

  python -m rsma.chat --run text_fast

Every turn, the conversation text passes through the model with persistent fast weights, so the
model adapts to you as you talk. The self-model forecasts whether each modification will help and
rolls it back otherwise. At tier 3 the fast weights are periodically merged into the slow weights
and the checkpoint is rewritten: the model you talk to tomorrow is the one today's conversation
produced. The state lives in runs/<run>/self/ and survives restarts.

Commands:
  /status        fast-weight norms, forecast vs actual loss, rollbacks, consolidations
  /consolidate   merge fast weights into slow weights now (verified, may be rejected)
  /sleep         gradient consolidation on recent conversation with corpus replay (verified)
  /reset         clear fast weights
  /save          write self-state and checkpoint
  /freeze        toggle self-modification on/off
  /quit
"""
import argparse
import json
import os
import time

import torch

from .config import RSMAConfig
from .model import RSMA
from .data.text import CharText
from .consolidate import consolidate, sleep
from .train import get_device


class SelfState:
    """Everything that persists between sessions besides the checkpoint itself."""

    def __init__(self, path):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.fast = None
        self.ema_loss = None
        self.turns = 0
        self.rollbacks = 0
        self.events = []
        self.recent = []  # recent conversation token ids for sleep

    def load(self, device):
        f = os.path.join(self.path, "state.pt")
        if os.path.exists(f):
            d = torch.load(f, map_location=device)
            self.fast = d["fast"]
            self.ema_loss = d["ema_loss"]
            self.turns = d["turns"]
            self.rollbacks = d["rollbacks"]
            self.events = d["events"]
            self.recent = d["recent"]
            return True
        return False

    def save(self):
        torch.save({"fast": self.fast, "ema_loss": self.ema_loss, "turns": self.turns,
                    "rollbacks": self.rollbacks, "events": self.events, "recent": self.recent[-200000:]},
                   os.path.join(self.path, "state.pt"))


class Chat:
    def __init__(self, run, device, tier=3, rollback_tol=0.15, consolidate_every=8, temperature=0.8, max_new=200):
        self.run_dir = os.path.join("runs", run)
        ck = torch.load(os.path.join(self.run_dir, "ckpt.pt"), map_location=device)
        meta = json.load(open(os.path.join(self.run_dir, "config.json")))
        self.cfg = RSMAConfig.from_dict(ck["cfg"])
        self.cfg.tier = tier
        self.model = RSMA(self.cfg).to(device).eval()
        self.model.load_state_dict(ck["model"])
        self.ds = CharText(seq_len=self.cfg.seq_len, device=device, corpus=meta["args"].get("corpus"),
                           vocab_chars=meta.get("vocab_chars"))
        self.device = device
        self.tol = rollback_tol
        self.consolidate_every = consolidate_every
        self.temperature = temperature
        self.max_new = max_new
        self.frozen = False
        self.transcript = []
        self.state = SelfState(os.path.join(self.run_dir, "self"))
        if self.state.load(device):
            print(f"[resumed self-state: {self.state.turns} turns, {len(self.state.events)} events]")
        self.holdout = [self.ds.batch(16, "val") for _ in range(2)]
        self.replay = [self.ds.batch(8, "train") for _ in range(8)]

    # ---- helpers ------------------------------------------------------------------------------
    def _chunks(self, ids):
        """split token ids into full seq_len windows; drop the remainder for the learning pass"""
        T = self.cfg.seq_len
        return [ids[i:i + T] for i in range(0, len(ids) - T + 1, T)]

    @torch.no_grad()
    def learn(self, text):
        """Pass text through the model with persistent fast weights. Returns dict of what happened."""
        ids = self.ds.encode(text)
        self.state.recent.extend(ids)
        self.transcript.extend(ids)
        if self.frozen or not self.model.fast_layer_ids:
            return {"learned": 0}
        C = self.cfg.chunk_size
        n = (len(self.transcript) // C) * C
        if n < C:
            return {"learned": 0}
        window = self.transcript[-min(n, self.cfg.seq_len):]
        window = window[:(len(window) // C) * C]
        x = torch.tensor(window[:-1] + [window[-1]], device=self.device)[None]
        y = torch.tensor(window[1:] + [window[-1]], device=self.device)[None]
        if self.state.fast is None:
            self.state.fast = self.model.init_state(1, self.device)
        snapshot = self.model.clone_state(self.state.fast)
        _, new_state, aux = self.model(x, self.state.fast, targets=y)
        actual = aux["lm_loss"].item()
        forecast = aux["pred_loss"][:, -1].item()
        rolled = False
        if self.state.ema_loss is None:
            self.state.ema_loss = actual
        elif self.cfg.tier >= 2 and forecast > self.state.ema_loss * (1 + self.tol):
            new_state = snapshot
            rolled = True
            self.state.rollbacks += 1
        self.state.ema_loss = 0.9 * self.state.ema_loss + 0.1 * actual
        self.state.fast = new_state
        return {"learned": len(window), "loss": actual, "forecast": forecast, "rolled_back": rolled,
                "delta_norm": aux["delta_norms"].mean().item()}

    @torch.no_grad()
    def reply(self, prompt_ids):
        ctx = torch.tensor(prompt_ids[-self.cfg.seq_len:], device=self.device)[None]
        out = self.model.generate(ctx, self.state.fast, max_new=self.max_new, temperature=self.temperature)
        gen = out[0, ctx.shape[1]:].tolist()
        text = self.ds.decode(gen)
        # stop at the end of the model's turn if it produces the user marker
        cut = text.find("\nYou:")
        return text if cut < 0 else text[:cut]

    def do_consolidate(self):
        if self.state.fast is None:
            return {"skipped": True}
        res = consolidate(self.model, self.state.fast, self.holdout, eta=1.0, tol=0.0)
        self.state.fast = res.pop("state")
        res["kind"] = "merge"
        res["turn"] = self.state.turns
        self.state.events.append(res)
        if res["accepted"]:
            self.save_checkpoint()
        return res

    def do_sleep(self, steps=20):
        T = self.cfg.seq_len
        wins = self._chunks(self.state.recent[-T * 32:])
        recent = []
        for w in wins:
            x = torch.tensor(w, device=self.device)[None]
            y = torch.tensor(w[1:] + [w[-1]], device=self.device)[None]
            recent.append((x, y))
        res = sleep(self.model, recent, self.replay, self.holdout, steps=steps)
        res["kind"] = "sleep"
        res["turn"] = self.state.turns
        self.state.events.append(res)
        if res.get("accepted"):
            self.save_checkpoint()
        return res

    def save_checkpoint(self):
        torch.save({"cfg": self.cfg.to_dict(), "model": self.model.state_dict(), "step": -1,
                    "self_modified_at": time.time()}, os.path.join(self.run_dir, "ckpt.pt"))

    def status(self):
        d = self.state
        norms = [f"{t.flatten(2).norm(dim=-1).mean().item():.3f}" for t in d.fast] if d.fast is not None else []
        return {"turns": d.turns, "frozen": self.frozen, "tier": self.cfg.tier, "ema_loss": d.ema_loss,
                "fast_norms_per_layer": norms, "rollbacks": d.rollbacks,
                "consolidations": [e for e in d.events if e.get("kind") == "merge"][-3:],
                "sleeps": [e for e in d.events if e.get("kind") == "sleep"][-3:],
                "recent_tokens": len(d.recent)}

    # ---- loop ---------------------------------------------------------------------------------
    def run(self):
        print(self.__doc__ or "")
        print(f"[tier {self.cfg.tier}, {self.model.n_params()/1e6:.2f}M params, device {self.device}]")
        while True:
            try:
                user = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                user = "/quit"
            if user.startswith("/"):
                cmd = user.strip().split()[0]
                if cmd == "/quit":
                    self.state.save()
                    print("[self-state saved]")
                    return
                if cmd == "/status":
                    print(json.dumps(self.status(), indent=1))
                elif cmd == "/consolidate":
                    print(json.dumps(self.do_consolidate(), indent=1))
                elif cmd == "/sleep":
                    print(json.dumps(self.do_sleep(), indent=1))
                elif cmd == "/reset":
                    self.state.fast = None
                    print("[fast weights cleared]")
                elif cmd == "/save":
                    self.state.save()
                    self.save_checkpoint()
                    print("[saved]")
                elif cmd == "/freeze":
                    self.frozen = not self.frozen
                    print(f"[self-modification {'off' if self.frozen else 'on'}]")
                else:
                    print("[unknown command]")
                continue
            info = self.learn(f"\nYou: {user}\nModel:")
            answer = self.reply(self.transcript)
            print(f"Model:{answer}")
            info2 = self.learn(answer + "\n")
            self.state.turns += 1
            tag = []
            if info.get("rolled_back") or info2.get("rolled_back"):
                tag.append("rolled back")
            if "loss" in info2:
                tag.append(f"loss {info2['loss']:.2f} forecast {info2['forecast']:.2f} |d| {info2['delta_norm']:.2f}")
            if self.cfg.tier == 3 and self.state.turns % self.consolidate_every == 0 and not self.frozen:
                res = self.do_consolidate()
                tag.append(f"consolidated: {'accepted' if res.get('accepted') else 'rejected'}")
            if tag:
                print(f"  [{' | '.join(tag)}]")
            self.state.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tier", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--consolidate-every", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    Chat(args.run, args.device or get_device(), tier=args.tier, temperature=args.temperature,
         max_new=args.max_new, consolidate_every=args.consolidate_every).run()


if __name__ == "__main__":
    main()
