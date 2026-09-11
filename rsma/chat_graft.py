"""
Talk to the grafted, self-modifying model.

  python -m rsma.chat_graft --run mentor --tier 3

The conversation uses the base model's chat format under the blended persona's system prompt.
Every turn passes through the model with persistent fast weights. The self-model forecasts
whether each modification helps and rolls it back otherwise. Every N turns the fast weights are
merged into the slow fast-weight matrices if a held-out mix of corpus and conversation does not
regress (tier 3); an accepted merge rewrites the checkpoint. /sleep fine-tunes the adapters and
fast layers on recent conversation with corpus replay, verified the same way.

Commands: /status /consolidate /sleep /reset /freeze /save /quit
"""
import argparse
import json
import os
import time

import torch

from .chat import SelfState
from .consolidate import consolidate, sleep
from .graft import GraftedRSMA
from .data.figures import FigureText, persona_prompt
from .train import get_device


class GraftChat:
    def __init__(self, run, device, tier=3, rollback_tol=0.02, consolidate_every=8, temperature=0.7, max_new=300,
                 persona_name=None, repetition_penalty=1.15, think=True, show_think=True):
        self.repetition_penalty = repetition_penalty
        self.think, self.show_think = think, show_think
        self.last_full = ""
        self.run_dir = os.path.join("runs", run)
        trained = os.path.join(self.run_dir, "ckpt.pt")           # written by training runs
        self.ckpt = os.path.join(self.run_dir, "self", "ckpt.pt")  # written by consolidation and sleep: lived experience
        os.makedirs(os.path.dirname(self.ckpt), exist_ok=True)
        if os.path.exists(self.ckpt) and os.path.getmtime(self.ckpt) >= os.path.getmtime(trained):
            source = self.ckpt
        else:
            source = trained
            if os.path.exists(self.ckpt):
                print("[a newer training checkpoint supersedes the lived checkpoint; starting from the trained weights]")
        self.model, ck = GraftedRSMA.load(source, device)
        self.cfg = self.model.cfg
        self.cfg.tier = tier
        self.tok = self.model.tokenizer
        self.device = device
        self.tol = rollback_tol
        self.consolidate_every = consolidate_every
        self.temperature = temperature
        self.max_new = max_new
        self.frozen = False
        self.persona_name = persona_name or ck.get("persona_name") or "Sankofa"
        self.system_prompt = persona_prompt(self.persona_name) if persona_name else (ck.get("system_prompt") or persona_prompt(self.persona_name))
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.transcript = self.tok(self.tok.apply_chat_template(self.messages, tokenize=False), add_special_tokens=False).input_ids
        self.state = SelfState(os.path.join(self.run_dir, "self"))
        if self.state.load(device):
            print(f"[resumed self-state: {self.state.turns} turns, {len(self.state.events)} events]")
        try:
            self.ds = FigureText(self.tok, seq_len=self.cfg.seq_len, device=device, persona_name=self.persona_name)
            self.holdout = [self.ds.batch(4, "val") for _ in range(2)]
            self.replay = [self.ds.batch(2, "train") for _ in range(8)]
        except FileNotFoundError:
            self.ds, self.holdout, self.replay = None, [], []
        self.stop_ids = {self.tok.eos_token_id}
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end >= 0:
            self.stop_ids.add(im_end)

    # ---- learning -----------------------------------------------------------------------------
    @torch.no_grad()
    def learn(self, ids):
        self.state.recent.extend(ids)
        self.transcript.extend(ids)
        if self.frozen:
            return {"learned": 0}
        C = self.cfg.chunk_size
        n = (len(self.transcript) // C) * C
        if n < C:
            return {"learned": 0}
        window = self.transcript[-min(n, self.cfg.seq_len):]
        window = window[:(len(window) // C) * C]
        x = torch.tensor(window, device=self.device)[None]
        y = torch.tensor(window[1:] + [window[-1]], device=self.device)[None]
        if self.state.fast is None:
            self.state.fast = self.model.init_state(1, self.device)
        snapshot = self.model.clone_state(self.state.fast)
        _, new_state, aux = self.model(x, self.state.fast, targets=y)
        actual = aux["lm_loss"].item()
        forecast = aux["pred_benefit"][:, -1].item()
        rolled = False
        if self.state.ema_loss is None:
            self.state.ema_loss = actual
        if self.cfg.tier >= 2:
            without = self.model.forecast(snapshot, aux["pooled_last"]).item()
            if forecast < without - self.tol:
                new_state = snapshot
                rolled = True
                self.state.rollbacks += 1
        self.state.ema_loss = 0.9 * self.state.ema_loss + 0.1 * actual
        self.state.fast = new_state
        return {"learned": len(window), "loss": actual, "forecast": forecast, "rolled_back": rolled,
                "delta_norm": aux["delta_norms"].mean().item()}

    def reply(self, user):
        """Returns (answer, thinking). Bases with a thinking mode emit a reasoning block first; it is
        shown separately, kept out of the reply history as the base's template expects, and included
        in what the model learns from, since it is part of its own experience."""
        self.messages.append({"role": "user", "content": user})
        try:
            prompt = self.tok.apply_chat_template(self.messages, add_generation_prompt=True, tokenize=False, enable_thinking=self.think)
        except TypeError:
            prompt = self.tok.apply_chat_template(self.messages, add_generation_prompt=True, tokenize=False)
        ids = self.tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
        out = self.model.generate(ids, self.state.fast, max_new=self.max_new, temperature=self.temperature, stop_ids=self.stop_ids,
                                  repetition_penalty=self.repetition_penalty)
        gen = out[0, ids.shape[1]:].tolist()
        full = self.tok.decode(gen, skip_special_tokens=True).strip()
        thinking, text = "", full
        if "</think>" in full:
            thinking, text = full.split("</think>", 1)
            thinking = thinking.replace("<think>", "").strip()
            text = text.strip()
        elif full.startswith("<think>"):
            thinking, text = full[len("<think>"):].strip(), ""  # ran out of tokens while thinking
        self.last_full = full
        self.messages.append({"role": "assistant", "content": text})
        # keep the message history bounded; the fast weights carry the rest
        if len(self.messages) > 21:
            self.messages = [self.messages[0]] + self.messages[-20:]
        return text, thinking

    def turn_ids(self, role, content):
        return self.tok(f"<|im_start|>{role}\n{content}<|im_end|>\n", add_special_tokens=False).input_ids

    # ---- consolidation --------------------------------------------------------------------------
    def conversation_holdout(self):
        T = self.cfg.seq_len
        r = self.state.recent
        if len(r) < T + 1:
            return []
        w = r[-T - 1:]
        return [(torch.tensor(w[:-1], device=self.device)[None], torch.tensor(w[1:], device=self.device)[None])]

    def do_consolidate(self):
        if self.state.fast is None or sum(d.norm().item() for d in self.state.fast) < 1e-6:
            return {"skipped": True, "reason": "no fast-weight changes to merge"}
        hold = self.holdout or self.conversation_holdout()
        res = consolidate(self.model, self.state.fast, hold, tol=0.0, conv_batches=self.conversation_holdout() if self.holdout else None)
        self.state.fast = res.pop("state")
        res.update(kind="merge", turn=self.state.turns, time=time.time())
        self.state.events.append(res)
        if res["accepted"]:
            self.save_checkpoint()
        return res

    def do_sleep(self, steps=20):
        T = self.cfg.seq_len
        r = self.state.recent[-T * 33:-T - 1]
        wins = [r[i:i + T + 1] for i in range(0, len(r) - T, T)]
        recent = [(torch.tensor(w[:-1], device=self.device)[None], torch.tensor(w[1:], device=self.device)[None]) for w in wins]
        res = sleep(self.model, recent, self.replay, self.holdout + self.conversation_holdout(), steps=steps,
                    params=self.model.trainable_parameters())
        res.update(kind="sleep", turn=self.state.turns, time=time.time())
        self.state.events.append(res)
        if res.get("accepted"):
            self.save_checkpoint()
        return res

    def save_checkpoint(self):
        self.model.save(self.ckpt, extra={"step": -1, "self_modified_at": time.time(), "system_prompt": self.system_prompt,
                                          "persona_name": self.persona_name})

    def status(self):
        d = self.state
        norms = [f"{t.flatten(2).norm(dim=-1).mean().item():.3f}" for t in d.fast] if d.fast is not None else []
        return {"persona": self.persona_name, "base": self.model.base_name, "turns": d.turns, "frozen": self.frozen,
                "tier": self.cfg.tier, "ema_loss": d.ema_loss, "fast_norms_per_layer": norms, "rollbacks": d.rollbacks,
                "consolidations": [e for e in d.events if e.get("kind") == "merge"][-3:],
                "sleeps": [e for e in d.events if e.get("kind") == "sleep"][-3:], "recent_tokens": len(d.recent)}

    # ---- loop -------------------------------------------------------------------------------------
    def run(self):
        print(f"[{self.persona_name} on {self.model.base_name}, tier {self.cfg.tier}, {self.model.n_params(True)/1e6:.1f}M self-modifying params, device {self.device}]")
        print("[/status /consolidate /sleep /reset /freeze /save /quit]")
        while True:
            try:
                user = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print()
                user = "/quit"
            if not user.strip():
                continue
            if user.startswith("/"):
                cmd = user.strip().split()[0]
                if cmd == "/quit":
                    self.state.save()
                    print("[self-state saved]")
                    return
                if cmd == "/status":
                    print(json.dumps(self.status(), indent=1, default=str))
                elif cmd == "/consolidate":
                    print(json.dumps(self.do_consolidate(), indent=1, default=str))
                elif cmd == "/sleep":
                    print(json.dumps(self.do_sleep(), indent=1, default=str))
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
            info = self.learn(self.turn_ids("user", user))
            answer, thinking = self.reply(user)
            if thinking and self.show_think:
                shown = thinking[:600] + ("..." if len(thinking) > 600 else "")
                print(f"  ({self.persona_name} thinking: {shown})")
            print(f"{self.persona_name}: {answer if answer else '[thinking used all the tokens; raise --max-new]'}")
            info2 = self.learn(self.turn_ids("assistant", self.last_full))
            self.state.turns += 1
            tag = []
            if info.get("rolled_back") or info2.get("rolled_back"):
                tag.append("rolled back")
            if "loss" in info2:
                tag.append(f"loss {info2['loss']:.2f} benefit forecast {info2['forecast']:+.3f} |d| {info2['delta_norm']:.2f}")
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
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--consolidate-every", type=int, default=8)
    ap.add_argument("--persona-name", default=None)
    ap.add_argument("--repetition-penalty", type=float, default=1.15, help="sampling setting; 1.0 disables it")
    ap.add_argument("--no-think", action="store_true", help="disable the base's thinking mode for this session")
    ap.add_argument("--hide-think", action="store_true", help="do not print the reasoning block")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    GraftChat(args.run, args.device or get_device(), tier=args.tier, temperature=args.temperature, max_new=args.max_new,
              consolidate_every=args.consolidate_every, persona_name=args.persona_name,
              repetition_penalty=args.repetition_penalty, think=not args.no_think, show_think=not args.hide_think).run()


if __name__ == "__main__":
    main()
