"""
Autonomous study. Each cycle the model gathers new material from the sources in study.json,
reads it through its fast weights with the self-model vetting every modification, consolidates
verified changes into its slow weights, runs a sleep pass with replay, and appends an audit
record. Nothing is kept that makes the model worse on its held-out set.

  python -m rsma.study --run mentor --once
  python -m rsma.study --run mentor --every 6h

Audit log: runs/<run>/study/log.jsonl. Seen sources: runs/<run>/study/seen.json.
To run nightly without this terminal: schedule the --once form with launchd or cron.
"""
import argparse
import json
import os
import time

import torch

from .chat_graft import GraftChat
from .data.web import get, strip_html, feed_entries, wikipedia_search, wikipedia_text, arxiv_search
from .train import get_device


class Study:
    def __init__(self, run, device, config_path="study.json"):
        self.chat = GraftChat(run, device, tier=3)
        self.cfg = json.load(open(config_path))
        self.dir = os.path.join("runs", run, "study")
        os.makedirs(self.dir, exist_ok=True)
        self.seen_path = os.path.join(self.dir, "seen.json")
        self.seen = set(json.load(open(self.seen_path))) if os.path.exists(self.seen_path) else set()

    # ---- gather ----------------------------------------------------------------------------------
    def gather(self):
        docs, cap = [], self.cfg.get("max_chars_per_doc", 12000)

        def add(key, source, text):
            if key in self.seen or not text or len(text) < 500:
                return
            docs.append({"key": key, "source": source, "text": text[:cap]})

        for url in self.cfg.get("feeds", []):
            try:
                for title, link, summary in feed_entries(url, limit=8):
                    if link in self.seen:
                        continue
                    try:
                        body = strip_html(get(link, timeout=30))
                    except Exception:
                        body = summary
                    add(link, f"feed {title}", body if len(body) > len(summary) else summary)
            except Exception as e:
                print(f"  ! feed {url}: {type(e).__name__}")
        for topic in self.cfg.get("topics", []):
            try:
                for title in wikipedia_search(topic, limit=self.cfg.get("wikipedia_per_topic", 2)):
                    add("wiki:" + title, f"wikipedia {title}", wikipedia_text(title))
            except Exception as e:
                print(f"  ! wikipedia {topic}: {type(e).__name__}")
        for cat in self.cfg.get("arxiv", []):
            try:
                for title, aid, abstract in arxiv_search("cat:" + cat, limit=self.cfg.get("arxiv_per_category", 5)):
                    add(aid, f"arxiv {title}", f"{title}\n\n{abstract}")
            except Exception as e:
                print(f"  ! arxiv {cat}: {type(e).__name__}")
        return docs[: self.cfg.get("max_candidates", 60)]

    # ---- read -------------------------------------------------------------------------------------
    def read(self, doc):
        """Feed one document through the model as study material, window by window."""
        tok = self.chat.tok
        ids = tok(f"<|im_start|>user\n[Study material: {doc['source']}]\n{doc['text']}<|im_end|>\n", add_special_tokens=False).input_ids
        T = self.chat.cfg.seq_len
        losses, rollbacks = [], 0
        for i in range(0, len(ids), T):
            info = self.chat.learn(ids[i:i + T])
            if "loss" in info:
                losses.append(info["loss"])
                rollbacks += int(info["rolled_back"])
        self.seen.add(doc["key"])
        return {"tokens": len(ids), "mean_loss": sum(losses) / max(len(losses), 1), "rollbacks": rollbacks}

    # ---- self-model-directed selection ------------------------------------------------------------
    @torch.no_grad()
    def rank(self, docs):
        """Let the self-model choose. For each candidate, read its first window WITHOUT keeping the
        modification and take the self-model's forecast of how much the resulting state would help
        on what comes next. Higher forecast benefit reads first; the rest wait for a later cycle."""
        chat = self.chat
        T = chat.cfg.seq_len
        C = chat.cfg.chunk_size
        scored = []
        if chat.state.fast is None:
            chat.state.fast = chat.model.init_state(1, chat.device)
        for d in docs:
            ids = chat.tok(f"<|im_start|>user\n[Study material: {d['source']}]\n{d['text']}<|im_end|>\n", add_special_tokens=False).input_ids[:T]
            ids = ids[: (len(ids) // C) * C]
            if len(ids) < C:
                continue
            x = torch.tensor(ids, device=chat.device)[None]
            _, _, aux = chat.model(x, chat.state.fast)  # state not kept: a probe, not a read
            d["forecast"] = aux["pred_benefit"][0, -1].item()
            scored.append(d)
        scored.sort(key=lambda d: d["forecast"], reverse=True)
        return scored

    # ---- cycle ------------------------------------------------------------------------------------
    def cycle(self):
        t0 = time.time()
        candidates = self.gather()
        ranked = self.rank(candidates)
        k = self.cfg.get("max_docs", 30)
        docs, deferred = ranked[:k], ranked[k:]
        print(f"[study] {len(candidates)} candidates, self-model chose {len(docs)}, deferred {len(deferred)}")
        if docs:
            print(f"        forecast benefit: best {docs[0]['forecast']:+.4f}, worst chosen {docs[-1]['forecast']:+.4f}")
        reads = []
        for d in docs:
            r = self.read(d)
            r["source"] = d["source"][:80]
            reads.append(r)
            print(f"  read {r['tokens']:5d} tok  loss {r['mean_loss']:.2f}  rollbacks {r['rollbacks']}  {r['source']}")
        merge = self.chat.do_consolidate() if self.cfg.get("consolidate", True) else {"skipped": True}
        nap = self.chat.do_sleep(steps=self.cfg.get("sleep_steps", 20)) if docs else {"skipped": True}
        self.chat.state.save()
        json.dump(sorted(self.seen), open(self.seen_path, "w"))
        rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0), "docs": len(docs),
               "candidates": len(candidates), "forecasts": [round(d["forecast"], 4) for d in docs],
               "tokens": sum(r["tokens"] for r in reads), "rollbacks": sum(r["rollbacks"] for r in reads),
               "consolidate": {k: v for k, v in merge.items() if k != "tried"}, "sleep": nap, "sources": [r["source"] for r in reads]}
        with open(os.path.join(self.dir, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(f"[study] consolidate: {merge.get('accepted', 'skipped')}  sleep: {nap.get('accepted', 'skipped')}  "
              f"holdout {nap.get('before', float('nan')):.4f} -> {nap.get('after', float('nan')):.4f}  ({rec['seconds']}s)")
        return rec


def parse_every(s):
    units = {"m": 60, "h": 3600, "d": 86400}
    return float(s[:-1]) * units[s[-1]] if s[-1] in units else float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="study.json")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--every", default=None, help="e.g. 6h, 30m; loop forever")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    st = Study(args.run, args.device or get_device(), args.config)
    if args.every and not args.once:
        period = parse_every(args.every)
        while True:
            st.cycle()
            time.sleep(period)
    else:
        st.cycle()


if __name__ == "__main__":
    main()
