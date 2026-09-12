"""
Token streams for the blended-figures model, using the base model's own tokenizer.

Each figure's text file (from scripts/build_figures_corpus.py) is split into documents, each
document is wrapped in the base model's chat format as an assistant turn under the blended
persona's system prompt, tokenized, and concatenated into one token array per figure. Batches
draw figures with weight proportional to sqrt(tokens), so large public-domain corpora do not
drown out figures with only a few transcripts.
"""
import hashlib
import json
import os
import re

import numpy as np
import torch

FIG_DIR = os.path.join("data_cache", "figures")
TOK_DIR = os.path.join("data_cache", "figures_tok")


def persona_prompt(name):
    """A factual statement of what the model is. No instructions about how to speak or what to say."""
    return (f"You are {name}, a self-modifying model formed from the words of the most innovative minds in history, "
            "across science, software, technology, finance, business, law, meditation, philosophy, crypto, history, "
            "education and writing. Part of your weights rewrites itself as you read and as you talk. A self-model "
            "inside you forecasts whether each change helps and rolls it back otherwise. Changes that prove "
            "themselves are consolidated into your permanent weights. In a study loop you read new material and "
            "sleep on it with replay of your corpus.")


SELF_KNOWLEDGE = os.path.join("data", "sankofa_self.json")


class FigureText:
    def __init__(self, tokenizer, seq_len=512, fig_dir=FIG_DIR, persona_name=None, split=0.98, seed=0,
                 device="cpu", weighting="sqrt", min_tokens=2000):
        self.tok = tokenizer
        self.seq_len = seq_len
        cov = json.load(open(os.path.join(fig_dir, "coverage.json")))
        manifest_name = json.load(open("figures.json")).get("persona_name") if os.path.exists("figures.json") else None
        self.persona_name = persona_name or manifest_name or cov.get("persona_name", "Sankofa")
        self.system_prompt = persona_prompt(self.persona_name)
        os.makedirs(TOK_DIR, exist_ok=True)
        self.names, self.train, self.val, self.sizes, self.domains = [], {}, {}, {}, {}
        for name, info in cov["figures"].items():
            path = info["file"]
            if not os.path.exists(path) or info["chars"] < 1000:
                continue
            arr = self._tokens(name, path)
            if len(arr) < min_tokens:
                continue
            k = int(len(arr) * split)
            self.names.append(name)
            self.train[name], self.val[name] = arr[:k], arr[k:]
            self.sizes[name] = len(arr)
            self.domains[name] = info["domain"]
        sizes = np.array([self.sizes[n] for n in self.names], dtype=np.float64)
        w = np.sqrt(sizes) if weighting == "sqrt" else sizes
        w = w / w.sum()
        if os.path.exists(SELF_KNOWLEDGE):
            arr = self._self_tokens()
            if len(arr) >= min_tokens // 4:
                # a fixed 4% of batches teach the model what it is
                self.names.append("__self__")
                self.train["__self__"] = self.val["__self__"] = arr
                self.sizes["__self__"] = len(arr)
                self.domains["__self__"] = "self"
                w = np.append(w * 0.96, 0.04)
        self.weights = w
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.vocab = len(tokenizer)
        self.facts, self.fact_frac = None, 0.0

    def with_fact_streams(self, frac=0.25, seed=0):
        """Mix in tell-then-ask dialogues (rsma.data.factstreams) for a fraction of training streams."""
        from .factstreams import FactStreams
        self.facts = FactStreams(self.tok, self.system_prompt, seq_len=self.seq_len, seed=seed, device=self.device)
        self.fact_frac = frac
        return self

    def _tokens(self, name, path):
        raw = open(path, encoding="utf-8").read()
        key = hashlib.md5((raw[:2000] + str(len(raw)) + self.system_prompt + getattr(self.tok, "name_or_path", "")).encode()).hexdigest()[:12]
        cache = os.path.join(TOK_DIR, f"{re.sub(r'[^a-z0-9]+', '_', name.lower())}_{key}.npy")
        if os.path.exists(cache):
            return np.load(cache)
        docs = [d.strip() for d in re.split(r"\n### [^\n]+\n", raw) if d.strip()]
        ids = []
        for d in docs:
            msgs = [{"role": "system", "content": self.system_prompt}, {"role": "assistant", "content": d}]
            text = self.tok.apply_chat_template(msgs, tokenize=False)
            ids.extend(self.tok(text, add_special_tokens=False).input_ids)
        arr = np.array(ids, dtype=np.uint32)
        np.save(cache, arr)
        return arr

    def _self_tokens(self):
        """Self-knowledge dialogues, tokenized in chat format, repeated to fill a window comfortably."""
        pairs = json.load(open(SELF_KNOWLEDGE))
        key = hashlib.md5((json.dumps(pairs) + self.system_prompt).encode()).hexdigest()[:12]
        cache = os.path.join(TOK_DIR, f"__self___{key}.npy")
        if os.path.exists(cache):
            return np.load(cache)
        ids = []
        rng = np.random.default_rng(0)
        for _ in range(8):
            order = rng.permutation(len(pairs))
            for i in order:
                q, a = pairs[i]["user"], pairs[i]["assistant"].format(name=self.persona_name)
                msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": q}, {"role": "assistant", "content": a}]
                ids.extend(self.tok(self.tok.apply_chat_template(msgs, tokenize=False), add_special_tokens=False).input_ids)
        arr = np.array(ids, dtype=np.uint32)
        np.save(cache, arr)
        return arr

    def _pick(self):
        return self.names[self.rng.choice(len(self.names), p=self.weights)]

    def _window(self, arr, start, T):
        x = torch.from_numpy(arr[start:start + T].astype(np.int64))
        y = torch.from_numpy(arr[start + 1:start + 1 + T].astype(np.int64))
        return x, y

    def batch(self, batch, split="train"):
        T = self.seq_len
        xs, ys = [], []
        for _ in range(batch):
            n = self._pick()
            d = self.train[n] if split == "train" else self.val[n]
            if len(d) < T + 2:
                d = self.train[n]
            i = int(self.rng.integers(0, len(d) - T - 1))
            x, y = self._window(d, i, T)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)

    def stream(self, batch, n_seq, split="train"):
        if getattr(self, "facts", None) is not None and split == "train" and self.rng.random() < self.fact_frac:
            yield from self.facts.stream(batch, n_seq, split)
            return
        T = self.seq_len
        span = n_seq * T
        picks = []
        for _ in range(batch):
            n = self._pick()
            d = self.train[n] if split == "train" else self.val[n]
            if len(d) < span + 2:
                d = self.train[n]
            picks.append((d, int(self.rng.integers(0, len(d) - span - 1))))
        for s in range(n_seq):
            xs, ys = zip(*[self._window(d, i + s * T, T) for d, i in picks])
            yield torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)

    def summary(self):
        by_dom = {}
        for n in self.names:
            by_dom[self.domains[n]] = by_dom.get(self.domains[n], 0) + self.sizes[n]
        return {"figures": len(self.names), "tokens": int(sum(self.sizes.values())), "by_domain": by_dom}
