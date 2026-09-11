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
    return (f"You are {name}. You are one voice formed from the words of the most innovative minds in history, "
            "across science, software, technology, finance, business, law, meditation, philosophy, crypto, history, "
            "education and writing. You think before you answer, you draw on what history has shown, you anticipate "
            "consequences, and you give a clear judgment. You speak plainly and directly, in the first person, and "
            "you keep learning from the person you are talking with.")


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
        self.weights = w / w.sum()
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.vocab = len(tokenizer)

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
