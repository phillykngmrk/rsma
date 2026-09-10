"""
Byte-level BPE tokens over a mix of sources. Each source is a uint16 token file; batches are
drawn from sources with configurable weights so a small persona corpus can be oversampled.

  build:  python scripts/build_pretrain_corpus.py     (writes data_cache/tokens/*.bin + tokenizer.json)
  use:    TokenText(seq_len, sources={"wikitext": 0.6, "gutenberg": 0.2, "malcolmx": 0.2})
"""
import json
import os

import numpy as np
import torch
from tokenizers import Tokenizer

TOK_DIR = os.path.join("data_cache", "tokens")


class TokenText:
    def __init__(self, seq_len=512, sources=None, tok_dir=TOK_DIR, split=0.98, seed=0, device="cpu"):
        self.tok = Tokenizer.from_file(os.path.join(tok_dir, "tokenizer.json"))
        self.vocab = self.tok.get_vocab_size()
        meta = json.load(open(os.path.join(tok_dir, "meta.json")))
        if sources is None:
            sources = {name: 1.0 for name in meta["sources"]}
        self.names = list(sources)
        w = np.array([sources[n] for n in self.names], dtype=np.float64)
        self.weights = w / w.sum()
        self.train, self.val = {}, {}
        for n in self.names:
            arr = np.fromfile(os.path.join(tok_dir, f"{n}.bin"), dtype=np.uint16)
            k = int(len(arr) * split)
            self.train[n], self.val[n] = arr[:k], arr[k:]
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.sizes = {n: len(self.train[n]) for n in self.names}

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
            i = int(self.rng.integers(0, len(d) - T - 1))
            x, y = self._window(d, i, T)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)

    def stream(self, batch, n_seq, split="train"):
        """n_seq consecutive windows per row, each row from one contiguous place in one source."""
        T = self.seq_len
        span = n_seq * T
        picks = []
        for _ in range(batch):
            n = self._pick()
            d = self.train[n] if split == "train" else self.val[n]
            picks.append((d, int(self.rng.integers(0, len(d) - span - 1))))
        for s in range(n_seq):
            xs, ys = zip(*[self._window(d, i + s * T, T) for d, i in picks])
            yield torch.stack(xs).to(self.device), torch.stack(ys).to(self.device)

    def encode(self, s):
        return self.tok.encode(s).ids

    def decode(self, ids):
        return self.tok.decode(list(ids))
