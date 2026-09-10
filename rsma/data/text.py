"""Character-level text. Downloads TinyShakespeare on first use."""
import os
import urllib.request
import torch

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class CharText:
    def __init__(self, seq_len=256, cache_dir="data_cache", split=0.9, seed=0, device="cpu", corpus=None, vocab_chars=None):
        os.makedirs(cache_dir, exist_ok=True)
        path = corpus or os.path.join(cache_dir, "tinyshakespeare.txt")
        if not os.path.exists(path):
            urllib.request.urlretrieve(URL, path)
        text = open(path, encoding="utf-8").read()
        chars = list(vocab_chars) if vocab_chars else sorted(set(text))
        if vocab_chars:
            keep = set(chars)
            text = "".join(c for c in text if c in keep)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = chars
        self.vocab = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * split)
        self.train, self.val = data[:n], data[n:]
        self.seq_len = seq_len
        self.g = torch.Generator().manual_seed(seed)
        self.device = device

    def batch(self, batch, split="train"):
        d = self.train if split == "train" else self.val
        ix = torch.randint(0, len(d) - self.seq_len - 1, (batch,), generator=self.g)
        x = torch.stack([d[i:i + self.seq_len] for i in ix])
        y = torch.stack([d[i + 1:i + 1 + self.seq_len] for i in ix])
        return x.to(self.device), y.to(self.device)

    def stream(self, batch, n_seq, split="val"):
        """Contiguous stream: consecutive sequences from one place in the corpus, for tier 2/3."""
        d = self.train if split == "train" else self.val
        span = n_seq * self.seq_len
        start = torch.randint(0, len(d) - span - 1, (batch,), generator=self.g)
        for s in range(n_seq):
            x = torch.stack([d[i + s * self.seq_len:i + (s + 1) * self.seq_len] for i in start])
            y = torch.stack([d[i + s * self.seq_len + 1:i + (s + 1) * self.seq_len + 1] for i in start])
            yield x.to(self.device), y.to(self.device)

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]
