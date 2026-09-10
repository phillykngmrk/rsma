"""
Build the pretraining mix: WikiText-103 (first shard), Project Gutenberg texts (American
political prose and slave narratives), and the Malcolm X speeches. Trains a byte-level BPE
tokenizer on the mix and writes uint16 token files.

  .venv/bin/python scripts/build_pretrain_corpus.py --vocab 8192
"""
import argparse
import glob
import json
import os
import re

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

CACHE = "data_cache"
OUT = os.path.join(CACHE, "tokens")


def gutenberg_text(path):
    t = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", t)
    if m:
        t = t[m.end():]
    m = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
    if m:
        t = t[:m.start()]
    t = t.replace("\r\n", "\n")
    # unwrap hard-wrapped paragraphs
    paras = re.split(r"\n\s*\n", t)
    paras = [re.sub(r"\s*\n\s*", " ", p).strip() for p in paras]
    return "\n\n".join(p for p in paras if p)


def wikitext_text(path, max_chars):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    out, n = [], 0
    for chunk in table.column("text").to_pylist():
        if not chunk or chunk.startswith(" = "):
            continue
        chunk = chunk.strip()
        # undo wikitext's tokenized spacing
        chunk = re.sub(r" @(.)@ ", r"\1", chunk)
        chunk = re.sub(r" ([,.;:!?')\]])", r"\1", chunk)
        chunk = re.sub(r"([(\[]) ", r"\1", chunk)
        chunk = chunk.replace(" n't", "n't").replace(" 's", "'s")
        out.append(chunk)
        n += len(chunk)
        if n >= max_chars:
            break
    return "\n\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=8192)
    ap.add_argument("--wikitext-chars", type=int, default=200_000_000)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    sources = {}
    sources["malcolmx"] = open(os.path.join(CACHE, "malcolmx.txt")).read()
    sources["gutenberg"] = "\n\n".join(gutenberg_text(p) for p in sorted(glob.glob(os.path.join(CACHE, "gutenberg", "*.txt"))))
    wt = os.path.join(CACHE, "wikitext103-train-0.parquet")
    if os.path.exists(wt):
        sources["wikitext"] = wikitext_text(wt, args.wikitext_chars)
    for k, v in sources.items():
        print(f"{k}: {len(v):,} chars")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    # train on a balanced sample so the persona corpus shapes the vocabulary
    def sample(text, cap):
        return [text[i:i + 100_000] for i in range(0, min(len(text), cap), 100_000)]
    tok.train_from_iterator(sample(sources["malcolmx"], 10_000_000) + sample(sources["gutenberg"], 10_000_000)
                            + sample(sources.get("wikitext", ""), 30_000_000), trainer=trainer)
    tok.save(os.path.join(OUT, "tokenizer.json"))

    meta = {"vocab": tok.get_vocab_size(), "sources": {}}
    for k, v in sources.items():
        ids = []
        for i in range(0, len(v), 1_000_000):
            ids.extend(tok.encode(v[i:i + 1_000_000]).ids)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(os.path.join(OUT, f"{k}.bin"))
        meta["sources"][k] = int(len(arr))
        print(f"{k}: {len(arr):,} tokens ({len(v)/max(len(arr),1):.2f} chars/token)")
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
