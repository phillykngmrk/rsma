# RSMA: Recursive Self-Modeling Architecture

The model built here is called **Sankofa**.

A self-modifying transformer. The model rewrites part of its own weights at inference time,
maintains a model of itself that predicts the consequences of those rewrites, and periodically
consolidates the rewrites into its permanent parameters.

## Three mechanisms

1. **Self-referential fast weights.** Selected layers hold a fast-weight matrix. At each chunk the
   matrix reads the input and produces its own output, query, key, and learning rate. It then
   updates itself with a delta rule so that it maps the key to what it currently maps the query to.
   The matrix that proposes the change is the matrix being changed (after Irie et al., 2022).
2. **Self-model gate.** A small network reads a compressed summary of the current fast-weight
   deltas and predicts the loss on the next chunk. It is trained on the loss that actually occurs.
   Its output gates the fast-weight learning rate, and at inference it triggers rollback when a
   modification is predicted to hurt.
3. **Consolidation.** Fast-weight deltas are periodically merged into the slow weights, verified on
   a held-out buffer, and either accepted or rolled back.

## Two ways to run it

**From scratch** (`rsma/model.py`): a small transformer trained here, with the fast-weight
sublayers in selected blocks. Clean science, small capability.

**Grafted** (`rsma/graft.py`): the same fast-weight sublayers and self-model inserted between the
blocks of a frozen open-weight base model (default Qwen2.5-0.5B-Instruct), plus low-rank adapters
on the base's attention so it can take on new voices. Only about 7M parameters are trainable, the
base's abilities are preserved at initialization, and every part of the self-modification loop is
shared with the from-scratch model. This is the path to a model that converses and reasons while
still rewriting itself from what it reads. The base is a flag: a larger base with a thinking mode
is the next step up in reasoning.

```
# corpus: ~170 figures across science, software, technology, finance, business, law, meditation,
# philosophy, crypto, history, education and writing (figures.json), from Gutenberg, Wikisource,
# BlackPast and YouTube transcripts
.venv/bin/python scripts/build_figures_corpus.py --skip-youtube     # public-domain sources
.venv/bin/python scripts/build_figures_corpus.py --yt-sleep 4       # add talks and interviews

# train the graft on the blended corpus
.venv/bin/python -m rsma.train --task figures --graft Qwen/Qwen2.5-0.5B-Instruct --name mentor \
    --steps 600 --stream 4 --batch 2 --seq-len 512 --chunk 32 --fast-layers 5,11,17,23 --lr 2e-4

# talk to Sankofa (persistent self-state in runs/mentor/self/, consolidation rewrites ckpt.pt)
.venv/bin/python -m rsma.chat_graft --run mentor --tier 3 --persona-name Sankofa

# let it study on its own: feeds, Wikipedia, arXiv from study.json; audit log in runs/mentor/study/
.venv/bin/python -m rsma.study --run mentor --once
.venv/bin/python -m rsma.study --run mentor --every 6h
```

The study loop is how the model learns without you: it reads new material through its fast
weights, the self-model rolls back modifications it forecasts as harmful, verified changes are
consolidated into the slow weights, and a sleep pass with corpus replay fine-tunes the adapters.
Nothing that regresses the held-out set is kept, and every cycle is logged.

## Persistence tiers

| Tier | Fast weights | Meaning |
|------|--------------|---------|
| 1 | reset every sequence | in-context adaptation only |
| 2 | persist across sequences | the running model drifts, with snapshots and rollback |
| 3 | merged into slow weights | the model permanently rewrites itself |

Tier is a config flag. Tier 3 is the target.

## Layout

```
rsma/
  config.py        configs and tier flag
  fastweights.py   self-referential fast-weight layer
  selfmodel.py     self-model head and gate
  model.py         RSMA transformer, snapshots, rollback
  consolidate.py   tier 3 merge-and-verify
  data/            synthetic rule-switch task, char-level text
  train.py         meta-training loop
  evaluate.py      adaptation, perplexity-by-position, self-model calibration
tests/             unit tests
scripts/           run scripts
```

## Workflow

```
# validate the mechanism on the rule-switch task: self-modifying vs frozen
.venv/bin/python -m rsma.train --task synthetic --name syn_fast
.venv/bin/python -m rsma.train --task synthetic --name syn_nofast --no-fast
.venv/bin/python -m rsma.evaluate --run syn_fast --baseline syn_nofast --stream 40 --tier 3

# build the Malcolm X corpus (collected speeches, debates and interviews 1960-1965)
.venv/bin/pip install pypdf && .venv/bin/python scripts/build_malcolmx_corpus.py

# train the text model on it (any UTF-8 file works with --corpus)
.venv/bin/python -m rsma.train --task text --corpus data_cache/malcolmx.txt --name malcolmx --steps 5000

# talk to it. Fast weights persist in runs/malcolmx/self/, consolidation rewrites ckpt.pt
.venv/bin/python -m rsma.chat --run malcolmx --tier 3
```

Chat commands: `/status`, `/consolidate`, `/sleep`, `/reset`, `/freeze`, `/save`, `/quit`.

The chat uses the corpus's own interview labels, `Question:` for you and `Malcolm X:` for the
model (override with `--user-label` / `--model-label`). Each turn passes through the model with
persistent fast weights. A modification is rolled back when the self-model forecasts a higher loss
with it than without it. Every 8 turns the fast weights are merged into the slow weights if that
does not regress a held-out mix of corpus and recent conversation; an accepted merge rewrites
`ckpt.pt`. `scripts/chat_smoke.py` exercises the whole loop without a terminal.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
```

## Results so far

### From-scratch 19M model, BPE tokens (2026-09-11)

Pretrained 1500 stream steps on WikiText-103 + Gutenberg + Malcolm X, fine-tuned 300 steps on
Malcolm X with Gutenberg replay. The strongest memory result so far, and the weakest language:

| Metric | value |
|---|---|
| loss on first 32 tokens of later windows, memory carried vs reset | 4.02 vs 4.46 |
| persistent stream (tier 3, 16 windows), carried vs reset | 3.87 vs 3.89 |
| self-model benefit forecast correlation | 0.93 |
| sample quality | word salad |

Memory and self-modeling work at this scale; language does not. That is the trade the grafted
model resolves, by borrowing language from a pretrained base.

### Grafted model (2026-09-11)

Qwen2.5-0.5B-Instruct frozen, 4 fast-weight layers at blocks 5/11/17/23, LoRA rank 16 on
attention, 7.2M trainable parameters. 600 stream steps on the blended figures corpus.

- Base ability preserved at initialization: loss 3.016 base vs 3.035 grafted on held-out text.
- After training the voice shifts toward the corpus (investing answers read like Buffett, mind
  answers like the meditation teachers) while answers stay coherent.
- Carried fast state helps early tokens of later windows by 0.064 nats on corpus text; the
  self-model's benefit forecast correlates 0.41 with the measured benefit.
- Study loop: one cycle reads ~30 new documents (feeds, Wikipedia, arXiv) in about 70 s; the sleep
  pass was accepted (held-out 3.368 to 3.334) and consolidation correctly rejected.
- Honest limit: at 600 steps (2.4M tokens) the fast path's influence on outputs is still small, so
  merges into the slow weights are rejected. A longer run is queued.

## Results so far (2026-09-10)

### Second architecture: direct value path, forget gate, benefit-forecasting self-model

Same synthetic task and training regime as below (5.5M parameters, 4-window streams).

| Metric | v2 self-modifying, memory carried | v2, memory reset | v1, memory carried | frozen baseline |
|---|---|---|---|---|
| loss on windows 2-4 | **1.924** | 2.280 | 2.179 | 2.194 |
| loss on first 32 tokens of windows 2-4 | **2.106** | 3.009 | 2.990 | 3.006 |
| self-model forecast correlation | **0.57** | | ~0.02 | |

The early-token gap went from 0.02 to 0.90 nats: the model now carries the current rule across
the attention window and reads it back almost immediately. The self-model's forecast of how much
the memory will help correlates at 0.57 with the measured benefit, up from chance. The
in-context learning transition also arrived 500 steps earlier.

What changed: the write target is now the read plus a direct projection of the input (v1 could
only store what it already read back); the self-model controls a per-head forget rate as well
as the write rate; and the self-model predicts the benefit of the carried state (loss reset
minus loss carried, measured with a second forward pass) instead of absolute loss.

### First architecture


Synthetic rule-switch task, 5.3M parameters, trained on streams of 4 windows with the fast
state carried across windows and gradient flowing through the whole stream. The frozen
baseline has the same architecture with the fast-weight sublayers removed.

| Metric | self-modifying, state carried | self-modifying, state reset | frozen baseline |
|---|---|---|---|
| loss on windows 2-4 | 2.179 | 2.238 | 2.194 |
| loss on first 32 tokens of windows 2-4 | 2.990 | 3.013 | 3.006 |
| single-window validation loss | 2.502 | | 2.419 |

With its memory the self-modifying model beats the frozen baseline by a small margin; without
it, it is worse. The fast path costs some within-window accuracy and buys some cross-window memory.

The model stores something about the current rule in its own weights and reads it back after
the attention window has moved on. The effect is consistent but small against the roughly one
nat available, so writing is still the weak link.

The self-model's gate closed from 0.96 to 0.43 at the moment in-context learning emerged. Its
loss forecast is near chance on this task because switches cannot be predicted from the past.

Tier 3 on a 24-sequence stream: three consolidations, the first improved held-out loss
(2.537 to 2.503), the next two regressed slightly. Acceptance is now strict (no regression).

Three bugs had to be fixed before any of this worked, each confirmed by a diagnostic first:
softmax keys were orthogonal to the layer-normed input (stored deltas changed outputs by ~1%);
summed per-chunk updates over-corrected and collapsed the delta to rank one; and detaching
state between windows removed the learning signal for what to write.

Malcolm X text model (character level, 5.3M parameters, 2500 stream steps): validation loss
1.18 nats per character. Carried state helps by only 0.003 on early tokens, so at this scale the
memory is nearly inert on text. Three tier 3 consolidations on a 24-window stream: two accepted
at quarter strength with tiny gains, one rejected. In the chat loop, merges of conversation-derived
fast weights into the slow weights were accepted when they improved held-out conversation loss.
