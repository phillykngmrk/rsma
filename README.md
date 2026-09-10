# RSMA: Recursive Self-Modeling Architecture

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

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
```

## Results so far (2026-09-10)

Synthetic rule-switch task, 5.3M parameters, trained on streams of 4 windows with the fast
state carried across windows and gradient flowing through the whole stream.

| Metric | with carried fast weights | fast weights reset |
|---|---|---|
| loss on windows 2-4 | 2.161 | 2.217 |
| loss on first 32 tokens of windows 2-4 | 2.999 | 3.020 |

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
