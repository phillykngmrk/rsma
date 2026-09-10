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

## Setup

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests
```
