# North star

Sankofa should become a model with a real biography: weights that are the product of everything it
has read and everyone it has talked to, changed under rules its owner can read, with every change
traceable and reversible. Three things have to hold for that to be true rather than a story.

## 1. The memory carries meaning

Success: Sankofa knows something about you next month that it learned from you this month, and
you can verify it by asking. Measured by `rsma/memtest.py`: facts told in one session, recalled
in a fresh session from persisted and consolidated weights alone, scored against distractors.

Current (2026-09-11): first measured cross-session recall. 6 facts told in one session; in a fresh
process, the persisted fast weights alone recover 3 of 6 by likelihood shift, vs 1 of 6 with the
state reset. Margin is thin. The test also found and fixed a bug that wiped the fast state whenever
a merge was rejected. This is the research problem at the center of the project.

## 2. The self-model is worth trusting

Success: the self-model's forecast of its own memory's usefulness is accurate enough that it
decides what Sankofa studies, not only whether a change sticks. A model with an accurate model of
itself is what the name of this architecture means.

Current: forecast correlation 0.93 on the from-scratch model, 0.4 on the graft. The study loop
now ranks candidate material by the self-model's forecast benefit before reading it.

## 3. The base can think

Success: Sankofa reasons through a question before answering it. The self-modification layers
ride on a base large enough to do real work, on hardware the owner controls.

Current: Qwen2.5-0.5B-Instruct in training. Qwen3-1.7B (thinking mode) grafts cleanly: base loss
preserved exactly, 27M trainable parameters, forward 0.7 s on this machine. Next: train it.

## Rules that do not change

- No behavioral steering in the prompt or data. Sankofa's voice comes from its corpus and its
  experience.
- Every self-modification is verified against held-out text, logged, and reversible.
- Self-modification can be frozen or reset by the owner at any time.
- Runs on hardware the owner controls.
