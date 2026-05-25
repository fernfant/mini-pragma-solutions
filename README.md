# mini-pragma — capstone solution

Reference implementation of the [Lesson 6 capstone](https://github.com/fernfant/mini-pragma/blob/main/course/06_capstone.md)
from [**fernfant/mini-pragma**](https://github.com/fernfant/mini-pragma) — a
hello-world Transformer course based on Revolut's PRAGMA paper.

> 👉 **Doing the capstone?** Try it yourself before reading this. Peek only
> when you're truly stuck — the learning is in the struggle.

## Files

| File | What it does |
|---|---|
| `capstone_data.py` | Generates 2000 synthetic users (95% normal, 5% fraud), saves to `data.pt`. |
| `capstone_pretrain.py` | Pre-trains a tiny PRAGMA-style encoder with MLM on all events. Saves encoder to `encoder.pt`. |
| `capstone_classify.py` | Loads the frozen encoder, trains a linear head, evaluates. Also trains a random-init baseline for comparison, across several labelled-data sizes. |
| `run_all.sh` | One command to run all three in order. |

## Run

```bash
pip3 install torch
bash run_all.sh
```

End-to-end runtime: ~30 seconds on CPU.

## Expected results

```
 train | pretrained acc pretrained recall | baseline acc baseline recall
--------------------------------------------------------------------------------
    20 |          1.000             1.000 |        0.960           0.333
    50 |          1.000             1.000 |        0.955           0.250
   200 |          1.000             1.000 |        0.990           0.833
  1000 |          1.000             1.000 |        1.000           1.000
```

The classifier on top of the pre-trained encoder reaches ~100% fraud recall
even with just 20 labelled users. The random-init baseline collapses to
predicting "normal" for everything in the low-label regime (recall ≈ 25-33%)
and only catches up at ~1000 labels.

This is the foundation-model pitch in miniature: **pre-training trades cheap
unlabelled data for expensive labels**.

## Design notes

- **Fraud signal**: bursts of `(purchase, tiny, night)` events with rare
  `login`s — a *contextual* pattern, not just a frequency one. Important: if
  fraud were detectable from raw token counts alone, attention and
  pre-training wouldn't help.
- **User-level pooling**: mean over all output positions. PRAGMA uses a
  prepended `[USR]` token; mean-pooling works fine at this scale.
- **Tiny model**: 32-d, 2 layers, 2 heads. ~5k parameters. Trains in seconds.
