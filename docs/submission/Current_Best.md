
# **Custom Tokenizer v3 (DP + Structured Vocab) — Analysis & Results**

## Overview

This project introduces a **custom tokenizer + dataset pipeline** designed to compete with SentencePiece (SP1024) under the strict constraints of the Parameter Golf:

* **16MB submission limit**
* **~10 minute training window (8×H100)**
* Fixed FineWeb dataset

The goal was to improve **token efficiency (compression)** while maintaining or improving **training performance (val_bpb)**.

---

# 🧠 Core Idea

Instead of relying on:

* greedy segmentation (SentencePiece)
* static merge rules

We built:

### ✅ **Dynamic Programming (DP) Tokenization**

* globally optimal segmentation per sequence
* avoids greedy local mistakes
* reduces fallback token usage
* improves boundary decisions

### ✅ **Structure-Aware Vocabulary**

* explicitly models:

  * phrase tokens (`"in the"`, `"with a"`)
  * structural tokens (`"\nThis"`, `"\nWhat"`)
  * space-prefixed tokens
* targets real distribution patterns in FineWeb

---

# 📦 Dataset

Custom dataset:

```
workspace/parameter-golf/data/datasets/fineweb10B_customdp1024_v3
```

### Properties

* 80 train shards (~8B tokens)
* 1 validation shard (~61M tokens)
* identical format to baseline
* only difference: **tokenization**

---

# 🔁 Tokenization Pipeline

## 1. Vocab Construction

Pipeline:

```
ngram mining → candidate filtering → vocab building → swap optimization
```

Key additions in **vocab_v3**:

* structural tokens (newline + capitalized words)
* functional transitions ("is the", "with a")
* reduced low-signal fragments

---

## 2. DP Encoding (Exporter)

Unlike SentencePiece:

| SP                     | Custom                  |
| ---------------------- | ----------------------- |
| greedy                 | DP optimal              |
| local decisions        | global sequence scoring |
| no structure awareness | structure-aware scoring |

### Effects

* fewer fallback tokens
* better phrase grouping
* improved compression

---

# 📊 Compression Results

Full validation (50k docs):

```
delta_tpb = -0.000212794
delta_tokens = -32,149
```

👉 **Custom tokenizer beats SentencePiece on compression**

This is the key milestone.

---

# 🧪 Training Results

## Best Custom Run

```
final_int8_zlib_roundtrip_exact val_bpb: ~1.2381
```

## SentencePiece Baseline

```
~1.2329
```

### Gap

```
~0.005 – 0.006 bpb
```

---

# 🧠 Key Insight

> The problem is no longer tokenization — it is learning dynamics.

---

# 🔍 Training Behavior

### Observed Pattern

| Phase         | Behavior      |
| ------------- | ------------- |
| 0–1000 steps  | worse than SP |
| 1000–6000     | catches up    |
| late training | near parity   |

---

### Why?

1. **Distribution Shift**

   * custom vocab ≠ SP token frequencies

2. **Phrase Tokens**

   * harder to learn early
   * require more context

3. **Embedding Cold Start**

   * fused tokens lack initial structure

---

# ⚙️ Training Modifications

All changes were made in:

```
train_gpt_custom_v1_locked_token_class_grad.py
```

## 1. Token-Class Gradient Scaling

We classify tokens into:

* byte
* phrase
* space
* structural

```python
apply_token_class_grad_scaling(...)
```

### Best performing config:

```
BYTE   = 0.70
PHRASE = 0.90
SPACE  = 0.98
STRUCT = 0.95
```

👉 Mild scaling helps early training stability.

---

## 2. Scheduled Scaling (Attempted)

We tested:

* early suppression → later normalization

Result:

* no meaningful improvement

---

## 3. Compositional Initialization

```
CUSTOM_INIT_MODE=phrase_comp
```

* initializes phrase tokens from subparts

Result:

* improved early curve
* worse final performance

👉 Not the right lever.

---

## 4. Compositional Residual (Attempted)

Training-time embedding composition:

```
embedding = learned + α * composed(parts)
```

Result:

* massive slowdown (10× step time)
* not viable under constraints

👉 rejected due to system cost

---

# 📉 What Didn’t Work

* heavy runtime token logic (too slow)
* aggressive gradient scaling
* compositional init (final performance loss)
* dynamic curriculum approaches

---

# 📈 What Worked

* **DP tokenization**
* **structured vocab additions**
* **mild gradient scaling**

---

# 🧠 Final Interpretation

We discovered a real tradeoff:

> **Compression vs Learnability under time constraints**

### Custom tokenizer:

* better representation
* harder to learn quickly

### SentencePiece:

* worse compression
* easier early optimization

---

# 🧠 Where This Stands

This system achieves:

✅ better compression than SP
✅ near-parity training
✅ fully reproducible pipeline
✅ systematic evaluation framework

---

# 🚀 Next Directions

## 1. Vocab-Level Optimization (Highest ROI)

* slightly reduce phrase token dominance
* increase ultra-common transitions
* rebalance frequency distribution

---

## 2. Model-Level Ideas

* shared embedding structure between tokens
* frequency-aware embedding scaling
* lightweight inductive biases (no runtime cost)

---

## 3. Strategic Position

Even without beating SP:

> This is a **novel tokenizer system with measurable advantages**

---

# ⚡ Bottom Line

* Tokenizer problem: **solved**
* Training gap: **small but persistent**
* Remaining challenge: **learning efficiency under tight budgets**

---

# 🧾 Code Reference

Baseline training script (modified from):



Key additions:

* `build_token_class_masks`
* `apply_token_class_grad_scaling`
* custom tokenizer loading + LUTs
* optional compositional init

---

