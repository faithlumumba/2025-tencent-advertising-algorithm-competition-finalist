# Method Overview

We tackle next-item prediction using a sampled-softmax objective, drawing negative examples from both in-batch positives and a global random pool. Training is further enhanced through in-batch hard top-k negative mining and a curriculum schedule that adapts difficulty over time.

## Loss Function

Given query **Q**, positive **K₊**, negatives **N**, and temperature **τ**,
(similarity is dot-product after L2-normalization → cosine):

$$
\mathcal{L} = -\log
\frac{\exp(s_+/\tau - \log q(i_+))}{
\exp(s_+/\tau - \log q(i_+)) + \sum_{j\in N}\exp(s_j/\tau - \log q(i_j))}
,\quad s_\bullet=\langle Q,K_\bullet\rangle
$$

All logits subtract **log q(·)** for **importance sampling correction**, yielding an **unbiased estimator of the full-softmax gradient**.

---

## Preparation

### 1. Frequency Statistics (`log q`)

Use `polars` to count item frequencies and build 

$$
q(i)\propto \mathrm{freq}(i)^{0.75}
$$

### 2. Global Negative Pool

Keep items whose **last appearance > May 29** (i.e., still active).
Exclude items already shown in the current sequence when sampling.

---

## Mixed Negative Sampling

### A. In-batch Sampling

* **Source:** Treat the **positive samples** from other examples within the same batch as negatives for the current sample. Compared with the softmax in standard classification, popular items will be sampled repeatedly, resulting in a **non-uniform sampling distribution**. The probability of an item being sampled is proportional to its occurrence frequency in the entire log. Therefore, to recover the behavior of standard softmax, we need to subtract **log q** from the logits for debiasing according to the principle of importance sampling.

* **Importance Correction:** Subtract the corresponding **log q** from the logits of these negative samples.

* **False-Negative Filtering:**

  * Do not treat items from the **same sequence** as negatives.
  * If an item’s similarity to the current positive exceeds 0.99, treat it as a potential interest item and filter it out.
  * Because of this filtering, the computation of **q** is weakened to the item frequency raised to the power of **0.75**.
  * Mathematically equivalent to popularity-based sampling, but more efficient in implementation.

* **Apply the same − log q to positive samples as well:**

  * After correcting the positive samples, the loss becomes exactly equivalent to full softmax.
  * The purpose of log q correction is to suppress popular items. Applying the correction to positive samples is crucial because it increases the weight of **cold (new) items**. In the current dataset, more than half of all items appear fewer than five times. Thus, we need to pay special attention to predicting cold/new items, which is why the correction is also applied to positive samples.

---

### B. Global Random Sampling

* **Source:** Randomly sample negatives from the active-item pool, excluding items that appear in the current sequence.

---

## Hard Negative Mining

In the mid and late training stages, most negatives are already easy (low-score), and retaining all of them **dilutes gradient**.
Thus, we keep only the **top-k highest-scoring negatives**.

---

## Curriculum Learning

As training progresses, gradually **reduce** `hard_topk` and decrease the proportion of global negatives, making the sampled negatives progressively **harder**.

---

## Feature Modeling

### 1. Item Tower (End-to-End)

Item ID and sparse features → embeddings → `itemdnn` (MLP).
No handcrafted features.

### 2. User as BOS + FiLM Conditioning

User features = **first token (mask = 2)**, `pos_emb = 0`.
Aggregated user vector → `(γ, β)` → FiLM modulation: $x_t' = x_t \cdot (1+\gamma) + \beta$

### 3. Time & Action

* **Cyclic time**: sin/cos(hour, weekday)
* **Relative Attention Bias (RAB)**: log-bucketed time difference as additive bias
* **Next action type**: added to the input via embeddings for conditional generation.

---

## Model Architecture: HSTU

**Single Layer Steps**

1. Linear → **U/V/Q/K** (multi-head split)
2. Compute logits:
   $\frac{QK^\top}{\sqrt{d_k}} + \mathrm{RAB}$
   → apply **SiLU** instead of softmax → weights A
4. Aggregate Y = A @ V
5. Normalize Y and **gate** with U
6. Linear projection + residual connection

---

## Inference

* Read `user_action_type.json` to set each user’s next-action condition
* Exclude items already seen in user history during sampling

---





