# KD Training — Challenges & Fixes

This note documents the knowledge-distillation loss used in `kd_trainer.py`, the
two bugs we hit when scaling it up, and why even a 96 GB GPU runs out of memory
at surprisingly small batch sizes.

---

## 1. The distillation loss

The student is trained against a convex combination of a **hard** cross-entropy
(CE) loss on the gold next-token labels and a **soft** KL-style loss against the
teacher's output distribution, restricted to the assistant span of each
conversation:

$$
\mathcal{L} \;=\; (1-\alpha)\,\mathcal{L}_{\text{hard}} \;+\; \alpha\,T^{2}\,\mathcal{L}_{\text{soft}}
$$

where $\alpha\in[0,1]$ mixes the two and $T>0$ is the softmax temperature.

### 1.1 Hard loss

Let $z^{s}_{b,n}\in\mathbb{R}^{V_s}$ be the student's logits at the position that
predicts assistant token $n$ in sample $b$, and $y_{b,n}$ the gold token id.
Let $m_{b,n}\in\{0,1\}$ be the span mask (1 on the assistant span, 0 elsewhere).
Then

$$
\mathcal{L}_{\text{hard}}
\;=\;
\frac{\sum_{b,n} m_{b,n}\;\bigl[-\log \operatorname{softmax}(z^{s}_{b,n})_{y_{b,n}}\bigr]}
     {\sum_{b,n} m_{b,n}}.
$$

This is standard next-token CE, but averaged only over assistant positions —
prompt and padding tokens do not contribute. In code this is exactly
`F.cross_entropy(...).reshape(B, N)` masked by `span_mask`.

### 1.2 Soft loss (top-K KL)

The teacher's full distribution $p^{t}_{b,n}\in\Delta^{V_t-1}$ is expensive to
store, so during `build_kd_dataset` we keep only the **top-K teacher logprobs**
$\{\ell_{b,n,k}\}_{k=1}^{K}$ and the corresponding token ids
$\{i_{b,n,k}\}_{k=1}^{K}$. At train time we renormalize these $K$ logprobs into a
proper distribution over the $K$ teacher-chosen ids:

$$
\tilde p^{t}_{b,n,k}
\;=\;
\frac{\exp(\ell_{b,n,k} / T)}{\sum_{k'=1}^{K}\exp(\ell_{b,n,k'} / T)}
\quad\text{with invalid slots zero-masked.}
$$

The student's matching distribution at the same $K$ token ids comes from a
softmax over its **full** vocabulary followed by a gather:

$$
q^{s}_{b,n,k}
\;=\;
\operatorname{softmax}\!\bigl(z^{s}_{b,n}/T\bigr)_{\,i_{b,n,k}}.
$$

The soft loss is the cross-entropy between $\tilde p^{t}$ and $q^{s}$:

$$
\mathcal{L}_{\text{soft}}
\;=\;
\frac{\sum_{b,n} m_{b,n}\;\bigl[-\sum_{k} \tilde p^{t}_{b,n,k}\,\log q^{s}_{b,n,k}\bigr]}
     {\sum_{b,n} m_{b,n}}.
$$

This is exactly the top-K restriction of the standard Hinton distillation term
$T^{2}\,\mathrm{KL}(p^{t}_T\,\Vert\,q^{s}_T)$, up to the (constant-in-student)
teacher entropy term. The $T^{2}$ prefactor compensates for the fact that the
gradient of a temperature-$T$ softmax is scaled by $1/T$, so that the relative
strength of the hard and soft signals does not change when $T$ is tuned.

### 1.3 Why top-K is a reasonable approximation

On Qwen2.5 with $V\approx 152{,}000$ and a decently-trained teacher, the top-10
probability mass at each position is typically >0.95. The tail distribution
contributes almost nothing to $\mathrm{KL}(p^{t}\,\Vert\,q^{s})$ in absolute
value, and renormalizing over the top-K preserves the relative ordering and
sharpness of the teacher's beliefs where it matters.

---

## 2. The vocabulary-mismatch bug

### 2.1 Symptom

At the first training step, a CUDA device-side assert fires inside
`F.cross_entropy`:

```
ScatterGatherKernel.cu:163: Assertion `idx_dim >= 0 && idx_dim < index_size' failed.
```

Because CUDA kernels launch asynchronously, the Python traceback points at
`cross_entropy` but the actual offending kernel is the earlier gather
`student_logp.gather(-1, topk_ids)` on the vocab dimension.

### 2.2 Root cause

Qwen2.5 checkpoints do **not** all share the same `vocab_size`. The lm_head is
padded up to a hardware-friendly multiple for each model size:

| Model              | `vocab_size` |
|--------------------|--------------|
| Qwen2.5-0.5B (student) | **151 936** |
| Qwen2.5-14B (teacher)  | **152 064** |

The extra 128 slots on the teacher are reserved/padding tokens that the
tokenizer will never emit for normal text, but during teacher inference the
softmax *is* computed over all 152 064 logits. Occasionally a small amount of
probability mass lands in slot $\geq 151{,}936$, and that id ends up in the
stored top-K.

At train time we then do

```python
student_logp.gather(-1, topk_ids)   # [B, N, V_s=151936]  indexed by ids up to 152063
```

which is an out-of-bounds read on the student's vocab axis → device-side
assert. The same failure mode exists for `gold_ids` if any gold token happens
to live in the padded range.

### 2.3 Fix

In `kd_trainer.compute_loss` we now clamp and mask **before** any gather:

```python
V = logits.shape[-1]                       # student vocab

oob_topk = topk_ids >= V
topk_valid = topk_valid & ~oob_topk        # drop OOB slots from the soft target
topk_ids   = topk_ids.masked_fill(oob_topk, 0)  # safe id; already masked out

oob_gold = gold_ids >= V
mask     = span_mask & ~oob_gold           # drop OOB golds from the hard CE
gold_ids = gold_ids.masked_fill(oob_gold, 0)
```

Masked-out top-K slots receive $-\infty$ logprob before the softmax, so
$\tilde p^{t}_{b,n,k}=0$ for them and they contribute nothing to
$\mathcal{L}_{\text{soft}}$. Masked-out gold positions drop out of the
normalizer of $\mathcal{L}_{\text{hard}}$, so they contribute nothing either.
The loss remains an unbiased estimate of the intended objective on the
in-vocabulary subset.

A cleaner long-term fix is to apply the same filter offline in
`build_kd_dataset.py`, so the artifact never ships ids the student cannot
represent.

### 2.4 Bonus bug: silent sequence truncation

Unsloth silently truncates `input_ids` longer than `max_seq_length` to fit, but
the KD metadata (`assistant_start`, `gold_ids`, `topk_ids`) is materialized at
dataset-build time against the *full* tokenized sequence. A truncated row
keeps its original `assistant_start`, which now points **past** the end of the
truncated logits tensor; the gather still works numerically (we clamp `pos` to
`T-1`) but reads the wrong positions. The training signal on that row is
noise.

`train_kd.py` now filters these rows at load time:

```python
train_dataset = train_dataset.filter(
    lambda ex: len(ex["input_ids"]) <= max_seq_length,
    num_proc=1,
)
```

with a printed report of how many rows were dropped.

---

## 3. Why KD is so VRAM-hungry even on a 96 GB GPU

The headline surprise: a *0.5 B* student needs **`per_device_batch_size=8`**
with gradient-accumulation 8 on a 96 GB Blackwell GPU. Vanilla SFT of the same
model happily runs at batch 32+ on much smaller cards. The gap is not the
model — it is the **logits tensor** the distillation loss forces you to
materialize.

### 3.1 The dominant tensor

For a forward pass with batch $B$, sequence length $T$, and student vocab
$V_s$, the lm_head output is

$$
z^{s} \in \mathbb{R}^{B \times T \times V_s}.
$$

Concretely, with $B=8$, $T=2048$, $V_s=151{,}936$, bf16 (2 bytes/element):

$$
8 \times 2048 \times 151{,}936 \times 2\ \text{B}
\;\approx\; 4.64\ \text{GB per tensor}.
$$

That single tensor is already larger than the entire 0.5 B model weights
(~1 GB in bf16). You pay for it at least **three** times over:

1. **Forward activation.** $z^{s}$ itself — kept live until backward completes.
2. **Backward grad buffer.** $\partial \mathcal{L}/\partial z^{s}$ has the same
   shape — another 4.64 GB in bf16.
3. **Loss intermediates.** `F.cross_entropy` and `F.log_softmax` internally
   upcast to fp32 and allocate a working copy of $z^{s}$ of shape $[B,T,V_s]$
   in **fp32** (4 bytes/element):

$$
8 \times 2048 \times 151{,}936 \times 4\ \text{B}\;\approx\; 9.28\ \text{GB}.
$$

Plus gradient-checkpointing recomputation keeps additional transient copies
live during backward. Adding adapter params, optimizer state (adamw_8bit),
and a sizeable RoPE cache pushes total occupancy well above 60 GB at
$B\!=\!16$.

### 3.2 Scaling law

The logits-driven term scales as $\mathcal{O}(B\,T\,V_s)$ per copy. Doubling
any of the three factors doubles all of it. That is why:

* dropping `max_seq_length` from 6048 → 2048 cut logits memory by $\sim 3\times$,
* dropping $B$ from 16 → 8 cut it by $2\times$,
* and the product of the two is what finally fit.

Compare a vanilla SFT step: HuggingFace's fused CE (or Liger / Unsloth's fused
kernels) never materialize a full $[B,T,V]$ fp32 softmax — they compute CE
directly from the bf16 logits with a streaming reduction. That saves the
~9 GB fp32 copy. Our KD loss **does** need an intermediate $\log\!\operatorname{softmax}$
over the full vocab (to then gather at the teacher's top-K ids), which
reinstates the fp32 blowup.

### 3.3 What would reduce it

We accept the cost for now, but the natural optimizations are:

1. **LogSumExp trick.** Compute only $\mathrm{lse}(z^{s}_{b,n})\in\mathbb{R}$
   and gather the raw logits at $\{y_{b,n}\} \cup \{i_{b,n,k}\}_{k=1}^{K}$.
   Then

   $$\log q^{s}_{b,n,k} \;=\; z^{s}_{b,n,\,i_{b,n,k}} / T \;-\; \mathrm{lse}(z^{s}_{b,n}/T).$$

   This replaces the $[B,T,V_s]$ fp32 tensor with a $[B,T,K{+}1]$ fp32 tensor
   and a $[B,T]$ scalar — roughly a $V_s/K \approx 15{,}000\times$ reduction on
   the dominant intermediate.
2. **Compute only over masked positions.** Flatten along the assistant span
   before any fp32 work — typically $M=\sum m_{b,n}$ is a small fraction of
   $B\cdot T$.
3. **Fused KD kernels.** Liger-Kernel ships a fused distillation loss that
   never materializes the full logits in fp32; dropping it in replaces steps
   (1) and (2) with a single call.

With these in place, the same student would comfortably train at
$B\!\geq\!32$ on the same hardware; the current settings are conservative but
correct.
