# KL Divergence vs Cross-Entropy in Knowledge Distillation

## 1. The two quantities

### Cross-Entropy H(p, q)

Given a **teacher** distribution $p$ and a **student** distribution $q$ over the same vocabulary:

$$H(p,\, q) = -\sum_{i} p_i \log q_i$$

This asks: *how many bits does it cost to encode samples drawn from p using a code optimised for q?*

### KL Divergence D_KL(p ∥ q)

$$D_{KL}(p \,\|\, q) = \sum_{i} p_i \log \frac{p_i}{q_i} = \sum_{i} p_i \log p_i - \sum_{i} p_i \log q_i$$

Rewriting in terms of the two simpler quantities:

$$\boxed{D_{KL}(p \,\|\, q) = H(p,\, q) - H(p)}$$

where $H(p) = -\sum_i p_i \log p_i$ is the **self-entropy of the teacher** (a constant once the teacher is frozen).

---

## 2. Why they are equivalent for optimisation

When you train the student, the only free variables are the **student parameters** $\theta$.  
The teacher is **frozen**, so $p$ is fixed, and therefore $H(p)$ is a constant — its gradient w.r.t. $\theta$ is **zero**.

$$\nabla_\theta\, D_{KL}(p \,\|\, q_\theta) = \nabla_\theta\, H(p,\, q_\theta) - \underbrace{\nabla_\theta\, H(p)}_{= 0}$$

$$\Rightarrow \quad \nabla_\theta\, D_{KL}(p \,\|\, q_\theta) = \nabla_\theta\, H(p,\, q_\theta)$$

**Minimising cross-entropy is identical to minimising KL divergence**, because the only difference between them is a constant offset.

---

## 3. What PyTorch's F.kl_div actually computes

```python
F.kl_div(input=log_q, target=p, reduction='batchmean')
```

This computes:

$$\sum_i p_i \left(\log p_i - \log q_i\right) = D_{KL}(p \,\|\, q)$$

Notice it **includes** the teacher entropy term $\sum_i p_i \log p_i$.

If you use cross-entropy instead:

```python
-(p * log_q).sum(dim=-1)   # = H(p, q)
```

You are computing something **smaller by exactly H(p)**. The numerical loss value will be different — but since $H(p)$ contributes no gradient, **the parameter updates are byte-for-byte identical**.

---

## 4. Why this project uses cross-entropy (and that is the right call)

### 4.1  The top-K truncation changes the teacher entropy

The full teacher vocabulary is $V_t \approx 152{,}000$. Storing and loading full distributions is prohibitive, so only the **top-K = 10** teacher logprobs are materialised on disk. At training time these are renormalised into a truncated distribution $\tilde{p}^t$:

$$\tilde p^t_k = \frac{\exp(\ell_k / T)}{\sum_{k'} \exp(\ell_{k'} / T)}, \quad k = 1, \ldots, K$$

This is a **different** distribution from the full teacher $p^t$. Its self-entropy $H(\tilde{p}^t)$ is also a constant (frozen teacher), so the CE-vs-KL equivalence still holds. But if you tried to compute $D_{KL}$ against the full vocabulary you would need the full teacher probabilities — which you don't have.

Using cross-entropy sidesteps this completely: you only need $p_i$ for the $K$ positions where the teacher gave you mass.

### 4.2  The kd_trainer.py implementation

```python
# kd_trainer.py:109-122
student_logp = F.log_softmax(student_slice_f32 / temperature, dim=-1)  # [B, N, V]
student_logp_k = student_logp.gather(-1, topk_ids_dev)                 # [B, N, K]

teacher_p = torch.softmax(teacher_logits_k, dim=-1)                    # [B, N, K]  (renorm over top-K)

soft_per_token = -(teacher_p * student_logp_k).sum(dim=-1)             # [B, N]  = H(p̃^t, q^s_k)
loss_soft = (temperature ** 2) * masked_mean(soft_per_token, mask)
```

This is exactly $H(\tilde{p}^t, \tilde{q}^s)$ restricted to the teacher's top-K ids, scaled by $T^2$.

Compare with what `F.kl_div` would give (if you naively used it):

```python
# If you used F.kl_div instead:
soft_per_token = F.kl_div(student_logp_k, teacher_p, reduction='none').sum(-1)
# = -(teacher_p * student_logp_k).sum(-1)  +  (teacher_p * teacher_p.log()).sum(-1)
#   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#         cross-entropy term (same gradient)      teacher entropy (constant, zero gradient)
```

You would carry an extra constant per position. Gradients are identical; the only effect is a slightly higher loss number in your logs.

### 4.3  Why other implementations sometimes do use F.kl_div

Some frameworks (e.g. DistilBERT, some TRL trainers) use `F.kl_div` with `log_target=True` to avoid computing `log(p)` from scratch. They pass the teacher's stored logprobs directly. In that case KL div is a convenience, not a mathematical requirement.

Other implementations use KL div because they operate on the **full vocabulary** and rely on PyTorch's fused kernel being faster than manually gathering and multiplying — it amortises the loop over V tokens on the GPU.

Neither of these advantages applies here:
- You **don't** have full teacher logprobs (only top-K).
- Gathering K ≤ 10 positions is trivially cheap compared to a V=152K softmax.

Cross-entropy over the top-K renormalised distribution is the **correct, minimal, and efficient choice** for this setup.

---

## 5. The T² factor — why it is there

When you apply temperature T to the logits, the softmax Jacobian scales by $1/T$:

$$\frac{\partial}{\partial z_j} \operatorname{softmax}(z/T)_i = \frac{1}{T}\left(\delta_{ij} - q_i q_j\right)$$

This means the gradient of the soft loss w.r.t. the raw student logits shrinks by $1/T$ relative to the hard loss gradient. Without compensation, increasing T would make the soft term progressively weaker — so you could not set alpha and T independently.

Multiplying $\mathcal{L}_{\text{soft}}$ by $T^2$ restores the natural scale:

$$\nabla_z \left[T^2 \cdot H(\tilde{p}^t_T, \tilde{q}^s_T)\right] \approx T^2 \cdot \frac{1}{T} \cdot (q_T - p_T) = T \cdot (q_T - p_T)$$

At $T = 1.0$ (our run), this is just $(q - p)$ — the standard distillation gradient — so the factor does not change anything numerically for our specific configuration, but it is the correct formulation for the general case and keeps the code correct if T is ever changed.

---

## 6. Summary table

| Property | Cross-Entropy H(p, q) | KL Divergence D_KL(p ∥ q) |
|---|---|---|
| Formula | $-\sum p_i \log q_i$ | $\sum p_i \log(p_i / q_i)$ |
| Gradient w.r.t. student | $-(p - q) / q$ (softmax jacobian) | **identical** |
| Numerical value | smaller | larger by $H(p)$ |
| Needs teacher self-entropy | No | Yes |
| Works with top-K truncation | Yes (naturally) | Needs care |
| PyTorch API | `-(p * log_q).sum()` | `F.kl_div(log_q, p)` |
| Used in this project | **Yes** | No (equivalent, not used) |
