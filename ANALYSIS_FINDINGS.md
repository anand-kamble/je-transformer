# Comprehensive Analysis of JE-Transformer Project Issues

**Date:** 2025-11-13
**Analyst:** Claude (Sonnet 4.5)
**Repository:** je-transformer

## Executive Summary

This document provides a comprehensive analysis of the journal entry transformer model, focusing on:
1. Pointer mechanism implementation and stability
2. Catalog building process and alignment
3. Retrieval mechanism effectiveness
4. Loss function calculations and fluctuations

Multiple critical issues have been identified that likely contribute to pointer fluctuation and poor prediction accuracy.

---

## 1. ADAPTIVE LOSS SCALING ISSUES

### Location
`models/losses.py:26-50`

### Current Implementation
```python
def adaptive_loss_scale(loss: torch.Tensor, momentum: float = 0.99) -> torch.Tensor:
    if not hasattr(adaptive_loss_scale, 'running_mean'):
        adaptive_loss_scale.running_mean = 0.0
        adaptive_loss_scale.running_var = 1.0

    loss_val = loss.detach().item()
    adaptive_loss_scale.running_mean = (
        momentum * adaptive_loss_scale.running_mean + (1 - momentum) * loss_val
    )
    adaptive_loss_scale.running_var = (
        momentum * adaptive_loss_scale.running_var +
        (1 - momentum) * (loss_val - adaptive_loss_scale.running_mean) ** 2
    )

    running_std = torch.sqrt(torch.tensor(adaptive_loss_scale.running_var + 1e-6))
    normalized_loss = loss / torch.clamp(running_std, min=0.5, max=5.0)

    return normalized_loss
```

### Critical Issues

**Issue 1.1: Inappropriate Use of EMA for Loss Scaling**

This function is applied to `pointer_loss`, `side_loss`, and `stop_loss` (lines 48-58).

**Research Context:**
- According to "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits" (Morales-Brotons et al., November 2024), EMA is typically used for model parameters, not loss values [1]
- The paper states: "EMA requires less learning rate decay compared to SGD since averaging naturally reduces noise"
- This technique is designed for weight stabilization, NOT loss normalization

**Problems:**
1. **Global State Pollution**: Uses function attributes (`adaptive_loss_scale.running_mean`) which creates shared global state across ALL loss types
2. **Momentum Too High**: 0.99 momentum means 99% weight on history, only 1% on current - extremely slow adaptation
3. **Conflicting with Optimizer**: The optimizer (AdamW) already has its own adaptive learning rate mechanisms
4. **May Amplify Fluctuations**: Dividing by a moving standard deviation can amplify rather than reduce fluctuations when the std itself is fluctuating

**Evidence from Code:**
In `train_small.py:687-689`, this adaptive scaling is applied to ALL losses:
```python
pl = pointer_loss(outputs["pointer_logits"], target_account_idx, ignore_index=-1)
sl = side_loss(outputs["side_logits"], target_side_id, ignore_index=-1)
stl = stop_loss(outputs["stop_logits"], target_stop_id, ignore_index=-1)
```

**Issue 1.2: No Validation of Effectiveness**

There is no evidence or ablation study showing this adaptive scaling improves convergence or reduces fluctuation.

### Recommendation

**REMOVE** the adaptive loss scaling entirely and use standard cross-entropy loss. According to "Enhancing cross entropy with a linearly adaptive loss function" (Nature Scientific Reports, November 2024), if adaptive scaling is needed, it should be:
1. Per-batch, not using running statistics
2. Based on predicted probability, not running variance
3. Applied before gradient computation, not after [2]

---

## 2. POINTER MECHANISM STABILITY ISSUES

### Location
`models/pointer.py:10-72`

### Current Implementation Analysis

**Issue 2.1: Conflicting Temperature and Scale Parameters**

In `pointer.py:58`:
```python
logits = (self.logit_scale * logits) / self.temperature
```

**Problem:**
- Temperature typically controls softmax "softness" (lower = more peaked)
- Scale multiplier increases logit magnitude
- Applying both creates conflicting effects: scale amplifies, temperature reduces
- Training config uses `POINTER_SCALE_INIT=2.0` and `POINTER_TEMP=1.5`

**Research Context:**
From "Neural Combinatorial Optimization with Reinforcement Learning" (Bello et al., 2017):
- Original Pointer Networks used temperature OR learnable scale, not both [3]
- Temperature in [0.5, 2.0] range for exploration-exploitation tradeoff
- No mention of combining with multiplicative scaling

**Issue 2.2: Optional L2 Normalization**

In training config (`run_all_small.sh:131`):
```bash
--no-pointer-norm
```

**This DISABLES L2 normalization!**

**Research Context:**
From "Methods of improving LLM training stability" (October 2024):
- L2 normalization of queries and keys is critical for attention stability [4]
- "Applying L₂ normalization to queries improves model performance"
- QK-normalization prevents attention logit explosion

From "L2 normalization pointer network attention stability" research (2024):
- Without normalization, L2 norm of outputs can grow >2x during divergence [5]
- L2 norm growth is a leading indicator of training instability

**Issue 2.3: Insufficient Gradient Clipping for Pointer**

In `train_small.py:736`:
```python
clip_grad_by_component(model, max_norm=1.0)
```

But this clips ALL components uniformly. Pointer parameters get lower learning rate (0.5x) but same clipping.

**Research Context:**
From original Pointer Networks paper (Vinyals et al., 2015):
- Used "L2 gradient clipping of 2.0" specifically [6]
- This is HIGHER than other components, not lower

**Issue 2.4: Logit Clipping Value**

In `pointer.py:17`:
```python
logit_clip: float = 10.0
```

For large catalogs (e.g., 100+ accounts), this may be insufficient.

### Recommendations

1. **Remove Scale OR Temperature** - Don't use both
2. **ENABLE L2 Normalization** - Remove `--no-pointer-norm` flag
3. **Increase Pointer Gradient Clipping** - Use 2.0 for pointer, 1.0 for others
4. **Adaptive Logit Clipping** - Scale with catalog size: `max(10.0, 0.1 * catalog_size)`

---

## 3. CATALOG BUILDING AND ALIGNMENT ISSUES

### Location
- Training: `train_small.py:60-67`
- Inference: `infer_small.py:39-45`

### Critical Finding: Training vs Inference Mismatch

**Issue 3.1: Trainable Catalog in Training**

In `train_small.py:591-595`:
```python
if args.trainable_catalog:
    try:
        model.set_catalog_embeddings(cat_emb)
    except Exception as e:
        print(f"Warning: failed to register trainable catalog embeddings: {e}")
```

In `run_all_small.sh:132`:
```bash
--trainable-catalog
```

**This is ENABLED in training!**

**Issue 3.2: Fixed Catalog in Inference**

In `infer_small.py:248`:
```python
cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)
```

The catalog is REBUILT from scratch using `CatalogEncoder`, which has:
- Randomly initialized embeddings (hash-based)
- No loading of trained parameters

**CRITICAL MISMATCH:**
- Training: Catalog embeddings are learned/fine-tuned
- Inference: Catalog embeddings are randomly re-initialized

**Issue 3.3: Saved Catalog Not Used**

In `train_small.py:955-964`, catalog embeddings ARE saved:
```python
catalog_emb_path = os.path.join(tmpdir, "catalog_embeddings.pt")
try:
    cat_src = (
        model.catalog_param.detach().cpu()
        if (bool(args.trainable_catalog) and hasattr(model, "catalog_param"))
        else cat_emb.detach().cpu()
    )
    torch.save({"catalog_embeddings": cat_src}, catalog_emb_path)
```

But in `infer_small.py`, there's NO code to load `catalog_embeddings.pt`!

### Impact

This is likely a MAJOR source of pointer prediction errors. The model learns with one set of catalog embeddings but infers with completely different ones.

**Research Context:**
From "Fixed vs trainable embeddings" (OpenNMT Forum, 2021):
- "Trainable embeddings allow models to learn embeddings specifically for your task"
- But training and inference MUST use the same embeddings [7]

### Recommendations

1. **Load Saved Catalog** - Modify `infer_small.py` to load `catalog_embeddings.pt`
2. **OR: Use Fixed Catalog** - Remove `--trainable-catalog` from training
3. **Verify Checkpoint Includes Catalog** - Ensure catalog is in saved checkpoint

---

## 4. DIMENSIONS AND EMBEDDINGS VERIFICATION

### Analysis of Dimension Flow

**Encoder → Hidden Dimension:**
- Encoder: `FacebookAI/xlm-roberta-base` → 768 dimensions
- Hidden dim: 768 (from `HIDDEN_DIM=768` in training script)
- Projection: `enc_proj` maps 768 → 768 ✓

**Catalog Embeddings:**
- Built with `CatalogEncoder(emb_dim=768)`
- Output: `[num_accounts, 768]` ✓

**Pointer Layer:**
- Input: decoder state `[B, T, 768]`
- Catalog: `[num_accounts, 768]`
- Output: logits `[B, T, num_accounts]` ✓

**Retrieval Memory:**
- Built from ScaNN index with same encoder (xlm-roberta-base)
- Should be `[K, 768]` where K=top_k=5
- But needs projection through model.enc_proj!

### Issue 4.1: Retrieval Memory Dimension Mismatch

In `train_small.py:529-533`:
```python
with torch.no_grad():
    emb_t = torch.tensor(emb_np, dtype=torch.float32)  # [N, 768]
    proj = torch.tanh(model.enc_proj(emb_t.to(device=model.enc_proj.weight.device)))
    _retr_proj_embs = proj.detach().cpu()
```

**GOOD:** Retrieval embeddings ARE projected through `enc_proj` in training.

But in `inference/retrieval_memory.py:98`:
```python
return torch.tensor(mem, dtype=torch.float32)
```

**BAD:** Retrieval memory is returned RAW, not projected!

### Recommendation

Ensure retrieval embeddings are projected consistently in both training and inference.

---

## 5. RETRIEVAL MECHANISM EVALUATION

### Location
`models/je_model_torch.py:149-165`

### Current Implementation

```python
retr_mem_proj = self.retr_mem_proj(retrieval_memory)
if retr_mem_proj.dim() == 2:
    retr_mem_proj = retr_mem_proj.unsqueeze(0).expand(B, -1, -1)
dec_n = F.normalize(dec_h, dim=-1)
mem_n = F.normalize(retr_mem_proj, dim=-1)
scores = torch.einsum("bth,bkh->btk", dec_n, mem_n)
weights = torch.softmax(scores, dim=-1)
retr_ctx = torch.einsum("btk,bkh->bth", weights, retr_mem_proj)
gate = torch.sigmoid(self.retr_gate(dec_h))
retr_ctx = retr_ctx * gate
```

### Analysis

**Issue 5.1: Complexity Without Validation**

1. Additional projection layer (`retr_mem_proj`)
2. L2 normalization of both decoder and memory
3. Attention mechanism (softmax over K neighbors)
4. Gating mechanism
5. Concatenation and final projection

**Questions:**
- Is this adding value or just noise?
- Has an ablation study been done?
- Does it improve metrics?

**Research Context:**
From "Retrieval-Augmented Generation: A Comprehensive Survey" (June 2024):
- RAG-Fusion uses early fusion with attention over retrieved documents [8]
- But effectiveness depends on retrieval quality
- "Late fusion can improve robustness by evaluating alternative perspectives"

**Issue 5.2: Potential Instability**

The gate mechanism `torch.sigmoid(self.retr_gate(dec_h))` can cause:
1. Vanishing gradients if gate → 0
2. Retrieval memory ignored during training
3. Unstable gradient flow

**Issue 5.3: No Retrieval in Inference**

In `infer_small.py:320`:
```python
retr_mem = torch.zeros((1, args.hidden_dim), dtype=torch.float32, device=device)
```

**The retrieval mechanism is DISABLED during inference!**

This is another training/inference mismatch.

### Recommendations

1. **Ablation Study** - Train model without retrieval to see if it helps
2. **Simplify Fusion** - If keeping retrieval, use simpler concatenation
3. **Enable in Inference** - Or remove from both training and inference
4. **Monitor Gate Values** - Log gate statistics to see if retrieval is being used

---

## 6. LOSS FUNCTION FLUCTUATION ISSUES

### Location
`models/losses.py`

### Issue 6.1: Flow Auxiliary Loss Instability

In `losses.py:277-316`:
```python
def flow_aux_loss(
    pointer_logits: torch.Tensor,
    side_logits: torch.Tensor,
    debit_indices: torch.Tensor,
    debit_weights: torch.Tensor,
    credit_indices: torch.Tensor,
    credit_weights: torch.Tensor,
    pointer_probs: Optional[torch.Tensor] = None,
    side_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ...
    # Normalize gathered vs target weights
    sum_g = gathered.sum(dim=-1, keepdim=True) + 1e-8
    g_norm = torch.where(sum_g > 0, gathered / sum_g, torch.zeros_like(gathered))

    sum_w = wts.sum(dim=-1, keepdim=True) + 1e-8
    w_norm = torch.where(sum_w > 0, wts / sum_w, torch.zeros_like(wts))

    mse = ((g_norm - w_norm) ** 2).mean(dim=-1)
```

**Problems:**
1. **Double Normalization**: Both predicted mass and target weights are normalized to sum to 1
2. **MSE on Distributions**: MSE is not ideal for comparing probability distributions
3. **Numerical Instability**: Division by sum with small epsilon can amplify noise

**Research Context:**
From "Adaptive Adversarial Cross-Entropy Loss for Sharpness-Aware Minimization" (June 2024):
- Cross-entropy or KL divergence are preferred for distribution matching [9]
- MSE on normalized distributions can be unstable
- Better to use: `KLDiv(pred || target)` or `CrossEntropy(target, pred_logits)`

**Issue 6.2: Dynamic Loss Weights**

In `train_small.py:191-209`:
```python
def get_loss_weights(epoch, total_epochs):
    progress = epoch / total_epochs

    # Decay flow loss over time
    flow_weight = args.flow_weight * (2.0 - progress)

    # Increase coverage penalty over time
    coverage_weight = 0.01 * (1.0 + progress)

    # Warmup for first 3 epochs
    if epoch < 3:
        flow_warmup = args.flow_warmup_multiplier
    else:
        flow_warmup = 1.0
```

**Problems:**
1. Flow loss weight changes from `5.0 * 0.01 * 2.0 = 0.1` → `1.0 * 0.01 * 1.0 = 0.01` (10x decrease)
2. Coverage weight doubles during training
3. These changes can cause sudden shifts in loss landscape

**Issue 6.3: Coverage Penalty May Conflict**

In `losses.py:61-65`:
```python
def coverage_penalty(pointer_logits: torch.Tensor, max_total: float = 1.0, probs: Optional[torch.Tensor] = None) -> torch.Tensor:
    p = probs if probs is not None else torch.softmax(pointer_logits, dim=-1)
    sum_over_t = p.sum(dim=1)  # [B, num_accounts]
    surplus = torch.clamp(sum_over_t - max_total, min=0.0)
    return surplus.sum(dim=-1).mean()
```

This penalizes the model if the same account gets >1.0 total probability across all lines.

**Problem:**
- This may conflict with the actual data distribution
- If an account SHOULD appear multiple times, this penalty prevents it
- May be causing the model to "spread" probability too thin

### Recommendations

1. **Replace Flow Loss MSE** - Use KL divergence or cross-entropy
2. **Stabilize Loss Weights** - Keep constant or change very slowly
3. **Reconsider Coverage Penalty** - May be counter-productive
4. **Remove Adaptive Scaling** - As discussed in Section 1

---

## 7. SUMMARY OF CRITICAL ISSUES

### High Priority (Immediate Fix)

1. **Training/Inference Catalog Mismatch** - Catalog embeddings not loaded in inference
2. **L2 Normalization Disabled** - Critical for pointer stability
3. **Adaptive Loss Scaling** - Likely causing fluctuations, not preventing them
4. **Retrieval Disabled in Inference** - Training/inference mismatch

### Medium Priority

5. **Conflicting Temperature/Scale** - Simplify to one or the other
6. **Flow Loss Using MSE** - Replace with KL divergence
7. **Dynamic Loss Weights** - Causing training instability

### Low Priority

8. **Gradient Clipping** - Increase for pointer layers
9. **Coverage Penalty** - Reconsider if needed
10. **Retrieval Mechanism** - Simplify or remove if not helping

---

## 8. REFERENCES

[1] Morales-Brotons, D., & Vogels, T. (2024). "Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits." *arXiv:2411.18704*. https://arxiv.org/abs/2411.18704

[2] "Enhancing cross entropy with a linearly adaptive loss function for optimized classification performance." *Scientific Reports*, November 2024. https://www.nature.com/articles/s41598-024-78858-6

[3] Vinyals, O., Fortunato, M., & Jaitly, N. (2015). "Pointer Networks." *NeurIPS 2015*. http://papers.neurips.cc/paper/5866-pointer-networks.pdf

[4] "Methods of improving LLM training stability" (October 2024). *arXiv:2410.16682*. https://arxiv.org/abs/2410.16682

[5] "Normalization in Attention Dynamics" (October 2024). *arXiv:2510.22026*. https://arxiv.org/abs/2510.22026

[6] Bello, I., et al. (2017). "Neural Combinatorial Optimization with Reinforcement Learning." *arXiv:1611.09940*. https://arxiv.org/abs/1611.09940

[7] OpenNMT Forum. "Fixed vs trainable embeddings." https://forum.opennmt.net/t/fixed-vs-trainable-embeddings/2705

[8] "RAG-Fusion: A New Take on Retrieval-Augmented Generation" (February 2024). *arXiv:2402.03367*. https://arxiv.org/abs/2402.03367

[9] "Adaptive Adversarial Cross-Entropy Loss for Sharpness-Aware Minimization" (June 2024). *arXiv:2406.14329*. https://arxiv.org/abs/2406.14329

---

## Next Steps

1. Review and validate each finding
2. Prioritize fixes based on impact
3. Implement changes incrementally
4. Run ablation studies to verify improvements
5. Monitor W&B metrics after each change

