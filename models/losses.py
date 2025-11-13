from __future__ import annotations

from typing import Optional
from collections import deque

import torch
import torch.nn.functional as F


def masked_sparse_ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """
    Compute sparse CE over last dimension of logits with mask where targets == ignore_index.
    Shapes:
      - logits: [B, T, V]
      - targets: [B, T]
    Returns mean loss over non-ignored positions.
    """
    if logits.dim() != 3:
        raise ValueError("logits must be [B, T, V]")
    if targets.dim() != 2:
        raise ValueError("targets must be [B, T]")
    B, T, V = logits.shape
    logits_2d = logits.reshape(B * T, V)
    targets_1d = targets.reshape(B * T)
    mask = (targets_1d != ignore_index).to(logits_2d.dtype)
    safe_targets = torch.where(mask > 0.5, targets_1d, torch.zeros_like(targets_1d))
    # Per-position CE
    ce = F.cross_entropy(logits_2d, safe_targets, reduction="none")  # [B*T]
    ce = ce * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return ce.sum() / denom

def adaptive_loss_scale(loss: torch.Tensor, momentum: float = 0.99) -> torch.Tensor:
    """
    Normalize loss by its running standard deviation to reduce variance.
    Based on research showing variance regularization improves stability.
    """
    if not hasattr(adaptive_loss_scale, 'running_mean'):
        adaptive_loss_scale.running_mean = 0.0
        adaptive_loss_scale.running_var = 1.0
    
    # Update running statistics
    loss_val = loss.detach().item()
    adaptive_loss_scale.running_mean = (
        momentum * adaptive_loss_scale.running_mean + (1 - momentum) * loss_val
    )
    adaptive_loss_scale.running_var = (
        momentum * adaptive_loss_scale.running_var + 
        (1 - momentum) * (loss_val - adaptive_loss_scale.running_mean) ** 2
    )
    
    # Normalize by running std (with floor to prevent division by zero)
    running_std = torch.sqrt(torch.tensor(adaptive_loss_scale.running_var + 1e-6))
    normalized_loss = loss / torch.clamp(running_std, min=0.5, max=5.0)
    
    return normalized_loss


def pointer_loss(pointer_logits: torch.Tensor, target_account_idx: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """
    Pointer loss with variance normalization to reduce oscillations.
    """
    # Use the existing masked_sparse_ce function
    loss = masked_sparse_ce(pointer_logits, target_account_idx, ignore_index=ignore_index)
    
    # Add batch-wise variance normalization to stabilize training
    # This reduces the high variance visible in training graphs
    return adaptive_loss_scale(loss)


def side_loss(side_logits: torch.Tensor, target_side_id: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    return adaptive_loss_scale(masked_sparse_ce(side_logits, target_side_id, ignore_index=ignore_index))


def stop_loss(stop_logits: torch.Tensor, target_stop_id: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    return adaptive_loss_scale(masked_sparse_ce(stop_logits, target_stop_id, ignore_index=ignore_index))


def coverage_penalty(pointer_logits: torch.Tensor, max_total: float = 1.0, probs: Optional[torch.Tensor] = None) -> torch.Tensor:
    p = probs if probs is not None else torch.softmax(pointer_logits, dim=-1)
    sum_over_t = p.sum(dim=1)
    surplus = torch.clamp(sum_over_t - max_total, min=0.0)
    return surplus.sum(dim=-1).mean()


class SetF1Metric:
    """
    Approximate set-level F1 over (account, side) pairs ignoring order.
    Uses greedy argmax decoding per step until STOP=1 in targets (or uses T steps if no explicit stop labels).
    Not a torch.nn.Module; keeps running totals in torch tensors.
    """

    def __init__(self, stop_id: int = 1) -> None:
        self.stop_id = stop_id
        device = torch.device("cpu")
        self.tp = torch.zeros((), dtype=torch.float32, device=device)
        self.fp = torch.zeros((), dtype=torch.float32, device=device)
        self.fn = torch.zeros((), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update_state(
        self,
        pointer_logits: torch.Tensor,
        side_logits: torch.Tensor,
        target_accounts: torch.Tensor,
        target_sides: torch.Tensor,
        target_stop: Optional[torch.Tensor] = None,
    ) -> None:
        pred_accounts = pointer_logits.argmax(dim=-1)  # [B, T]
        pred_sides = side_logits.argmax(dim=-1)  # [B, T]

        if target_stop is not None:
            has_stop = (target_stop == self.stop_id).any(dim=-1)  # [B]
            stop_idx = (target_stop == self.stop_id).int().argmax(dim=-1)  # [B]
        else:
            B, T = pred_accounts.shape
            has_stop = torch.zeros((B,), dtype=torch.bool)
            stop_idx = torch.zeros((B,), dtype=torch.int64)

        B = pred_accounts.shape[0]
        for b in range(B):
            if target_stop is not None and has_stop[b]:
                L = int(stop_idx[b].item() + 1)
            else:
                L = int(pred_accounts.shape[1])
            pa = pred_accounts[b, :L].tolist()
            ps = pred_sides[b, :L].tolist()
            ta = target_accounts[b, :L].tolist()
            ts = target_sides[b, :L].tolist()
            pred_pairs = list({(int(a), int(s)) for a, s in zip(pa, ps)})
            true_pairs = list({(int(a), int(s)) for a, s in zip(ta, ts)})
            intersection = 0
            true_set = set(true_pairs)
            for p in pred_pairs:
                if p in true_set:
                    intersection += 1
            fp = max(0, len(pred_pairs) - intersection)
            fn = max(0, len(true_pairs) - intersection)
            self.tp += float(intersection)
            self.fp += float(fp)
            self.fn += float(fn)

    @torch.no_grad()
    def result(self) -> torch.Tensor:
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
        return f1

    @torch.no_grad()
    def reset_states(self) -> None:
        self.tp.zero_()
        self.fp.zero_()
        self.fn.zero_()


class WindowedSetF1Metric:
    """
    Rolling window SetF1 over recent N steps.
    Stores per-batch (tp, fp, fn) tuples and computes F1 from rolling sums.
    """

    def __init__(self, window_size: int = 100, stop_id: int = 1) -> None:
        self.window_size = int(window_size)
        self.stop_id = stop_id
        self.buffer = deque(maxlen=self.window_size)  # type: ignore[var-annotated]

    @torch.no_grad()
    def update_state(
        self,
        pointer_logits: torch.Tensor,
        side_logits: torch.Tensor,
        target_accounts: torch.Tensor,
        target_sides: torch.Tensor,
        target_stop: Optional[torch.Tensor] = None,
    ) -> None:
        pred_accounts = pointer_logits.argmax(dim=-1)  # [B, T]
        pred_sides = side_logits.argmax(dim=-1)  # [B, T]

        B = pred_accounts.shape[0]
        if target_stop is not None:
            has_stop = (target_stop == self.stop_id).any(dim=-1)  # [B]
            stop_idx = (target_stop == self.stop_id).int().argmax(dim=-1)  # [B]
        else:
            has_stop = torch.zeros((B,), dtype=torch.bool)
            stop_idx = torch.zeros((B,), dtype=torch.int64)
        tp_val = 0.0
        fp_val = 0.0
        fn_val = 0.0
        for b in range(B):
            if target_stop is not None and has_stop[b]:
                L = int(stop_idx[b].item() + 1)
            else:
                L = int(pred_accounts.shape[1])
            pa = pred_accounts[b, :L].tolist()
            ps = pred_sides[b, :L].tolist()
            ta = target_accounts[b, :L].tolist()
            ts = target_sides[b, :L].tolist()
            pred_pairs = list({(int(a), int(s)) for a, s in zip(pa, ps)})
            true_pairs = list({(int(a), int(s)) for a, s in zip(ta, ts)})
            true_set = set(true_pairs)
            intersection = 0
            for p in pred_pairs:
                if p in true_set:
                    intersection += 1
            fp_val += max(0, len(pred_pairs) - intersection)
            fn_val += max(0, len(true_pairs) - intersection)
            tp_val += float(intersection)
        self.buffer.append((tp_val, fp_val, fn_val))

    @torch.no_grad()
    def result(self) -> torch.Tensor:
        if not self.buffer:
            return torch.tensor(0.0)
        tp_sum = sum(t for t, _, _ in self.buffer)
        fp_sum = sum(f for _, f, _ in self.buffer)
        fn_sum = sum(n for _, _, n in self.buffer)
        precision = tp_sum / (tp_sum + fp_sum + 1e-6)
        recall = tp_sum / (tp_sum + fn_sum + 1e-6)
        return torch.tensor(2.0 * precision * recall / (precision + recall + 1e-6), dtype=torch.float32)

    @torch.no_grad()
    def reset_states(self) -> None:
        self.buffer.clear()

class SetF1Hungarian:
    """Set-level F1 using Hungarian matching (via SciPy) on CPU tensors."""

    def __init__(self, stop_id: int = 1) -> None:
        self.stop_id = stop_id
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    @torch.no_grad()
    def update_state(
        self,
        pointer_logits: torch.Tensor,
        side_logits: torch.Tensor,
        target_accounts: torch.Tensor,
        target_sides: torch.Tensor,
        target_stop: Optional[torch.Tensor] = None,
    ) -> None:
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
        except Exception:
            return  # silently skip if SciPy unavailable

        B, T, C = pointer_logits.shape
        for b in range(B):
            acc = pointer_logits[b].argmax(dim=-1).cpu().numpy().tolist()
            sid = side_logits[b].argmax(dim=-1).cpu().numpy().tolist()
            if target_stop is not None:
                stop_vec = target_stop[b].cpu().numpy().tolist()
                L = stop_vec.index(self.stop_id) + 1 if self.stop_id in stop_vec else T
            else:
                L = T
            p_pairs = [(int(acc[i]), int(sid[i])) for i in range(L)]
            t_acc = target_accounts[b].cpu().numpy().tolist()
            t_sid = target_sides[b].cpu().numpy().tolist()
            t_pairs = [(int(t_acc[i]), int(t_sid[i])) for i in range(L)]
            if len(p_pairs) == 0 and len(t_pairs) == 0:
                continue
            if len(p_pairs) == 0:
                self.fn += float(len(t_pairs))
                continue
            if len(t_pairs) == 0:
                self.fp += float(len(p_pairs))
                continue
            P = len(p_pairs)
            Tn = len(t_pairs)
            cost = torch.ones((P, Tn), dtype=torch.float32).numpy()
            # zero cost where equal pairs
            t_map = {}
            for j, tp in enumerate(t_pairs):
                t_map.setdefault(tp, []).append(j)
            for i, pp in enumerate(p_pairs):
                if pp in t_map:
                    for j in t_map[pp]:
                        cost[i, j] = 0.0
            row_ind, col_ind = linear_sum_assignment(cost)
            matched = int((cost[row_ind, col_ind] < 0.5).sum())
            self.tp += float(matched)
            self.fp += float(P - matched)
            self.fn += float(Tn - matched)

    @torch.no_grad()
    def result(self) -> float:
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        return float(2.0 * precision * recall / (precision + recall + 1e-6))

    @torch.no_grad()
    def reset_states(self) -> None:
        self.tp = self.fp = self.fn = 0.0


@torch.no_grad()
def _pad_to_same_length(tensors: torch.Tensor, fill: float) -> torch.Tensor:
    # Not used; kept for reference
    return tensors


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
    """
    Align predicted per-side mass over accounts with normalized debit/credit amounts.
    
    - pointer_logits: [B, T, C]
    - side_logits: [B, T, 2]
    - debit_indices: [B, D] (int64 indices into catalog; pad with -1)
    - debit_weights: [B, D] (float; should sum to 1 per row; zeros if no debits)
    - credit_indices: [B, K] (int64; pad with -1)
    - credit_weights: [B, K] (float; sum to 1 per row; zeros if no credits)
    - pointer_probs: Optional pre-computed softmax(pointer_logits, dim=-1)
    - side_probs: Optional pre-computed softmax(side_logits, dim=-1)
    
    Returns mean of per-side MSE across valid rows.
    """
    # Use pre-computed probabilities if provided, otherwise compute
    p = pointer_probs if pointer_probs is not None else torch.softmax(pointer_logits, dim=-1)  # [B, T, C]
    s = side_probs if side_probs is not None else torch.softmax(side_logits, dim=-1)  # [B, T, 2]
    
    # Predicted mass per account for each side
    pred_debit_mass = (s[:, :, 0].unsqueeze(-1) * p).sum(dim=1)  # [B, C]
    pred_credit_mass = (s[:, :, 1].unsqueeze(-1) * p).sum(dim=1)  # [B, C]
    
    def side_mse(pred_mass: torch.Tensor, idxs: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
        idxs = idxs.clone()
        mask = (idxs >= 0).to(pred_mass.dtype)
        idxs_clamped = idxs.clamp(min=0)
        gathered = torch.gather(pred_mass, 1, idxs_clamped)  # [B, L]
        
        # Normalize gathered predictions
        sum_g = gathered.sum(dim=-1, keepdim=True) + 1e-8
        g_norm = torch.where(sum_g > 0, gathered / sum_g, torch.zeros_like(gathered))
        
        # Normalize weights
        wts = torch.where(mask > 0, wts, torch.zeros_like(wts))
        sum_w = wts.sum(dim=-1, keepdim=True) + 1e-8
        w_norm = torch.where(sum_w > 0, wts / sum_w, torch.zeros_like(wts))
        
        mse = ((g_norm - w_norm) ** 2).mean(dim=-1)  # [B]
        valid = (mask.sum(dim=-1) > 0).to(mse.dtype)
        return (mse * valid).sum() / (valid.sum() + 1e-6)
    
    ld = side_mse(pred_debit_mass, debit_indices, debit_weights)
    lc = side_mse(pred_credit_mass, credit_indices, credit_weights)
    return 0.5 * (ld + lc)
