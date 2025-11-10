from __future__ import annotations

from typing import Optional

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


def pointer_loss(pointer_logits: torch.Tensor, target_account_idx: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    return masked_sparse_ce(pointer_logits, target_account_idx, ignore_index=ignore_index)


def side_loss(side_logits: torch.Tensor, target_side_id: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    return masked_sparse_ce(side_logits, target_side_id, ignore_index=ignore_index)


def stop_loss(stop_logits: torch.Tensor, target_stop_id: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    return masked_sparse_ce(stop_logits, target_stop_id, ignore_index=ignore_index)


def coverage_penalty(pointer_logits: torch.Tensor, max_total: float = 1.0) -> torch.Tensor:
    """
    Encourage the model not to overspread mass repeatedly on the same accounts across time.
    Compute softmax over accounts per step, sum over time per account, and penalize surplus over max_total.
      - pointer_logits: [B, T, C]
    Returns mean surplus across batch.
    """
    p = torch.softmax(pointer_logits, dim=-1)  # [B, T, C]
    sum_over_t = p.sum(dim=1)  # [B, C]
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
            # first index where target_stop == stop_id, else T
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
