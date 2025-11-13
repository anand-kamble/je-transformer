from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from inference.retrieval_memory import build_retrieval_memory_for_text
from models.hierarchy_utils import AccountHierarchy


@dataclass
class BeamState:
    accounts: List[int]
    sides: List[int]
    logprob: float
    finished: bool

    def length(self) -> int:
        return len(self.accounts)


def length_penalized_score(logprob: float, length: int, alpha: float) -> float:
    L = max(1, length)
    if alpha <= 0.0:
        return logprob
    return logprob / (L ** alpha)


# Add parent-aware beam expansion
def expand_with_hierarchy(
    beam_candidates: List[dict],
    hierarchy: AccountHierarchy,
    fallback_threshold: float = 0.3
):
    """
    If top predictions have low confidence, 
    add parent accounts as additional candidates
    """
    expanded = []
    for cand in beam_candidates:
        expanded.append(cand)
        
        # If confidence is low, consider parent
        if cand['prob'] < fallback_threshold:
            last_account = cand['accounts'][-1] if cand['accounts'] else None
            if last_account is not None:
                parent_idx = hierarchy.get_parent(last_account)
                if parent_idx is not None:
                    parent_cand = cand.copy()
                    parent_cand['accounts'] = cand['accounts'].copy()
                    parent_cand['accounts'][-1] = parent_idx
                    parent_cand['prob'] *= 0.9  # Small penalty for fallback
                    expanded.append(parent_cand)
    
    return expanded


def beam_search_decode(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    catalog_embeddings: torch.Tensor,
    retrieval_memory: Optional[torch.Tensor] = None,
    cond_numeric: Optional[torch.Tensor] = None,
    currency: Optional[List[str]] = None,
    journal_entry_type: Optional[List[str]] = None,
    beam_size: int = 10,
    alpha: float = 0.7,
    max_lines: int = 8,
    topk_accounts: Optional[int] = None,
    tau: Optional[float] = None,
    
    query_text: Optional[str] = None,
    index_dir: Optional[str] = None,
    ids_uri: Optional[str] = None,
    embeddings_uri: Optional[str] = None,
    tokenizer_loc: str = "bert-base-multilingual-cased",
    encoder_loc: str = "bert-base-multilingual-cased",
    use_cls: bool = False,
    top_k_retrieval: int = 5,
) -> List[Dict[str, Any]]:
    if retrieval_memory is None:
        if query_text and index_dir and ids_uri and embeddings_uri:
            retrieval_memory = build_retrieval_memory_for_text(
                query_text, index_dir, ids_uri, embeddings_uri, tokenizer_loc=tokenizer_loc, encoder_loc=encoder_loc, use_cls=use_cls, top_k=top_k_retrieval
            )
        else:
            retrieval_memory = torch.zeros((1, int(catalog_embeddings.shape[-1])), dtype=torch.float32, device=catalog_embeddings.device)
    if cond_numeric is None:
        cond_numeric = torch.zeros((1, 8), dtype=torch.float32, device=input_ids.device)
    if currency is None:
        currency = [""]
    if journal_entry_type is None:
        journal_entry_type = [""]

    C = int(catalog_embeddings.shape[0])
    if topk_accounts is None or topk_accounts > C:
        topk_accounts = min(C, beam_size)

    beams: List[BeamState] = [BeamState(accounts=[], sides=[], logprob=0.0, finished=False)]
    finished_beams: List[BeamState] = []

    B = int(input_ids.shape[0])
    assert B == 1, "beam_search_decode expects batch size 1"

    for t in range(max_lines):
        new_beams: List[BeamState] = []
        for b in beams:
            if b.finished:
                new_beams.append(b)
                continue

            prev_acc = torch.full((1, max_lines), -1, dtype=torch.long, device=input_ids.device)
            prev_side = torch.full((1, max_lines), -1, dtype=torch.long, device=input_ids.device)
            if b.accounts:
                prev_acc[0, : len(b.accounts)] = torch.tensor(b.accounts, dtype=torch.long, device=input_ids.device)
                prev_side[0, : len(b.accounts)] = torch.tensor(b.sides, dtype=torch.long, device=input_ids.device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prev_account_idx=prev_acc,
                    prev_side_id=prev_side,
                    catalog_embeddings=catalog_embeddings,
                    retrieval_memory=retrieval_memory,
                    cond_numeric=cond_numeric,
                    currency=currency,
                    journal_entry_type=journal_entry_type,
                )
            ptr_logits = outputs["pointer_logits"][0, t, :].detach().cpu().numpy()  
            side_logits = outputs["side_logits"][0, t, :].detach().cpu().numpy()   
            stop_logits = outputs["stop_logits"][0, t, :].detach().cpu().numpy()   

            
            ptr_logp = ptr_logits - np.logaddexp.reduce(ptr_logits)
            side_logp = side_logits - np.logaddexp.reduce(side_logits)
            stop_logp = stop_logits - np.logaddexp.reduce(stop_logits)

            
            stop_lp = float(stop_logp[1])
            new_beams.append(BeamState(accounts=b.accounts.copy(), sides=b.sides.copy(), logprob=b.logprob + stop_lp, finished=True))

            
            cont_lp = float(stop_logp[0])
            top_idx = np.argpartition(-ptr_logp, range(topk_accounts))[:topk_accounts]
            top_idx = top_idx[np.argsort(-ptr_logp[top_idx])]
            for acc_id in top_idx:
                for side_id in (0, 1):
                    lp = cont_lp + float(ptr_logp[acc_id]) + float(side_logp[side_id])
                    new_beams.append(
                        BeamState(accounts=b.accounts + [int(acc_id)], sides=b.sides + [int(side_id)], logprob=b.logprob + lp, finished=False)
                    )

        new_beams.sort(key=lambda bs: length_penalized_score(bs.logprob, max(1, bs.length()), alpha), reverse=True)
        beams = new_beams[:beam_size]
        finished_beams.extend([b for b in beams if b.finished])
        if all(b.finished for b in beams):
            break

    if not finished_beams:
        finished_beams = beams

    results = []
    for b in finished_beams:
        score = length_penalized_score(b.logprob, max(1, b.length()), alpha)
        results.append({
            "accounts": b.accounts,
            "sides": b.sides,
            "score": float(score),
            "prob": float(math.exp(score)),
            "logprob": float(b.logprob),
            "length": int(b.length()),
        })
    results.sort(key=lambda r: r["score"], reverse=True)

    if tau is not None:
        kept = [r for r in results if r["prob"] >= float(tau)]
        if kept:
            return kept
        return results[:1]
    return results


