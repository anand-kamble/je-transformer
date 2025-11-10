from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from models.hash_utils import hash_strings_to_buckets
from models.pointer import PointerLayer


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


class JEModel(nn.Module):
    def __init__(
        self,
        encoder_loc: str = "bert-base-multilingual-cased",
        hidden_dim: int = 256,
        max_lines: int = 8,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_lines = int(max_lines)

        self.encoder = AutoModel.from_pretrained(encoder_loc)
        enc_dim = self.encoder.config.hidden_size
        self.enc_proj = nn.Linear(enc_dim, hidden_dim)

        # Conditioning
        num_dim = max(32, hidden_dim // 4)
        self.cond_num_mlp = nn.Sequential(nn.Linear(8, num_dim), nn.ReLU())
        self.cur_emb = nn.Embedding(128, max(8, hidden_dim // 32))
        self.typ_emb = nn.Embedding(256, max(12, hidden_dim // 24))
        self.cond_proj = nn.Sequential(
            nn.Linear(num_dim + self.cur_emb.embedding_dim + self.typ_emb.embedding_dim, hidden_dim),
            nn.ReLU(),
        )
        self.film_params = nn.Linear(hidden_dim, 2 * hidden_dim)

        # Decoder
        self.prev_side_emb = nn.Embedding(2, hidden_dim)
        self.dec_in_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Retrieval
        self.retr_mem_proj = nn.Linear(hidden_dim, hidden_dim)
        self.retrieval_gate = nn.Linear(hidden_dim, 1)
        self.post_retr_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Heads
        self.pointer = PointerLayer(temperature=temperature)
        self.side_head = nn.Linear(hidden_dim, 2)
        self.stop_head = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_ids = inputs["input_ids"].long()
        attention_mask = inputs["attention_mask"].long()
        prev_account_idx = inputs["prev_account_idx"].long()
        prev_side_id = inputs["prev_side_id"].long()
        cat_embs = inputs["catalog_embeddings"].to(dtype=torch.float32)
        retr_mem = inputs.get("retrieval_memory", None)
        cond_numeric = inputs["cond_numeric"].to(dtype=torch.float32)
        currency = inputs["currency"]
        je_type = inputs["journal_entry_type"]

        B = input_ids.shape[0]
        T = prev_account_idx.shape[1]

        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = enc_out.last_hidden_state  # [B, L, Henc]
        enc_pooled = mean_pool(last_hidden, attention_mask)
        enc_proj = torch.tanh(self.enc_proj(enc_pooled))  # [B, H]

        # FiLM conditioning
        cond_num_vec = self.cond_num_mlp(cond_numeric)
        cur_ids = hash_strings_to_buckets(currency, self.cur_emb.num_embeddings).to(input_ids.device)
        typ_ids = hash_strings_to_buckets(je_type, self.typ_emb.num_embeddings).to(input_ids.device)
        cur_e = self.cur_emb(cur_ids)
        typ_e = self.typ_emb(typ_ids)
        cond_vec = self.cond_proj(torch.cat([cond_num_vec, cur_e, typ_e], dim=-1))
        gamma_beta = self.film_params(cond_vec)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)
        enc_ctx = enc_proj * (1.0 + gamma) + beta  # [B, H]

        # Teacher forcing embeddings
        if cat_embs.dim() == 2:
            cat_b = cat_embs.unsqueeze(0).expand(B, -1, -1)  # [B, C, H]
        else:
            cat_b = cat_embs
        safe_idx = prev_account_idx.clamp_min(0)
        prev_acc_emb = torch.zeros(B, T, self.hidden_dim, dtype=cat_b.dtype, device=cat_b.device)
        for b in range(B):
            prev_acc_emb[b] = cat_b[b].index_select(0, safe_idx[b])
        bos_mask = (prev_account_idx == -1).unsqueeze(-1).to(prev_acc_emb.dtype)
        prev_acc_emb = prev_acc_emb * (1.0 - bos_mask)

        safe_side = prev_side_id.clamp_min(0)
        prev_side_emb = self.prev_side_emb(safe_side)  # [B, T, H]
        side_bos_mask = (prev_side_id == -1).unsqueeze(-1).to(prev_side_emb.dtype)
        prev_side_emb = prev_side_emb * (1.0 - side_bos_mask)

        enc_tiled = enc_ctx.unsqueeze(1).expand(B, T, self.hidden_dim)
        dec_in = torch.cat([enc_tiled, prev_acc_emb, prev_side_emb], dim=-1)
        dec_in = F.relu(self.dec_in_proj(dec_in))

        dec_h, _ = self.gru(dec_in)  # [B, T, H]

        # Retrieval attention
        if retr_mem is None:
            retr_b = torch.zeros((B, 1, self.hidden_dim), dtype=dec_h.dtype, device=dec_h.device)
        else:
            mem = retr_mem.to(dtype=dec_h.dtype)
            if mem.dim() == 2:
                retr_b = mem.unsqueeze(0).expand(B, -1, -1)
            else:
                retr_b = mem
        retr_proj = self.retr_mem_proj(retr_b)  # [B, K, H]
        dec_n = F.normalize(dec_h, p=2, dim=-1)
        mem_n = F.normalize(retr_proj, p=2, dim=-1)
        scores = torch.einsum("bth,bkh->btk", dec_n, mem_n)
        weights = F.softmax(scores, dim=-1)
        retr_ctx = torch.einsum("btk,bkh->bth", weights, retr_proj)
        gate = torch.sigmoid(self.retrieval_gate(dec_h))
        retr_ctx = retr_ctx * gate
        dec_h_fused = torch.cat([dec_h, retr_ctx], dim=-1)
        dec_h_fused = F.relu(self.post_retr_proj(dec_h_fused))

        flat_dec = dec_h_fused.reshape(B * T, self.hidden_dim)
        logits_flat = self.pointer(flat_dec, cat_embs)  # [B*T, C]
        C = logits_flat.shape[-1]
        pointer_logits = logits_flat.reshape(B, T, C)

        side_logits = self.side_head(dec_h_fused)
        stop_logits = self.stop_head(dec_h_fused)

        return {"pointer_logits": pointer_logits, "side_logits": side_logits, "stop_logits": stop_logits}


def build_je_model(
    encoder_loc: str = "bert-base-multilingual-cased",
    hidden_dim: int = 256,
    max_lines: int = 8,
    temperature: float = 1.0,
):
    return JEModel(encoder_loc=encoder_loc, hidden_dim=hidden_dim, max_lines=max_lines, temperature=temperature)



