from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.hash_utils import md5_to_bucket
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
        pointer_scale_init: float = 1.0,
        pointer_learnable_scale: bool = False,
        use_pointer_norm: bool = True,
        learn_catalog: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.max_lines = int(max_lines)
        self._learn_catalog = bool(learn_catalog)

        # Text encoder
        self.encoder = AutoModel.from_pretrained(encoder_loc)
        enc_width = int(self.encoder.config.hidden_size)
        self.enc_proj = nn.Linear(enc_width, hidden_dim)

        # Conditioning: numeric [8] -> FiLM params
        self.cond_num_mlp = nn.Sequential(
            nn.Linear(8, max(32, hidden_dim // 4)), nn.ReLU(), nn.Linear(max(32, hidden_dim // 4), hidden_dim)
        )
        # Categorical embeddings via hashing buckets
        self.cur_emb = nn.Embedding(128, max(8, hidden_dim // 32))
        self.typ_emb = nn.Embedding(256, max(12, hidden_dim // 24))
        self.film_params = nn.Linear(hidden_dim + self.cur_emb.embedding_dim + self.typ_emb.embedding_dim, 2 * hidden_dim)

        # Decoder input projection and GRU
        self.dec_inp_proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=3 ,batch_first=True, dropout=0.1)

        # Retrieval projection and gate
        self.retr_mem_proj = nn.Linear(hidden_dim, hidden_dim)
        self.retr_gate = nn.Linear(hidden_dim, 1)
        self.post_retr_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # Heads
        self.pointer = PointerLayer(
            temperature=temperature,
            learnable_scale=pointer_learnable_scale,
            scale_init=pointer_scale_init,
            use_norm=use_pointer_norm,
        )
        self.side_head = nn.Linear(hidden_dim, 2)
        self.stop_head = nn.Linear(hidden_dim, 2)

        # Embedding for previous side id (0/1), BOS handled by zeros externally
        self.prev_side_emb = nn.Embedding(2, hidden_dim)

    def set_catalog_embeddings(self, catalog_embeddings: torch.Tensor) -> None:
        """
        Optionally register catalog embeddings as a trainable parameter.
        """
        self.catalog_param = nn.Parameter(catalog_embeddings.detach().clone().to(dtype=torch.float32))

    def _hash_strs(self, xs: List[str], buckets: int) -> torch.Tensor:
        ids = [md5_to_bucket(x or "", buckets) for x in xs]
        return torch.tensor(ids, dtype=torch.long, device=self.enc_proj.weight.device)

    def _cat_embs(self, currency: Union[List[str], torch.Tensor], journal_entry_type: Union[List[str], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(currency, torch.Tensor):
            cur_ids = currency.to(dtype=torch.long, device=self.cur_emb.weight.device)
        else:
            cur_ids = self._hash_strs(currency, 128)
        if isinstance(journal_entry_type, torch.Tensor):
            typ_ids = journal_entry_type.to(dtype=torch.long, device=self.typ_emb.weight.device)
        else:
            typ_ids = self._hash_strs(journal_entry_type, 256)
        return self.cur_emb(cur_ids), self.typ_emb(typ_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prev_account_idx: torch.Tensor,
        prev_side_id: torch.Tensor,
        catalog_embeddings: torch.Tensor,
        retrieval_memory: Optional[torch.Tensor],
        cond_numeric: torch.Tensor,
        currency: Union[List[str], torch.Tensor],
        journal_entry_type: Union[List[str], torch.Tensor],
		return_retrieval_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        B = input_ids.shape[0]
        T = self.max_lines
        # Encoder
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = enc_out.last_hidden_state  # [B, L, Henc]
        enc_pooled = mean_pool(last_hidden, attention_mask)
        enc_proj = torch.tanh(self.enc_proj(enc_pooled))  # [B, H]

        # Conditioning FiLM
        cur_vec, typ_vec = self._cat_embs(currency, journal_entry_type)
        cond_num_vec = self.cond_num_mlp(cond_numeric)
        cond_vec = torch.cat([cond_num_vec, cur_vec, typ_vec], dim=-1)
        gamma_beta = self.film_params(cond_vec)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        enc_ctx = enc_proj * (1.0 + gamma) + beta  # [B, H]

        # Prepare decoder inputs
        cat_embs = catalog_embeddings  # [C, H] or [B, C, H]
        safe_idx = prev_account_idx.clamp_min(0)
        prev_acc_emb = F.embedding(safe_idx, cat_embs if cat_embs.dim() == 2 else cat_embs[0])  # [B, T, H]
        bos_mask = (prev_account_idx == -1).to(prev_acc_emb.dtype).unsqueeze(-1)
        prev_acc_emb = prev_acc_emb * (1.0 - bos_mask)

        safe_side = prev_side_id.clamp_min(0)
        prev_side_emb = self.prev_side_emb(safe_side)
        bos_side_mask = (prev_side_id == -1).to(prev_side_emb.dtype).unsqueeze(-1)
        prev_side_emb = prev_side_emb * (1.0 - bos_side_mask)

        enc_tiled = enc_ctx.unsqueeze(1).expand(B, T, -1)
        dec_inp = torch.cat([enc_tiled, prev_acc_emb, prev_side_emb], dim=-1)
        dec_inp = F.relu(self.dec_inp_proj(dec_inp))

        dec_h, _ = self.gru(dec_inp)  # [B, T, H]

        # Retrieval fusion
        if retrieval_memory is None:
            cat_src = self.catalog_param if (self._learn_catalog and hasattr(self, "catalog_param")) else catalog_embeddings
            H = cat_src.shape[-1] if cat_src.dim() == 2 else cat_src.shape[-1]
            retrieval_memory = torch.zeros((1, H), dtype=dec_h.dtype, device=dec_h.device)
        retr_mem_proj = self.retr_mem_proj(retrieval_memory)
        if retr_mem_proj.dim() == 2:
            retr_mem_proj = retr_mem_proj.unsqueeze(0).expand(B, -1, -1)  # [B, K, H]
        dec_n = F.normalize(dec_h, dim=-1)
        mem_n = F.normalize(retr_mem_proj, dim=-1)
        scores = torch.einsum("bth,bkh->btk", dec_n, mem_n)
        weights = torch.softmax(scores, dim=-1)
        retr_ctx = torch.einsum("btk,bkh->bth", weights, retr_mem_proj)
        gate = torch.sigmoid(self.retr_gate(dec_h))
        retr_ctx = retr_ctx * gate
        dec_h_fused = torch.cat([dec_h, retr_ctx], dim=-1)
        dec_h_fused = F.relu(self.post_retr_proj(dec_h_fused))

        flat_dec = dec_h_fused.reshape(B * T, self.hidden_dim)
        # Choose catalog embeddings (internal trainable if available)
        cat_for_pointer = self.catalog_param if (self._learn_catalog and hasattr(self, "catalog_param")) else catalog_embeddings
        logits_flat = self.pointer(flat_dec, cat_for_pointer)
        pointer_logits = logits_flat.reshape(B, T, -1)

        side_logits = self.side_head(dec_h_fused)
        stop_logits = self.stop_head(dec_h_fused)
        out: Dict[str, torch.Tensor] = {
			"pointer_logits": pointer_logits,
			"side_logits": side_logits,
			"stop_logits": stop_logits,
		}
        if return_retrieval_weights:
            out["retrieval_weights"] = weights
        return out
