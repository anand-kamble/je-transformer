from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from models.hash_utils import md5_to_bucket


class CatalogEncoder(nn.Module):
    """
    Money-flow-aware account encoder (lightweight, hash-based).

    Inputs: dict with Python lists of strings (length = num_accounts):
      - number: account code like "502-0116"
      - name: account name
      - nature: e.g., "A","L","E","R","X" (asset/liability/...)

    Produces: float32 tensor of shape [num_accounts, emb_dim]
    """

    def __init__(
        self,
        emb_dim: int = 256,
        code_hash_bins: int = 4096,
        name_hash_bins: int = 16384,
        nature_hash_bins: int = 32,
    ) -> None:
        super().__init__()
        self.emb_dim = int(emb_dim)
        proj_dim = max(64, emb_dim // 2)

        self.code_hash_bins = int(code_hash_bins)
        self.name_hash_bins = int(name_hash_bins)
        self.nature_hash_bins = int(nature_hash_bins)

        # Embeddings for hashed ids
        self.code_emb = nn.Embedding(self.code_hash_bins, proj_dim)
        self.name_emb = nn.Embedding(self.name_hash_bins, proj_dim)
        self.nature_emb = nn.Embedding(self.nature_hash_bins, max(16, emb_dim // 8))

        self.proj = nn.Linear(proj_dim * 2 + max(16, emb_dim // 8), self.emb_dim)
        self.norm = nn.LayerNorm(self.emb_dim)

    @torch.no_grad()
    def _hash_strings(self, xs: List[str], num_buckets: int) -> torch.Tensor:
        idxs = [md5_to_bucket(x or "", num_buckets) for x in xs]
        return torch.tensor(idxs, dtype=torch.long)

    def forward(self, inputs: Dict[str, List[str]]) -> torch.Tensor:
        numbers = inputs.get("number", [])
        names = inputs.get("name", [])
        natures = inputs.get("nature", [])
        if not (len(numbers) == len(names) == len(natures)):
            raise ValueError("All input fields must have the same length")

        code_ids = self._hash_strings(numbers, self.code_hash_bins)
        name_ids = self._hash_strings(names, self.name_hash_bins)
        nature_ids = self._hash_strings(natures, self.nature_hash_bins)

        code_vec = self.code_emb(code_ids)
        name_vec = self.name_emb(name_ids)
        nature_vec = self.nature_emb(nature_ids)

        x = torch.cat([code_vec, name_vec, nature_vec], dim=-1)
        x = self.proj(x)
        x = self.norm(x)
        return x
