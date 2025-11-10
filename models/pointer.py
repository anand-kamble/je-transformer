from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerLayer(nn.Module):
    """
    Dot-product pointer over a catalog (with L2 normalization and optional mask).

    forward(decoder_state, catalog_embeddings, mask=None) -> logits
      - decoder_state: [batch, hidden]
      - catalog_embeddings: [catalog_size, hidden] or [batch, catalog_size, hidden]
      - mask: [catalog_size] or [batch, catalog_size], 1 for valid, 0 to mask

    Returns:
      - logits over catalog indices: [batch, catalog_size]
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = float(max(1e-6, temperature))

    def forward(
        self,
        decoder_state: torch.Tensor,
        catalog_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dec = decoder_state.to(dtype=torch.float32)  # [B, H]
        cat = catalog_embeddings.to(dtype=torch.float32)  # [C, H] or [B, C, H]

        if cat.dim() == 2:
            # [C, H] -> [B, C, H]
            cat = cat.unsqueeze(0).expand(dec.shape[0], -1, -1)

        # L2 normalize
        dec_n = F.normalize(dec, p=2.0, dim=-1)  # [B, H]
        cat_n = F.normalize(cat, p=2.0, dim=-1)  # [B, C, H]

        # Dot product -> [B, C]
        logits = torch.einsum("bh,bch->bc", dec_n, cat_n) / self.temperature

        if mask is not None:
            m = mask.to(dtype=torch.float32)
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(dec.shape[0], -1)
            very_neg = torch.tensor(-1e9, dtype=torch.float32, device=logits.device)
            logits = torch.where(m > 0.5, logits, very_neg)

        return logits

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"
