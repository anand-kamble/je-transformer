from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerLayer(nn.Module):
    def __init__(
        self,
        temperature: float = 1.0,
        learnable_scale: bool = False,
        scale_init: float = 1.0,
        use_norm: bool = True,
        logit_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.temperature = float(max(1e-6, temperature))
        self.logit_clip = logit_clip
        self.use_norm = bool(use_norm)
        if learnable_scale:
            self.logit_scale = nn.Parameter(torch.tensor(float(scale_init), dtype=torch.float32))
            self._scale_learnable = True
        else:
            
            self.register_buffer("logit_scale", torch.tensor(float(scale_init), dtype=torch.float32), persistent=False)
            self._scale_learnable = False

    def forward(
        self,
        decoder_state: torch.Tensor,
        catalog_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dec = decoder_state.to(dtype=torch.float32)  
        cat = catalog_embeddings.to(dtype=torch.float32)  

        if cat.dim() == 2:
            
            cat = cat.unsqueeze(0).expand(dec.shape[0], -1, -1)

        
        if self.use_norm:
            dec = F.normalize(dec, p=2.0, dim=-1)  
            cat = F.normalize(cat, p=2.0, dim=-1)  

        
        logits = torch.einsum("bh,bch->bc", dec, cat)
        logits = torch.clamp(
            logits, 
            min=-self.logit_clip, 
            max=self.logit_clip
        )

        
        logits = (self.logit_scale * logits) / self.temperature

        if mask is not None:
            m = mask.to(dtype=torch.float32)
            if m.dim() == 1:
                m = m.unsqueeze(0).expand(dec.shape[0], -1)
            very_neg = torch.tensor(-1e9, dtype=torch.float32, device=logits.device)
            logits = torch.where(m > 0.5, logits, very_neg)

        return logits

    def extra_repr(self) -> str:
        scale = float(self.logit_scale.detach().cpu()) if isinstance(self.logit_scale, torch.Tensor) else float(self.logit_scale)
        return f"temperature={self.temperature}, scale={scale}, learnable_scale={self._scale_learnable}, use_norm={self.use_norm}"
