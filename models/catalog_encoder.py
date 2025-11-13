from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from models.hash_utils import md5_to_bucket
from models.hierarchy_utils import AccountHierarchy


class CatalogEncoder(nn.Module):
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


class HierarchicalCatalogEncoder(nn.Module):
    def __init__(
        self,
        emb_dim: int = 256,
        max_depth: int = 5,
        name_hash_bins: int = 16384,
        nature_hash_bins: int = 32,
        use_positional: bool = True,
        use_path_aggregation: bool = True,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_depth = max_depth
        
        # Decompose account code into hierarchical components
        self.depth_emb = nn.Embedding(max_depth + 1, emb_dim // 8)
        self.segment_emb = nn.ModuleList([
            nn.Embedding(1000, emb_dim // 4)  # Each segment gets its own embedding
            for _ in range(max_depth)
        ])
        
        # Original embeddings for name and nature
        self.name_emb = nn.Embedding(name_hash_bins, emb_dim // 2)
        self.nature_emb = nn.Embedding(nature_hash_bins, emb_dim // 8)
        
        # Tree positional encoding (similar to Tree Transformers)
        if use_positional:
            self.position_emb = nn.Embedding(1000, emb_dim // 8)  # Position among siblings
        
        # Path aggregation layers
        if use_path_aggregation:
            self.path_gru = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim // 2,
                num_layers=1,
                batch_first=True
            )
        
        # Final projection with residual
        proj_input_dim = (
            emb_dim // 4 * max_depth +  # Segment embeddings
            emb_dim // 2 +  # Name
            emb_dim // 8 * 3  # Nature + depth + position
        )
        
        self.proj = nn.Linear(proj_input_dim, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        
        # Parent-child relationship modeling
        self.parent_child_attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

    def forward(
        self, 
        inputs: Dict[str, List[str]], 
        hierarchy: Optional[AccountHierarchy] = None
    ) -> torch.Tensor:
        numbers = inputs.get("number", [])
        names = inputs.get("name", [])
        natures = inputs.get("nature", [])
        
        batch_size = len(numbers)
        device = self.depth_emb.weight.device
        
        all_segment_embs = []
        depth_indices = []
        
        # Parse hierarchical structure
        for number in numbers:
            segments = number.split('-')
            depth = len(segments)
            depth_indices.append(min(depth, self.max_depth))
            
            # Embed each segment separately
            segment_embs_list = []
            for i, segment in enumerate(segments[:self.max_depth]):
                # Convert segment to index (you might want a more sophisticated approach)
                try:
                    seg_idx = int(segment) % 1000
                except:
                    seg_idx = hash(segment) % 1000
                
                seg_emb = self.segment_emb[i](
                    torch.tensor([seg_idx], device=device)
                )
                segment_embs_list.append(seg_emb)
            
            # Pad if needed
            while len(segment_embs_list) < self.max_depth:
                segment_embs_list.append(
                    torch.zeros(1, self.emb_dim // 4, device=device)
                )
            
            all_segment_embs.append(torch.cat(segment_embs_list, dim=-1))
        
        # Stack segment embeddings
        segment_features = torch.cat(all_segment_embs, dim=0)  # [B, emb_dim // 4 * max_depth]
        
        # Depth embedding
        depth_tensor = torch.tensor(depth_indices, device=device)
        depth_features = self.depth_emb(depth_tensor)  # [B, emb_dim // 8]
        
        # Name and nature embeddings (keep existing logic)
        name_ids = self._hash_strings(names, self.name_emb.num_embeddings)
        nature_ids = self._hash_strings(natures, self.nature_emb.num_embeddings)
        
        name_features = self.name_emb(name_ids.to(device))
        nature_features = self.nature_emb(nature_ids.to(device))
        
        # Combine all features
        combined = torch.cat([
            segment_features,
            name_features,
            nature_features,
            depth_features,
        ], dim=-1)
        
        # Project and normalize
        output = self.proj(combined)
        output = self.norm(output)
        
        # If hierarchy is provided, enhance with parent-child relationships
        if hierarchy is not None:
            output = self._enhance_with_hierarchy(output, hierarchy)
        
        return output
    
    def _enhance_with_hierarchy(
        self, 
        embeddings: torch.Tensor,
        hierarchy: AccountHierarchy
    ) -> torch.Tensor:
        """Add parent-child relational information"""
        batch_size = embeddings.size(0)
        
        # Build attention mask based on parent-child relationships
        attn_mask = torch.zeros(batch_size, batch_size, device=embeddings.device)
        
        for i in range(batch_size):
            # Allow attention to self
            attn_mask[i, i] = 1.0
            
            # Allow attention to parent
            parent_idx = hierarchy.get_parent(i)
            if parent_idx is not None and parent_idx < batch_size:
                attn_mask[i, parent_idx] = 1.0
            
            # Allow attention to children
            children = hierarchy.get_children(i)
            for child_idx in children:
                if child_idx < batch_size:
                    attn_mask[i, child_idx] = 1.0
        
        # Apply masked self-attention
        attn_output, _ = self.parent_child_attn(
            embeddings.unsqueeze(0),
            embeddings.unsqueeze(0),
            embeddings.unsqueeze(0),
            attn_mask=attn_mask.unsqueeze(0)
        )
        
        # Residual connection
        output = embeddings + 0.5 * attn_output.squeeze(0)
        
        return output

    @torch.no_grad()
    def _hash_strings(self, xs: List[str], num_buckets: int) -> torch.Tensor:
        from models.hash_utils import md5_to_bucket
        idxs = [md5_to_bucket(x or "", num_buckets) for x in xs]
        return torch.tensor(idxs, dtype=torch.long)