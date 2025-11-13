from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

class AccountHierarchy:
    def __init__(self, accounts: List[Dict[str, str]]):
        self.accounts = accounts
        self.code_to_idx = {a['number']: i for i, a in enumerate(accounts)}
        self.parent_map = {}  # child_idx -> parent_idx
        self.children_map = {}  # parent_idx -> [child_indices]
        self.depth_map = {}  # idx -> depth in tree
        self.path_map = {}  # idx -> [ancestor_indices] from root to node
        
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        for idx, account in enumerate(self.accounts):
            code = account['number']
            parts = code.split('-')
            depth = len(parts)
            self.depth_map[idx] = depth
            
            # Build path from root
            path = []
            for i in range(1, len(parts) + 1):
                parent_code = '-'.join(parts[:i])
                if parent_code in self.code_to_idx:
                    path.append(self.code_to_idx[parent_code])
            self.path_map[idx] = path
            
            # Identify parent
            if depth > 1:
                parent_code = '-'.join(parts[:-1])
                if parent_code in self.code_to_idx:
                    parent_idx = self.code_to_idx[parent_code]
                    self.parent_map[idx] = parent_idx
                    if parent_idx not in self.children_map:
                        self.children_map[parent_idx] = []
                    self.children_map[parent_idx].append(idx)
    
    def get_parent(self, idx: int) -> Optional[int]:
        return self.parent_map.get(idx)
    
    def get_children(self, idx: int) -> List[int]:
        return self.children_map.get(idx, [])
    
    def get_ancestors(self, idx: int) -> List[int]:
        """Returns all ancestors from root to parent"""
        path = self.path_map.get(idx, [])
        return path[:-1] if path else []  # Exclude self