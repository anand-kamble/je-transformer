from __future__ import annotations

import hashlib
from typing import Iterable, List

import torch


def hash_strings_to_buckets(values: Iterable[str], num_bins: int) -> torch.LongTensor:
    ids: List[int] = []
    for v in values:
        s = "" if v is None else str(v)
        h = hashlib.md5(s.encode("utf-8")).hexdigest()
        
        ids.append(int(h, 16) % int(num_bins))
    return torch.tensor(ids, dtype=torch.long)


def md5_to_bucket(text: str, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive")
    if text is None:
        text = ""
    digest = hashlib.md5(text.encode("utf-8")).digest()
    
    val = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(val % num_buckets)


