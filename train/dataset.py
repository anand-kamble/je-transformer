from __future__ import annotations

import glob
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from data.text_normalization import normalize_description


def build_targets_from_sets(
    debits: List[int], credits: List[int], max_lines: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d_len = len(debits)
    c_len = len(credits)
    seq_acc = debits + credits
    seq_side = [0] * d_len + [1] * c_len
    L = len(seq_acc)
    Lc = min(L, max_lines)

    target_accounts = np.full((max_lines,), -1, dtype=np.int64)
    target_sides = np.full((max_lines,), -1, dtype=np.int64)
    if Lc > 0:
        target_accounts[:Lc] = np.asarray(seq_acc[:Lc], dtype=np.int64)
        target_sides[:Lc] = np.asarray(seq_side[:Lc], dtype=np.int64)

    prev_accounts = np.full((max_lines,), -1, dtype=np.int64)
    prev_sides = np.full((max_lines,), -1, dtype=np.int64)
    if Lc > 0:
        prev_accounts[1:Lc] = target_accounts[: Lc - 1]
        prev_sides[1:Lc] = target_sides[: Lc - 1]

    stop = np.zeros((max_lines,), dtype=np.int64)
    stop[0 if Lc == 0 else Lc - 1] = 1
    return prev_accounts, prev_sides, target_accounts, target_sides, stop


class ParquetJEDataset(Dataset):
    def __init__(
        self,
        pattern: str,
        tokenizer_loc: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        max_lines: int = 8,
    ) -> None:
        self.files: List[str] = sorted(glob.glob(pattern)) if not pattern.startswith("gs://") else [pattern]
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_loc, use_fast=True)
        self.max_length = int(max_length)
        self.max_lines = int(max_lines)
        if not self.files:
            raise ValueError(f"No Parquet files matched pattern: {pattern}")
        dfs = [pd.read_parquet(p) for p in self.files] if not pattern.startswith("gs://") else [pd.read_parquet(pattern)]
        self.df = pd.concat(dfs, ignore_index=True)

    def __len__(self) -> int:
        return int(len(self.df))

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        row = self.df.iloc[idx]
        desc = normalize_description(str(row.get("description", "")))
        enc = self.tokenizer(
            desc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)

        # Helpers to safely convert Parquet-loaded list-like fields
        def _safe_to_list(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            # Only call pd.isna for obvious scalars, not arrays
            if isinstance(value, (bytes, bytearray)):
                try:
                    s_text_bytes: str = value.decode("utf-8").strip()
                except Exception:
                    return []
                if len(s_text_bytes) >= 2 and ((s_text_bytes[0] == "[" and s_text_bytes[-1] == "]") or (s_text_bytes[0] == "(" and s_text_bytes[-1] == ")")):
                    try:
                        s_json = "[" + s_text_bytes[1:-1] + "]" if (s_text_bytes[0] == "(" and s_text_bytes[-1] == ")") else s_text_bytes
                        parsed = json.loads(s_json)
                        return list(parsed) if isinstance(parsed, (list, tuple)) else []
                    except Exception:
                        return []
                return []
            if isinstance(value, str):
                s_text: str = value.strip()
                if len(s_text) >= 2 and ((s_text[0] == "[" and s_text[-1] == "]") or (s_text[0] == "(" and s_text[-1] == ")")):
                    try:
                        s_json = "[" + s_text[1:-1] + "]" if (s_text[0] == "(" and s_text[-1] == ")") else s_text
                        parsed = json.loads(s_json)
                        return list(parsed) if isinstance(parsed, (list, tuple)) else []
                    except Exception:
                        return []
                return []
            # Fallback: treat scalars/others as empty
            try:
                if pd.isna(value):
                    return []
            except Exception:
                pass
            return []

        def _to_int_list(value: Any) -> List[int]:
            return [int(x) for x in _safe_to_list(value)]

        def _to_float_list(value: Any) -> List[float]:
            return [float(x) for x in _safe_to_list(value)]

        deb = _to_int_list(row.get("debit_accounts"))
        cre = _to_int_list(row.get("credit_accounts"))
        
        prev_acc, prev_side, tgt_acc, tgt_side, tgt_stop = build_targets_from_sets(deb, cre, self.max_lines)

        # Flow supervision (optional)
        debit_weights = _to_float_list(row.get("debit_amounts_norm"))
        credit_weights = _to_float_list(row.get("credit_amounts_norm"))

        features = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prev_account_idx": torch.tensor(prev_acc, dtype=torch.long),
            "prev_side_id": torch.tensor(prev_side, dtype=torch.long),
            "cond_numeric": torch.tensor(
                [
                    float(row.get("date_year", 0)),
                    float(row.get("date_month", 0)),
                    float(row.get("date_day", 0)),
                    float(row.get("date_dow", 0)),
                    float(row.get("date_month_sin", 0.0)),
                    float(row.get("date_month_cos", 0.0)),
                    float(row.get("date_day_sin", 0.0)),
                    float(row.get("date_day_cos", 0.0)),
                ],
                dtype=torch.float32,
            ),
            "currency": str(row.get("currency", "")),
            "journal_entry_type": str(row.get("journal_entry_type", "")),
        }
        targets = {
            "target_account_idx": torch.tensor(tgt_acc, dtype=torch.long),
            "target_side_id": torch.tensor(tgt_side, dtype=torch.long),
            "target_stop_id": torch.tensor(tgt_stop, dtype=torch.long),
            "debit_indices": torch.tensor(deb, dtype=torch.long) if deb else torch.tensor([], dtype=torch.long),
            "debit_weights": torch.tensor(debit_weights, dtype=torch.float32) if debit_weights else torch.tensor([], dtype=torch.float32),
            "credit_indices": torch.tensor(cre, dtype=torch.long) if cre else torch.tensor([], dtype=torch.long),
            "credit_weights": torch.tensor(credit_weights, dtype=torch.float32) if credit_weights else torch.tensor([], dtype=torch.float32),
        }
        return features, targets


def collate_fn(batch: List[Tuple[Dict[str, Any], Dict[str, Any]]]):
    feats, targs = zip(*batch)
    out_feats: Dict[str, Any] = {}
    out_targs: Dict[str, torch.Tensor] = {}

    def stack_tensor_field(key: str, dtype: torch.dtype):
        out_feats[key] = torch.stack([f[key] for f in feats]).to(dtype=dtype)

    stack_tensor_field("input_ids", torch.long)
    stack_tensor_field("attention_mask", torch.long)
    stack_tensor_field("prev_account_idx", torch.long)
    stack_tensor_field("prev_side_id", torch.long)
    out_feats["cond_numeric"] = torch.stack([f["cond_numeric"] for f in feats]).to(dtype=torch.float32)
    out_feats["currency"] = [f["currency"] for f in feats]
    out_feats["journal_entry_type"] = [f["journal_entry_type"] for f in feats]

    out_targs["target_account_idx"] = torch.stack([t["target_account_idx"] for t in targs]).to(dtype=torch.long)
    out_targs["target_side_id"] = torch.stack([t["target_side_id"] for t in targs]).to(dtype=torch.long)
    out_targs["target_stop_id"] = torch.stack([t["target_stop_id"] for t in targs]).to(dtype=torch.long)

    # Pad variable-length debit/credit lists to the max length in batch
    def pad_stack(key: str, fill: float, dtype: torch.dtype) -> torch.Tensor:
        lists = [t[key] for t in targs]
        max_len = max((x.numel() for x in lists), default=0)
        if max_len == 0:
            return torch.zeros((len(lists), 0), dtype=dtype)
        out = []
        for x in lists:
            if x.numel() < max_len:
                pad = torch.full((max_len - x.numel(),), fill_value=fill, dtype=dtype)
                out.append(torch.cat([x.to(dtype), pad], dim=0))
            else:
                out.append(x.to(dtype))
        return torch.stack(out)

    out_targs["debit_indices"] = pad_stack("debit_indices", -1, torch.long)
    out_targs["debit_weights"] = pad_stack("debit_weights", 0.0, torch.float32)
    out_targs["credit_indices"] = pad_stack("credit_indices", -1, torch.long)
    out_targs["credit_weights"] = pad_stack("credit_weights", 0.0, torch.float32)

    return out_feats, out_targs


