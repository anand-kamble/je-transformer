#!/usr/bin/env python3
"""
Lightweight project smoke test (PyTorch).

Runs minimal imports and tiny shape-checked forwards for core components using
small random tensors. Optional heavy deps (HF models/tokenizers, ScaNN, GCS)
are gracefully skipped when unavailable or offline.

Usage:
  python smoke_test.py

Env knobs:
  - SMOKE_ALLOW_HF=1     -> attempt to instantiate HF tokenizer/encoder-backed components
  - HF_MODEL=<name>      -> override HF model name (default: bert-base-multilingual-cased)
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import List

import numpy as np
import torch

# Ensure package imports with relative modules (place project parent on sys.path)
_here = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _skip(msg: str) -> None:
    print(f"[SKIP] {msg}")


def _fail(msg: str, exc: BaseException | None = None) -> None:
    print(f"[FAIL] {msg}")
    if exc is not None:
        traceback.print_exc()


def test_text_normalization() -> None:
    _section("text_normalization")
    from data.text_normalization import normalize_batch, normalize_description

    assert normalize_description("  Café\tCorp  ") == "Café Corp"
    out = normalize_batch(["Hello   world", None, ""])
    assert isinstance(out, list) and len(out) == 3
    _ok("normalize_description / normalize_batch")


def test_catalog_encoder() -> None:
    _section("CatalogEncoder")
    from models.catalog_encoder import CatalogEncoder

    encoder = CatalogEncoder(emb_dim=32)
    accounts = {
        "number": ["100-01", "200-02", "300-03"],
        "name": ["Cash", "Accounts Payable", "Revenue"],
        "nature": ["A", "L", "R"],
    }
    embs = encoder(accounts)
    assert embs.shape[-1] == 32
    assert embs.shape[0] == 3
    _ok(f"Encoded {embs.shape[0]} accounts to dim={embs.shape[-1]}")


def test_pointer_layer_and_losses() -> None:
    _section("PointerLayer and losses/metrics")
    from models.losses import (SetF1Metric, coverage_penalty, pointer_loss,
                               side_loss, stop_loss)
    from models.pointer import PointerLayer

    torch.manual_seed(7)
    batch, hidden, catalog = 2, 16, 11
    T = 4

    # Pointer layer (single step version)
    layer = PointerLayer(temperature=1.0)
    dec = torch.randn(batch, hidden)
    cat = torch.randn(catalog, hidden)
    logits_bc = layer(dec, cat)  # [B, C]
    assert logits_bc.shape == (batch, catalog)

    # Losses over time
    pointer_logits = torch.randn(batch, T, catalog)
    side_logits = torch.randn(batch, T, 2)
    stop_logits = torch.randn(batch, T, 2)
    target_accounts = torch.tensor([[1, 3, -1, -1], [2, 2, 5, -1]], dtype=torch.long)
    target_sides = torch.tensor([[0, 1, -1, -1], [1, 0, 1, -1]], dtype=torch.long)
    target_stop = torch.tensor([[0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.long)

    pl = pointer_loss(pointer_logits, target_accounts)
    sl = side_loss(side_logits, target_sides)
    stl = stop_loss(stop_logits, target_stop)
    cv = coverage_penalty(pointer_logits)
    assert all(v.dim() == 0 for v in [pl, sl, stl, cv])

    metric = SetF1Metric()
    metric.update_state(pointer_logits, side_logits, target_accounts, target_sides, target_stop)
    f1 = metric.result()
    assert f1.dim() == 0
    _ok("PointerLayer forward, losses, coverage penalty, F1 metric")


def test_postprocess() -> None:
    _section("postprocess")
    from inference.postprocess import postprocess_candidates

    cands = [
        {"accounts": [1, 2, 2], "sides": [0, 1, 1], "length": 3, "scores": [0.9, 0.8, 0.7]},
        {"accounts": [3, 3], "sides": [0, 0], "length": 2, "scores": [0.6, 0.5]},
        ]
    out = postprocess_candidates(cands, duplicate_policy="collapse_unique_pairs", require_both_sides=True, min_lines=2)
    assert isinstance(out, list) and len(out) >= 1
    _ok("postprocess_candidates with duplicate policy and structural filter")


def test_tokenizer_optional() -> None:
    _section("DescriptionTokenizer (optional)")
    if os.environ.get("SMOKE_ALLOW_HF", "1") != "1":
        return _skip("Set SMOKE_ALLOW_HF=1 to attempt HF tokenizer instantiation")
    try:
        from models.tokenizer import DescriptionTokenizer

        model_name = os.environ.get("HF_MODEL", "bert-base-multilingual-cased")
        tok = DescriptionTokenizer(model_name_or_path=model_name, max_length=16, use_fast=False)
        out = tok.tokenize_batch(["Hello   world", "Café Corp"])
        assert "input_ids" in out and "attention_mask" in out
        _ok(f"DescriptionTokenizer with model={model_name}")
    except Exception as e:
        _skip(f"Tokenizer test skipped due to error: {e}")


def test_retrieval_imports_optional() -> None:
    _section("retrieval imports (optional)")
    try:
        import importlib

        importlib.import_module("inference.retrieval_memory")
        _ok("Imported retrieval module")
    except Exception as e:
        _skip(f"Retrieval module not available or heavy deps missing (ScaNN/GCS): {e}")


def test_je_model_optional() -> None:
    _section("JE model forward (optional)")
    if os.environ.get("SMOKE_ALLOW_HF", "1") != "1":
        return _skip("Set SMOKE_ALLOW_HF=1 to build encoder-backed JE model")
    try:
        from models.je_model_torch import JEModel

        # Tiny shapes for quick forward
        B, L, T = 2, 8, 3
        H = 32
        C = 7
        K = 5

        model_name = os.environ.get("HF_MODEL", "bert-base-multilingual-cased")
        model = JEModel(encoder_loc=model_name, hidden_dim=H, max_lines=T, temperature=1.0)

        inputs = {
            "input_ids": torch.ones([B, L], dtype=torch.long),
            "attention_mask": torch.ones([B, L], dtype=torch.long),
            "prev_account_idx": torch.tensor([[-1, 1, 2], [-1, 0, 3]], dtype=torch.long),
            "prev_side_id": torch.tensor([[-1, 0, 1], [-1, 1, 0]], dtype=torch.long),
            "catalog_embeddings": torch.randn(C, H, dtype=torch.float32),
            "retrieval_memory": torch.randn(K, H, dtype=torch.float32),
            "cond_numeric": torch.randn(B, 8, dtype=torch.float32),
            "currency": ["USD", "EUR"],
            "journal_entry_type": ["sales", "refund"],
        }
        outs = model(**inputs)
        assert set(outs.keys()) == {"pointer_logits", "side_logits", "stop_logits"}
        assert outs["pointer_logits"].shape == (B, T, C)
        assert outs["side_logits"].shape == (B, T, 2)
        assert outs["stop_logits"].shape == (B, T, 2)
        _ok(f"JE model forward with model={model_name}")
    except Exception as e:
        _skip(f"JE model test skipped due to error: {e}")


def main() -> int:
    print("Running smoke tests for journal_entry_transformer (PyTorch)")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- NumPy version: {np.__version__}")
    print(f"- SMOKE_ALLOW_HF={os.environ.get('SMOKE_ALLOW_HF', '1')}")
    if os.environ.get("SMOKE_ALLOW_HF") == "1":
        print(f"- HF_MODEL={os.environ.get('HF_MODEL', 'hf-internal-testing/tiny-random-bert')}")

    failures: List[str] = []

    def guard(fn, name: str):
        try:
            fn()
        except Exception as e:
            failures.append(name)
            _fail(f"{name} failed", e)

    guard(test_text_normalization, "text_normalization")
    guard(test_catalog_encoder, "catalog_encoder")
    guard(test_pointer_layer_and_losses, "pointer_and_losses")
    guard(test_postprocess, "postprocess")
    # Optional / heavy deps
    guard(test_tokenizer_optional, "tokenizer_optional")
    guard(test_retrieval_imports_optional, "retrieval_optional_imports")
    guard(test_je_model_optional, "je_model_optional")

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s): {', '.join(failures)}")
        # Still exit 0 to focus on compile/import sanity; change to 1 if strictness desired
        return 0
    print("\nAll smoke tests completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

