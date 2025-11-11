#!/usr/bin/env python3
"""
Build an ANN retrieval index (ScaNN) over description embeddings (PyTorch).

GCS-only flow:
  - Read Parquet shards from a gs:// glob (produced by data/ingest_to_parquet.py)
  - Extract description and journal_entry_id
  - Normalize + tokenize, encode to embeddings with a pretrained encoder (PyTorch)
  - L2-normalize embeddings (optional)
  - Build ScaNN index
  - Write the following under --output-index-dir (gs://.../index):
      - serialized_partitioner.pb, scann_config.pb, scann_assets.pbtxt, ah_codebook.pb
      - embeddings.npy (float32), ids.txt (aligned to embeddings)
      - index_manifest.json (metadata for loader)
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import fnmatch
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scann
import torch
import json
from google.cloud import storage
from transformers import AutoModel, AutoTokenizer

# Ensure project root is on sys.path for local imports when run from notebooks
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from data.text_normalization import normalize_description
except Exception:
    def normalize_description(text: str) -> str:
        return (text or "").strip().lower()


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def _upload_dir_to_gcs(local_dir: str, gcs_dir: str) -> None:
    if not gcs_dir.startswith("gs://"):
        raise ValueError("output-index-dir must be a gs:// URI")
    client = storage.Client()
    _, path = gcs_dir.split("gs://", 1)
    bucket_name, prefix = path.split("/", 1)
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for f in files:
            lp = os.path.join(root, f)
            blob = bucket.blob(f"{prefix.rstrip('/')}/{os.path.basename(lp)}")
            blob.upload_from_filename(lp)


def _expand_gcs_glob(pattern: str) -> List[str]:
    if not (pattern.startswith("gs://") and ("*" in pattern or "?" in pattern)):
        raise ValueError("--parquet-pattern must be a gs:// glob")
    _, path = pattern.split("gs://", 1)
    bucket_name, key_pattern = path.split("/", 1)
    prefix = key_pattern.rsplit("/", 1)[0]
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix.rstrip("/")))
    candidates = [f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith(".parquet")]
    return sorted([p for p in candidates if fnmatch.fnmatch(p, pattern)])


def main():
    parser = argparse.ArgumentParser(description="Build ScaNN index over description embeddings (GCS-only)")
    parser.add_argument("--parquet-pattern", type=str, required=True, help="gs://.../parquet/*.parquet (GCS glob)")
    parser.add_argument("--output-index-dir", type=str, required=True, help="gs://.../index (GCS directory for index + artifacts)")
    parser.add_argument("--encoder-loc", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-cls", action="store_true")
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--scann-leaves", type=int, default=2000)
    parser.add_argument("--scann-leaves-to-search", type=int, default=100)
    parser.add_argument("--scann-reorder", type=int, default=250)
    args = parser.parse_args()

    # GCS-only parquet expansion
    try:
        paths = _expand_gcs_glob(args.parquet_pattern)
    except Exception:
        paths = []
    if not paths:
        raise ValueError(f"No Parquet files matched pattern: {args.parquet_pattern}")

    # Load Parquet shards (select only the needed columns)
    frames = [pq.read_table(p, columns=["journal_entry_id", "description"]).to_pandas() for p in paths]
    df = pd.concat(frames, ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_loc, use_fast=False)
    encoder = AutoModel.from_pretrained(args.encoder_loc).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    encoder.to(device)

    all_embs: List[np.ndarray] = []
    all_ids: List[str] = []

    # Batch iterate
    for start in range(0, len(df), args.batch_size):
        batch_df = df.iloc[start : start + args.batch_size]
        norm_texts = [normalize_description(t or "") for t in batch_df["description"].tolist()]
        toks = tokenizer(norm_texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        with torch.no_grad():
            outputs = encoder(**toks)
            if args.use_cls:
                pooled = outputs.last_hidden_state[:, 0, :]
            else:
                pooled = mean_pool(outputs.last_hidden_state, toks["attention_mask"])  # [B, H]
        embs_np = pooled.detach().cpu().numpy().astype(np.float32)
        if args.l2_normalize:
            norms = np.linalg.norm(embs_np, axis=1, keepdims=True) + 1e-8
            embs_np = embs_np / norms
        all_embs.append(embs_np)
        all_ids.extend([str(x) for x in batch_df["journal_entry_id"].tolist()])

    embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, encoder.config.hidden_size), dtype=np.float32)

    # Build ScaNN index
    builder = scann.scann_ops_pybind.builder(embs, 10, "dot_product")
    builder = builder.tree(num_leaves=args.scann_leaves, num_leaves_to_search=args.scann_leaves_to_search)
    # Call via getattr to avoid static analysis issues when scann stubs are unavailable
    builder = getattr(builder, "score_ah")(2, anisotropic_quantization_threshold=0.2)  # type: ignore[attr-defined]
    builder = builder.reorder(args.scann_reorder)
    searcher = builder.build()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Serialize ScaNN index artifacts into tmpdir
        searcher.serialize(tmpdir)
        # Always write aligned embeddings and ids alongside the index for simplicity
        ids_path = os.path.join(tmpdir, "ids.txt")
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_ids))
        emb_path = os.path.join(tmpdir, "embeddings.npy")
        np.save(emb_path, embs)
        # Write a small manifest for downstream loaders
        manifest = {
            "encoder_loc": args.encoder_loc,
            "pooling": "cls" if args.use_cls else "mean",
            "l2_normalized": bool(args.l2_normalize),
            "metric": "dot_product",
            "hidden_size": int(embs.shape[1]) if embs.size else int(encoder.config.hidden_size),
            "num_vectors": int(embs.shape[0]),
            "max_length": int(args.max_length),
            "scann": {
                "leaves": int(args.scann_leaves),
                "leaves_to_search": int(args.scann_leaves_to_search),
                "reorder": int(args.scann_reorder),
            },
            "files": {
                "ids": "ids.txt",
                "embeddings": "embeddings.npy",
            },
        }
        with open(os.path.join(tmpdir, "index_manifest.json"), "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2)
        # Upload to GCS (root of index dir)
        _upload_dir_to_gcs(tmpdir, args.output_index_dir)


if __name__ == "__main__":
    main()

