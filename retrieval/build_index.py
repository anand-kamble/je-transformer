#!/usr/bin/env python3
"""
Build an ANN retrieval index (ScaNN) over description embeddings (PyTorch).

Pipeline:
  - Read Parquet shards produced by data/ingest_to_parquet.py
  - Extract description and journal_entry_id
  - Normalize + tokenize, encode to embeddings with a pretrained encoder (PyTorch)
  - L2-normalize embeddings (optional, default true)
  - Train and build ScaNN index
  - Upload index artifacts and ids list to GCS
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile
import fnmatch
from typing import Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scann
import torch
from google.cloud import storage
from transformers import AutoModel, AutoTokenizer

# Ensure project root is on sys.path for local imports when run from notebooks
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from data.text_normalization import normalize_description
except Exception:
    # Fallback: simple normalization if project import fails
    def normalize_description(text: str) -> str:
        return (text or "").strip().lower()


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def upload_dir_to_gcs(local_dir: str, gcs_dir: str) -> None:
    if not gcs_dir.startswith("gs://"):
        raise ValueError("gcs_dir must start with gs://")
    client = storage.Client()
    _, path = gcs_dir.split("gs://", 1)
    bucket_name, prefix = path.split("/", 1)
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for f in files:
            lp = os.path.join(root, f)
            # Flatten: upload all serialized files directly under the target prefix
            blob = bucket.blob(f"{prefix.rstrip('/')}/{os.path.basename(lp)}")
            blob.upload_from_filename(lp)


def write_ids_to_gcs(ids: List[str], gcs_uri: str) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    client = storage.Client()
    _, path = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_string("\n".join(ids), content_type="text/plain")


def main():
    parser = argparse.ArgumentParser(description="Build ScaNN index over description embeddings (Parquet → PyTorch)")
    parser.add_argument("--parquet-pattern", type=str, required=True, help="Parquet glob, e.g., gs://bucket/prefix/parquet/*.parquet")
    parser.add_argument("--output-index-dir", type=str, required=True, help="GCS dir for ScaNN index artifacts")
    parser.add_argument("--output-ids-uri", type=str, required=True, help="GCS URI for ids list (one journal_entry_id per line)")
    parser.add_argument("--output-embeddings-uri", type=str, default=None, help="Optional GCS URI to store raw embedding matrix (.npy)")
    parser.add_argument("--encoder-loc", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--use-cls", action="store_true")
    parser.add_argument("--l2-normalize", action="store_true")
    parser.add_argument("--scann-leaves", type=int, default=2000)
    parser.add_argument("--scann-leaves-to-search", type=int, default=100)
    parser.add_argument("--scann-reorder", type=int, default=250)
    args = parser.parse_args()

    import glob
    pattern = args.parquet_pattern
    if pattern.startswith("gs://") and ("*" in pattern or "?" in pattern):
        # Expand GCS wildcard manually
        try:
            from google.cloud import storage  # type: ignore
            _, path = pattern.split("gs://", 1)
            bucket_name, key_pattern = path.split("/", 1)
            prefix = key_pattern.rsplit("/", 1)[0]
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix.rstrip("/")))
            candidates = [f"gs://{bucket_name}/{b.name}" for b in blobs if b.name.endswith(".parquet")]
            paths = sorted([p for p in candidates if fnmatch.fnmatch(p, pattern)])
        except Exception:
            paths = []
    else:
        paths = sorted(glob.glob(pattern))
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
    searcher = scann.scann_ops_pybind.builder(embs, 10, "dot_product") \
        .tree(num_leaves=args.scann_leaves, num_leaves_to_search=args.scann_leaves_to_search) \
        .score_ah(2, anisotropic_quantization_threshold=0.2) \
        .reorder(args.scann_reorder) \
        .build()

    with tempfile.TemporaryDirectory() as tmpdir:
        searcher.serialize(tmpdir)
        upload_dir_to_gcs(tmpdir, args.output_index_dir)

    write_ids_to_gcs(all_ids, args.output_ids_uri)
    if args.output_embeddings_uri:
        if not args.output_embeddings_uri.startswith("gs://"):
            raise ValueError("--output-embeddings-uri must start with gs://")
        client = storage.Client()
        _, path = args.output_embeddings_uri.split("gs://", 1)
        bucket_name, blob_name = path.split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        buf = io.BytesIO()
        np.save(buf, embs)
        buf.seek(0)
        blob.upload_from_file(buf, content_type="application/octet-stream")


if __name__ == "__main__":
    main()

