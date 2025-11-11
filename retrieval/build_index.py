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

    print("[build_index] Args:")
    print(f"  parquet_pattern = {args.parquet_pattern}")
    print(f"  output_index_dir = {args.output_index_dir}")
    print(f"  encoder_loc = {args.encoder_loc}")
    print(f"  max_length = {args.max_length}, batch_size = {args.batch_size}, use_cls = {bool(args.use_cls)}, l2_normalize = {bool(args.l2_normalize)}")
    print(f"  scann: leaves={args.scann_leaves}, leaves_to_search={args.scann_leaves_to_search}, reorder={args.scann_reorder}")

    # GCS-only parquet expansion
    try:
        paths = _expand_gcs_glob(args.parquet_pattern)
    except Exception:
        paths = []
    if not paths:
        raise ValueError(f"No Parquet files matched pattern: {args.parquet_pattern}")
    print(f"[build_index] Matched {len(paths)} parquet files under pattern.")
    if len(paths) > 5:
        print("[build_index] First 5 files:")
        for p in paths[:5]:
            print(f"    {p}")
    else:
        print("[build_index] Files:")
        for p in paths:
            print(f"    {p}")

    # Load Parquet shards (select only the needed columns)
    frames = [pq.read_table(p, columns=["journal_entry_id", "description"]).to_pandas() for p in paths]
    df = pd.concat(frames, ignore_index=True)
    print(f"[build_index] Loaded DataFrame with {len(df)} rows")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_loc, use_fast=False)
    encoder = AutoModel.from_pretrained(args.encoder_loc).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    encoder.to(device)

    all_embs: List[np.ndarray] = []
    all_ids: List[str] = []

    # Batch iterate
    print("[build_index] Encoding to embeddings...")
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
    print(f"[build_index] Built embeddings list: {len(all_embs)} batches, ids={len(all_ids)}")

    embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, encoder.config.hidden_size), dtype=np.float32)
    print(f"[build_index] Embedding matrix shape: {embs.shape}, dtype={embs.dtype}")

    # Build ScaNN index
    builder = scann.scann_ops_pybind.builder(embs, 10, "dot_product")
    builder = builder.tree(num_leaves=args.scann_leaves, num_leaves_to_search=args.scann_leaves_to_search)
    # Call via getattr to avoid static analysis issues when scann stubs are unavailable
    builder = getattr(builder, "score_ah")(2, anisotropic_quantization_threshold=0.2)  # type: ignore[attr-defined]
    builder = builder.reorder(args.scann_reorder)
    print("[build_index] Building ScaNN index ...")
    searcher = builder.build()
    print("[build_index] ScaNN builder complete.")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Serialize ScaNN index artifacts into tmpdir
        searcher.serialize(tmpdir)
        print(f"[build_index] Serialized ScaNN index to tmpdir: {tmpdir}")
        try:
            print("[build_index] Serialized files and sizes:")
            for f in sorted(os.listdir(tmpdir)):
                fp = os.path.join(tmpdir, f)
                try:
                    sz = os.path.getsize(fp)
                except Exception:
                    sz = -1
                print(f"    {f}  ({sz} bytes)")
        except Exception as e:
            print(f"[build_index] Warning: failed to list tmpdir contents: {e}")
        # Rewrite asset paths to be relative (avoid absolute tmp paths baked into assets)
        try:
            assets_path = os.path.join(tmpdir, "scann_assets.pbtxt")
            if os.path.exists(assets_path):
                with open(assets_path, "r", encoding="utf-8") as af:
                    assets_txt = af.read()
                before = assets_txt
                # Replace any absolute tmpdir references with just the basename
                assets_txt = assets_txt.replace(tmpdir.rstrip("/") + "/", "")
                # Basic safety: ensure we no longer reference tmpdir
                if tmpdir in assets_txt:
                    print("[build_index] Warning: tmpdir path still present in scann_assets.pbtxt after rewrite.")
                if assets_txt != before:
                    with open(assets_path, "w", encoding="utf-8") as af:
                        af.write(assets_txt)
                    print("[build_index] Rewrote scann_assets.pbtxt to use relative paths.")
                else:
                    print("[build_index] scann_assets.pbtxt already uses relative paths.")
            else:
                print("[build_index] Warning: scann_assets.pbtxt not found; load may rely on defaults.")
        except Exception as e:
            print(f"[build_index] Warning: failed to rewrite scann_assets.pbtxt: {e}")
        # Always write aligned embeddings and ids alongside the index for simplicity
        ids_path = os.path.join(tmpdir, "ids.txt")
        with open(ids_path, "w", encoding="utf-8") as f:
            f.write("\n".join(all_ids))
        emb_path = os.path.join(tmpdir, "embeddings.npy")
        np.save(emb_path, embs)
        print("[build_index] Wrote ids.txt and embeddings.npy")
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
        print(f"[build_index] Wrote index_manifest.json: {json.dumps(manifest, indent=2)[:500]}{'...' if len(json.dumps(manifest))>500 else ''}")
        # Upload to GCS (root of index dir)
        print(f"[build_index] Uploading serialized artifacts to {args.output_index_dir} ...")
        _upload_dir_to_gcs(tmpdir, args.output_index_dir)
        try:
            # List uploaded files under GCS prefix for verification
            _, path = args.output_index_dir.split("gs://", 1)
            bucket_name, prefix = path.split("/", 1)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            names = [b.name for b in bucket.list_blobs(prefix=prefix.rstrip("/")) if not b.name.endswith("/")]
            print("[build_index] GCS uploaded objects:")
            for n in sorted(names):
                print(f"    gs://{bucket_name}/{n}")
        except Exception as e:
            print(f"[build_index] Warning: failed to list GCS objects after upload: {e}")


if __name__ == "__main__":
    main()

