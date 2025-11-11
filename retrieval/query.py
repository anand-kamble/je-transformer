#!/usr/bin/env python3
"""
Query a ScaNN index for nearest neighbor descriptions (PyTorch).
Given an input description, returns top-k similar journal_entry_ids.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from typing import List

import numpy as np
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
    def normalize_description(text: str) -> str:
        return (text or "").strip().lower()


def download_gcs_dir(gcs_dir: str, local_dir: str) -> None:
    if not gcs_dir.startswith("gs://"):
        raise ValueError("gcs_dir must start with gs://")
    client = storage.Client()
    _, path = gcs_dir.split("gs://", 1)
    bucket_name, prefix = path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix.rstrip("/")))
    for b in blobs:
        if b.name.endswith("/"):
            continue
        rel = os.path.relpath(b.name, start=prefix.rstrip("/"))
        lp = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        b.download_to_filename(lp)


def read_ids_from_gcs(gcs_uri: str) -> List[str]:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    client = storage.Client()
    _, path = gcs_uri.split("gs://", 1)
    bucket_name, blob_name = path.split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content = blob.download_as_text()
    return [line.strip() for line in content.splitlines() if line.strip()]


def embed_texts(texts: List[str], tokenizer_loc: str, encoder_loc: str, max_length: int, use_cls: bool) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_loc, use_fast=False)
    enc = AutoModel.from_pretrained(encoder_loc).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu") )
    enc.to(device)
    norm_texts = [normalize_description(t) for t in texts]
    batch = tokenizer(norm_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = enc(**batch)
        if use_cls:
            pooled = outputs.last_hidden_state[:, 0, :]
        else:
            mask = batch["attention_mask"].unsqueeze(-1).to(dtype=outputs.last_hidden_state.dtype)
            summed = (outputs.last_hidden_state * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp_min(1e-6)
            pooled = summed / denom
    vecs = pooled.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    return vecs / norms


def main():
    parser = argparse.ArgumentParser(description="Query ScaNN index for nearest descriptions (PyTorch)")
    parser.add_argument("--index-dir", type=str, required=True, help="GCS dir of ScaNN index artifacts")
    parser.add_argument("--ids-uri", type=str, required=True, help="GCS URI for ids list")
    parser.add_argument("--tokenizer-loc", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--encoder-loc", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--use-cls", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--text", type=str, required=True, help="Query description text")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        download_gcs_dir(args.index_dir, tmpdir)
        searcher = scann.scann_ops_pybind.load_searcher(tmpdir)
        ids = read_ids_from_gcs(args.ids_uri)

        q = embed_texts([args.text], args.tokenizer_loc, args.encoder_loc, args.max_length, args.use_cls)
        neighbors, distances = searcher.search_batched(q, final_num_neighbors=args.top_k)
        nbr_idx = np.array(neighbors[0]).tolist()
        for rank, idx in enumerate(nbr_idx, 1):
            print(f"{rank}\t{ids[idx]}\t{distances[0][rank-1]}")


if __name__ == "__main__":
    main()

