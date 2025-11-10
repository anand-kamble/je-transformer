#!/usr/bin/env python3
"""
Build an ANN retrieval index (ScaNN) over description embeddings.

Pipeline:
  - Read TFRecords produced by data/build_tfrecords.py
  - Extract description and journal_entry_id
  - Normalize + tokenize, encode to embeddings with a pretrained (optionally adapted) encoder
  - L2-normalize embeddings (optional, default true)
  - Train and build ScaNN index
  - Upload index artifacts and ids list to GCS
"""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import scann
import tensorflow as tf
from google.cloud import storage
from transformers import AutoTokenizer, TFAutoModel

from data.text_normalization import normalize_description


def parse_example(serialized: tf.Tensor) -> Dict[str, tf.Tensor]:
    features = {
        "journal_entry_id": tf.io.FixedLenFeature([], tf.string),
        "description": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(serialized, features)
    return ex


def build_files_dataset(input_pattern: str, num_parallel_reads: int) -> tf.data.Dataset:
    files = tf.io.gfile.glob(input_pattern)
    if not files:
        raise ValueError(f"No TFRecord files matched pattern: {input_pattern}")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads, compression_type=None)
    return ds


def py_normalize(x: tf.Tensor) -> tf.Tensor:
    s = x.numpy().decode("utf-8")
    s = normalize_description(s)
    return tf.convert_to_tensor(s.encode("utf-8"))


def make_tokenize_fn(tokenizer: AutoTokenizer, max_length: int):
    def _tok(x: tf.Tensor) -> Dict[str, tf.Tensor]:
        out = tf.py_function(
            func=lambda b: _encode_bytes(tokenizer, b, max_length),
            inp=[x],
            Tout=[tf.int32, tf.int32],
        )
        input_ids, attention_mask = out
        input_ids.set_shape([max_length])
        attention_mask.set_shape([max_length])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    return _tok


def _encode_bytes(tokenizer: AutoTokenizer, b: tf.Tensor, max_length: int):
    text = b.numpy().decode("utf-8")
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    return tf.convert_to_tensor(enc["input_ids"], dtype=tf.int32), tf.convert_to_tensor(
        enc["attention_mask"], dtype=tf.int32
    )


def mean_pool(last_hidden_state: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
    # attention_mask: [B, L] -> float mask
    mask = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)  # [B, L, 1]
    summed = tf.reduce_sum(last_hidden_state * mask, axis=1)  # [B, H]
    denom = tf.reduce_sum(mask, axis=1) + 1e-6
    return summed / denom


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / norms


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
            rel = os.path.relpath(lp, start=local_dir)
            blob = bucket.blob(f"{prefix.rstrip('/')}/{rel}")
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
    parser = argparse.ArgumentParser(description="Build ScaNN index over description embeddings.")
    parser.add_argument("--input-pattern", type=str, required=True, help="TFRecord glob, e.g., gs://bucket/prefix/tfrecords/*.tfrecord")
    parser.add_argument("--output-index-dir", type=str, required=True, help="GCS dir for ScaNN index artifacts")
    parser.add_argument("--output-ids-uri", type=str, required=True, help="GCS URI for ids list (one journal_entry_id per line)")
    parser.add_argument("--output-embeddings-uri", type=str, default=None, help="Optional GCS URI to store raw embedding matrix (.npy)")
    parser.add_argument("--encoder-loc", type=str, default="bert-base-multilingual-cased", help="HF model name or local/GCS path (if downloaded)")
    parser.add_argument("--tokenizer-loc", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-parallel-reads", type=int, default=tf.data.AUTOTUNE)
    parser.add_argument("--use-cls", action="store_true", help="Use [CLS] token embedding instead of mean pooling")
    parser.add_argument("--l2-normalize", action="store_true", help="L2 normalize embeddings before indexing")
    parser.add_argument("--scann-leaves", type=int, default=2000)
    parser.add_argument("--scann-leaves-to-search", type=int, default=100)
    parser.add_argument("--scann-reorder", type=int, default=250)
    args = parser.parse_args()

    # Dataset
    raw = build_files_dataset(args.input_pattern, args.num_parallel_reads).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    raw = raw.map(
        lambda ex: {
            "journal_entry_id": ex["journal_entry_id"],
            "description": tf.py_function(lambda b: tf.convert_to_tensor(normalize_description(b.numpy().decode("utf-8")).encode("utf-8")), [ex["description"]], tf.string),
        },
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_loc, use_fast=True)
    tok_fn = make_tokenize_fn(tokenizer, args.max_length)
    tokenized = raw.map(lambda ex: {"journal_entry_id": ex["journal_entry_id"], **tok_fn(ex["description"])}, num_parallel_calls=tf.data.AUTOTUNE)
    tokenized = tokenized.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    encoder = TFAutoModel.from_pretrained(args.encoder_loc)

    all_embs: List[np.ndarray] = []
    all_ids: List[str] = []

    for batch in tokenized:
        outputs = encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            training=False,
        )
        if args.use_cls:
            pooled = outputs.last_hidden_state[:, 0, :]  # [B, H]
        else:
            pooled = mean_pool(outputs.last_hidden_state, batch["attention_mask"])  # [B, H]
        embs_np = pooled.numpy().astype(np.float32)
        ids_np = batch["journal_entry_id"].numpy().tolist()
        all_embs.append(embs_np)
        all_ids.extend([x.decode("utf-8") for x in ids_np])

    embs = np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, encoder.config.hidden_size), dtype=np.float32)
    if args.l2_normalize:
        embs = l2_normalize_np(embs)

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
        # Save embeddings matrix as .npy to GCS
        if not args.output_embeddings_uri.startswith("gs://"):
            raise ValueError("--output-embeddings-uri must start with gs://")
        client = storage.Client()
        _, path = args.output_embeddings_uri.split("gs://", 1)
        bucket_name, blob_name = path.split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        import io
        buf = io.BytesIO()
        np.save(buf, embs)
        buf.seek(0)
        blob.upload_from_file(buf, content_type="application/octet-stream")


if __name__ == "__main__":
    main()

