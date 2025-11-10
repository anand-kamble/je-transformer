#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false, reportGeneralTypeIssues=false
"""
Domain-adaptive pretraining (MLM) on journal entry descriptions.

Reads TFRecords produced by data/build_tfrecords.py, extracts `description` text,
tokenizes with a pretrained tokenizer (default: bert-base-multilingual-cased),
applies BERT-style random masking, and fine-tunes a TFBertForMaskedLM model.

Saves the adapted tokenizer (if different settings) and model artifacts to GCS.
"""
from __future__ import annotations

import argparse
import os
import tempfile
from typing import Any

# Ensure KERAS_BACKEND default before importing keras
if not os.environ.get("KERAS_BACKEND"):
	os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from google.cloud import storage
from transformers import AutoTokenizer, TFBertForMaskedLM

from data.text_normalization import normalize_description


def parse_example(serialized):
    features = {
        "description": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(serialized, features)
    return ex


def build_files_dataset(input_pattern, num_parallel_reads):
    files = tf.io.gfile.glob(input_pattern)
    if not files:
        raise ValueError(f"No TFRecord files matched pattern: {input_pattern}")
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=num_parallel_reads, compression_type=None)
    return ds


def py_normalize(x):
    s = x.numpy().decode("utf-8")
    s = normalize_description(s)
    return tf.convert_to_tensor(s.encode("utf-8"))


def make_tokenize_fn(tokenizer: Any, max_length: int):
    def _tok(x):
        # py_function returns tensors with dtype=string; we convert after
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


def _encode_bytes(tokenizer: Any, b, max_length: int):
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


def apply_mlm_masking(inputs, tokenizer: Any, mlm_prob: float = 0.15):
    """
    BERT-style masking:
      - choose tokens with probability p
      - 80% -> [MASK], 10% -> random token, 10% -> unchanged
    Labels are original ids where masked, and 0 elsewhere with sample_weight to gate loss.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    vocab_size = tokenizer.vocab_size
    mask_token_id = tokenizer.mask_token_id

    # candidate positions: exclude special tokens (CLS, SEP, PAD)
    special_ids = tf.constant(tokenizer.all_special_ids, dtype=tf.int32)
    is_special = tf.reduce_any(tf.equal(tf.expand_dims(input_ids, -1), special_ids), axis=-1)
    candidate = tf.logical_and(attention_mask > 0, tf.logical_not(is_special))

    # sample mask positions
    rnd = tf.random.uniform(tf.shape(input_ids), 0.0, 1.0)
    mask_positions = tf.logical_and(candidate, rnd < mlm_prob)

    labels = tf.where(mask_positions, input_ids, tf.zeros_like(input_ids))
    sample_weight = tf.cast(mask_positions, tf.float32)

    # 80% -> [MASK]
    rnd2 = tf.random.uniform(tf.shape(input_ids), 0.0, 1.0)
    to_mask = tf.logical_and(mask_positions, rnd2 < 0.8)
    masked_ids = tf.where(to_mask, tf.fill(tf.shape(input_ids), tf.cast(mask_token_id, tf.int32)), input_ids)

    # 10% -> random token (from vocab)
    to_random = tf.logical_and(mask_positions, tf.logical_and(rnd2 >= 0.8, rnd2 < 0.9))
    random_tokens = tf.random.uniform(tf.shape(input_ids), 0, vocab_size, dtype=tf.int32)
    masked_ids = tf.where(to_random, random_tokens, masked_ids)

    # 10% -> unchanged (already covered by leaving masked_ids as is where rnd2 >= 0.9)

    new_inputs = {"input_ids": masked_ids, "attention_mask": attention_mask}
    if "token_type_ids" in inputs:
        new_inputs["token_type_ids"] = inputs["token_type_ids"]
    return new_inputs, labels, sample_weight


def to_training_triplet(tokenizer: Any, mlm_prob: float):
    def _fn(inputs):
        return apply_mlm_masking(inputs, tokenizer, mlm_prob)

    return _fn


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


def main():
    parser = argparse.ArgumentParser(description="Domain-adaptive MLM pretraining on descriptions.")
    parser.add_argument("--input-pattern", type=str, required=True, help="TFRecord glob, e.g., gs://bucket/prefix/tfrecords/*.tfrecord")
    parser.add_argument("--output-model-dir", type=str, required=True, help="GCS dir for saving the adapted model")
    parser.add_argument("--tokenizer-loc", type=str, default="bert-base-multilingual-cased", help="HF model name or GCS dir with tokenizer artifacts")
    parser.add_argument("--output-tokenizer-dir", type=str, default=None, help="Optional GCS dir to save tokenizer (for reproducibility)")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--mlm-prob", type=float, default=0.15)
    parser.add_argument("--num-parallel-reads", type=int, default=tf.data.AUTOTUNE)
    args = parser.parse_args()

    # Build dataset
    raw = build_files_dataset(args.input_pattern, args.num_parallel_reads).map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    # Normalize text
    raw = raw.map(lambda ex: {"description": tf.py_function(py_normalize, [ex["description"]], tf.string)}, num_parallel_calls=tf.data.AUTOTUNE)

    # Tokenizer (supports GCS dir)
    if args.tokenizer_loc.startswith("gs://"):
        with tempfile.TemporaryDirectory() as tmp_tok:
            download_gcs_dir(args.tokenizer_loc, tmp_tok)
            tokenizer = AutoTokenizer.from_pretrained(tmp_tok, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_loc, use_fast=True)
    tok_fn = make_tokenize_fn(tokenizer, args.max_length)
    tokens = raw.map(lambda ex: tok_fn(ex["description"]), num_parallel_calls=tf.data.AUTOTUNE)

    # MLM masking to create (inputs, labels, sample_weight)
    triplets = tokens.map(lambda d: to_training_triplet(tokenizer, args.mlm_prob)(d), num_parallel_calls=tf.data.AUTOTUNE)
    triplets = triplets.shuffle(10000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Model
    model = TFBertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
    optimizer = keras.optimizers.Adam(learning_rate=3e-5)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="sum_over_batch_size")
    model.compile(optimizer=optimizer, loss=loss)

    # Keras expects (x, y, sample_weight); x and y are dict/tensor compatible shapes
    def split_xyw(batch):
        (inputs, labels, sw) = batch
        return inputs, labels, sw

    model.fit(triplets.map(split_xyw), epochs=args.epochs)

    # Save adapted model to GCS
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        upload_dir_to_gcs(tmpdir, args.output_model_dir)

    # Optionally save tokenizer artifacts to GCS
    if args.output_tokenizer_dir:
        with tempfile.TemporaryDirectory() as tmp_tok_out:
            tokenizer.save_pretrained(tmp_tok_out)
            upload_dir_to_gcs(tmp_tok_out, args.output_tokenizer_dir)


if __name__ == "__main__":
    main()

