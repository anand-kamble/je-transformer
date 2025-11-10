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
from typing import Any, List

import pandas as pd
import pyarrow.parquet as pq
from google.cloud import storage
from transformers import (AutoModelForMaskedLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from data.text_normalization import normalize_description


def build_texts(parquet_pattern: str) -> List[str]:
    import glob
    paths = sorted(glob.glob(parquet_pattern))
    if not paths:
        raise ValueError(f"No Parquet files matched pattern: {parquet_pattern}")
    frames = [pq.read_table(p).to_pandas(columns=["description"]) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    texts = [normalize_description(t or "") for t in df["description"].tolist()]
    return texts


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


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        self.texts = texts
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


def main():
    parser = argparse.ArgumentParser(description="Domain-adaptive MLM pretraining on descriptions (PyTorch)")
    parser.add_argument("--parquet-pattern", type=str, required=True, help="Parquet glob, e.g., gs://bucket/prefix/parquet/*.parquet")
    parser.add_argument("--output-model-dir", type=str, required=True, help="GCS dir for saving the adapted model")
    parser.add_argument("--tokenizer-loc", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--mlm-prob", type=float, default=0.15)
    args = parser.parse_args()

    import torch

    texts = build_texts(args.parquet_pattern)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_loc, use_fast=True)
    ds = TextDataset(texts, tokenizer, args.max_length)

    model = AutoModelForMaskedLM.from_pretrained(args.tokenizer_loc)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob)

    with tempfile.TemporaryDirectory() as tmp_out:
        training_args = TrainingArguments(
            output_dir=tmp_out,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            logging_steps=50,
            save_steps=0,
            report_to=[],
        )
        trainer = Trainer(model=model, args=training_args, data_collator=collator, train_dataset=ds)
        trainer.train()
        model.save_pretrained(tmp_out)
        upload_dir_to_gcs(tmp_out, args.output_model_dir)


if __name__ == "__main__":
    main()

