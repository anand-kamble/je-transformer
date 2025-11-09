from __future__ import annotations

import os
import tempfile
from typing import List

from ..data.text_normalization import normalize_batch


class DescriptionTokenizer:
    """
    Wrapper around a pretrained HuggingFace tokenizer (default: bert-base-multilingual-cased)
    with light normalization suitable for accounting descriptions.
    """

    def __init__(
        self,
        model_name_or_path: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        use_fast: bool = True,
    ) -> None:
        from transformers import \
            AutoTokenizer  # defer import to avoid tooling stub issues

        self.model_name_or_path = model_name_or_path
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)

    def tokenize_batch(self, texts: List[str]):
        norm_texts = normalize_batch(texts)
        encoded = self.tokenizer(
            norm_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        out = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "token_type_ids" in encoded:
            out["token_type_ids"] = encoded["token_type_ids"]
        return out

    def save_to_gcs(self, gcs_dir: str) -> str:
        """
        Saves tokenizer artifacts to a GCS directory (e.g., gs://bucket/path/tokenizer).
        Returns the GCS directory used.
        """
        if not gcs_dir.startswith("gs://"):
            raise ValueError("gcs_dir must start with gs://")

        from google.cloud import \
            storage  # local import to minimize global deps

        with tempfile.TemporaryDirectory() as tmpdir:
            self.tokenizer.save_pretrained(tmpdir)
            client = storage.Client()
            _, path = gcs_dir.split("gs://", 1)
            bucket_name, prefix = path.split("/", 1)
            bucket = client.bucket(bucket_name)
            for filename in os.listdir(tmpdir):
                local_path = os.path.join(tmpdir, filename)
                blob = bucket.blob(f"{prefix.rstrip('/')}/{filename}")
                blob.upload_from_filename(local_path)
        return gcs_dir

    @classmethod
    def from_gcs_or_model(cls, location: str, max_length: int = 128) -> "DescriptionTokenizer":
        """
        If `location` is a GCS dir, download to temp and load; otherwise treat as HF model name/path.
        """
        if location.startswith("gs://"):
            from google.cloud import storage  # local import
            from transformers import AutoTokenizer  # local import

            client = storage.Client()
            _, path = location.split("gs://", 1)
            bucket_name, prefix = path.split("/", 1)
            bucket = client.bucket(bucket_name)
            blobs = list(bucket.list_blobs(prefix=prefix.rstrip("/")))
            if not blobs:
                raise ValueError(f"No artifacts found under {location}")
            tmpdir = tempfile.mkdtemp()
            try:
                for b in blobs:
                    if b.name.endswith("/"):
                        continue
                    local_name = os.path.basename(b.name)
                    local_path = os.path.join(tmpdir, local_name)
                    b.download_to_filename(local_path)
                tok = AutoTokenizer.from_pretrained(tmpdir, use_fast=True)
                obj = cls(model_name_or_path=tmpdir, max_length=max_length)
                obj.tokenizer = tok
                return obj
            finally:
                # Leave tmpdir until process end; cleanup handled by OS/container lifecycle.
                pass
        else:
            return cls(model_name_or_path=location, max_length=max_length)

