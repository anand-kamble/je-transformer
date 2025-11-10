from __future__ import annotations

import io
import os
from typing import List, Tuple

import numpy as np
import scann
import tensorflow as tf
from google.cloud import storage
from transformers import AutoTokenizer, TFAutoModel

from data.text_normalization import normalize_description


def _download_gcs_blob(gcs_uri: str) -> bytes:
	if not gcs_uri.startswith("gs://"):
		raise ValueError("gcs_uri must start with gs://")
	client = storage.Client()
	_, path = gcs_uri.split("gs://", 1)
	bucket_name, blob_name = path.split("/", 1)
	bucket = client.bucket(bucket_name)
	blob = bucket.blob(blob_name)
	return blob.download_as_bytes()


def load_embeddings_from_gcs(npy_uri: str) -> np.ndarray:
	data = _download_gcs_blob(npy_uri)
	buf = io.BytesIO(data)
	return np.load(buf)


def load_ids_from_gcs(ids_uri: str) -> List[str]:
	text = _download_gcs_blob(ids_uri).decode("utf-8")
	return [line.strip() for line in text.splitlines() if line.strip()]


def load_searcher_from_gcs(index_dir: str) -> scann.ScannSearcher:
	import tempfile
	with tempfile.TemporaryDirectory() as tmpdir:
		# Download all artifacts under index_dir to tmpdir
		if not index_dir.startswith("gs://"):
			raise ValueError("index_dir must start with gs://")
		client = storage.Client()
		_, path = index_dir.split("gs://", 1)
		bucket_name, prefix = path.split("/", 1)
		bucket = client.bucket(bucket_name)
		for blob in bucket.list_blobs(prefix=prefix.rstrip("/")):
			if blob.name.endswith("/"):
				continue
			local_path = os.path.join(tmpdir, os.path.relpath(blob.name, start=prefix.rstrip("/")))
			os.makedirs(os.path.dirname(local_path), exist_ok=True)
			blob.download_to_filename(local_path)
		# Load searcher
		searcher = scann.scann_ops_pybind.load_searcher(tmpdir)
	return searcher


def embed_text(text: str, tokenizer_loc: str, encoder_loc: str, max_length: int = 128, use_cls: bool = False) -> np.ndarray:
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_loc, use_fast=True)
	encoder = TFAutoModel.from_pretrained(encoder_loc)
	desc = normalize_description(text)
	tok = tokenizer([desc], padding=True, truncation=True, max_length=max_length, return_tensors="tf")
	outputs = encoder(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"], training=False)
	if use_cls:
		pooled = outputs.last_hidden_state[:, 0, :]
	else:
		mask = tf.cast(tf.expand_dims(tok["attention_mask"], -1), tf.float32)
		summed = tf.reduce_sum(outputs.last_hidden_state * mask, axis=1)
		denom = tf.reduce_sum(mask, axis=1) + 1e-6
		pooled = summed / denom
	vec = pooled.numpy().astype(np.float32)[0]
	# L2 normalize
	norm = np.linalg.norm(vec) + 1e-8
	return (vec / norm).astype(np.float32)


def build_retrieval_memory_for_text(
	description: str,
	index_dir: str,
	ids_uri: str,
	embeddings_uri: str,
	tokenizer_loc: str = "bert-base-multilingual-cased",
	encoder_loc: str = "bert-base-multilingual-cased",
	max_length: int = 128,
	use_cls: bool = False,
	top_k: int = 5,
) -> tf.Tensor:
	"""
	Returns retrieval memory tensor of shape [K, H_enc] built from the top-k neighbor embeddings.
	"""
	searcher = load_searcher_from_gcs(index_dir)
	ids = load_ids_from_gcs(ids_uri)
	all_embs = load_embeddings_from_gcs(embeddings_uri)  # [N, H_enc]
	q = embed_text(description, tokenizer_loc, encoder_loc, max_length=max_length, use_cls=use_cls)  # [H_enc]
	neighbors, _ = searcher.search(q, final_num_neighbors=top_k)
	idxs = neighbors.tolist()
	mem = all_embs[idxs] if len(idxs) > 0 else all_embs[:0]
	return tf.convert_to_tensor(mem, dtype=tf.float32)


