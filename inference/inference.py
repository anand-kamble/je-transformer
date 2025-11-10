from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

# Ensure KERAS_BACKEND default before importing model modules
if not os.environ.get("KERAS_BACKEND"):
	os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf

from data.text_normalization import normalize_description
from inference.beam_decode import beam_search_decode
from inference.postprocess import postprocess_candidates
from inference.retrieval_memory import build_retrieval_memory_for_text
from models.catalog_encoder import CatalogEncoder
from models.je_model import build_je_model
from models.tokenizer import DescriptionTokenizer


def load_json_from_uri(uri: str) -> Dict[str, Any]:
	if uri.startswith("gs://"):
		from google.cloud import storage
		client = storage.Client()
		_, path = uri.split("gs://", 1)
		bucket_name, blob_path = path.split("/", 1)
		blob = client.bucket(bucket_name).blob(blob_path)
		data = blob.download_as_bytes()
		return json.loads(data.decode("utf-8"))
	with tf.io.gfile.GFile(uri, "r") as f:
		return json.load(f)


def build_catalog_embeddings(artifact: Dict[str, Any], emb_dim: int) -> tf.Tensor:
	accounts = artifact["accounts"]
	number = tf.convert_to_tensor([a.get("number", "") for a in accounts], dtype=tf.string)
	name = tf.convert_to_tensor([a.get("name", "") for a in accounts], dtype=tf.string)
	nature = tf.convert_to_tensor([a.get("nature", "") for a in accounts], dtype=tf.string)
	enc = CatalogEncoder(emb_dim=emb_dim)
	return tf.convert_to_tensor(enc({"number": number, "name": name, "nature": nature}).numpy(), dtype=tf.float32)


def make_cond_numeric(
	year: Optional[int] = None,
	month: Optional[int] = None,
	day: Optional[int] = None,
	dow: Optional[int] = None,
	month_sin: Optional[float] = None,
	month_cos: Optional[float] = None,
	day_sin: Optional[float] = None,
	day_cos: Optional[float] = None,
) -> tf.Tensor:
	vals = [
		float(year or 0),
		float(month or 0),
		float(day or 0),
		float(dow or 0),
		float(month_sin or 0.0),
		float(month_cos or 0.0),
		float(day_sin or 0.0),
		float(day_cos or 0.0),
	]
	return tf.convert_to_tensor([vals], dtype=tf.float32)


def infer(
	description: str,
	accounts_artifact: str,
	encoder: str = "bert-base-multilingual-cased",
	hidden_dim: int = 256,
	max_length: int = 128,
	max_lines: int = 8,
	currency: str = "",
	journal_entry_type: str = "",
	date_features: Optional[Dict[str, float]] = None,
	beam_size: int = 20,
	alpha: float = 0.7,
	tau: float = 0.5,
	duplicate_policy: str = "collapse_unique_pairs",
	max_dup_per_account: Optional[int] = None,
	# Optional retrieval artifacts
	index_dir: Optional[str] = None,
	ids_uri: Optional[str] = None,
	embeddings_uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
	# Normalize + tokenize
	tok = DescriptionTokenizer(model_name_or_path=encoder, max_length=max_length)
	desc_norm = normalize_description(description)
	tb = tok.tokenize_batch([desc_norm])
	input_ids = tf.convert_to_tensor([tb["input_ids"][0]], dtype=tf.int32)
	attention_mask = tf.convert_to_tensor([tb["attention_mask"][0]], dtype=tf.int32)

	# Catalog
	artifact = load_json_from_uri(accounts_artifact)
	cat_emb = build_catalog_embeddings(artifact, emb_dim=hidden_dim)

	# Model
	model = build_je_model(
		encoder_loc=encoder,
		hidden_dim=hidden_dim,
		max_lines=max_lines,
		temperature=1.0,
	)

	# Conditioning
	date_feats = date_features or {}
	cond_numeric = make_cond_numeric(
		year=int(date_feats.get("year", 0) or 0),
		month=int(date_feats.get("month", 0) or 0),
		day=int(date_feats.get("day", 0) or 0),
		dow=int(date_feats.get("dow", 0) or 0),
		month_sin=float(date_feats.get("month_sin", 0.0) or 0.0),
		month_cos=float(date_feats.get("month_cos", 0.0) or 0.0),
		day_sin=float(date_feats.get("day_sin", 0.0) or 0.0),
		day_cos=float(date_feats.get("day_cos", 0.0) or 0.0),
	)
	cur = tf.convert_to_tensor([currency], dtype=tf.string)
	je_type = tf.convert_to_tensor([journal_entry_type], dtype=tf.string)

	# Retrieval memory (auto if artifacts provided)
	retr_mem = None
	if index_dir and ids_uri and embeddings_uri:
		retr_mem = build_retrieval_memory_for_text(
			description=desc_norm,
			index_dir=index_dir,
			ids_uri=ids_uri,
			embeddings_uri=embeddings_uri,
			tokenizer_loc=encoder,
			encoder_loc=encoder,
			max_length=max_length,
			use_cls=False,
			top_k=5,
		)

	# Decode
	candidates = beam_search_decode(
		model=model,
		input_ids=input_ids,
		attention_mask=attention_mask,
		catalog_embeddings=cat_emb,
		retrieval_memory=retr_mem,
		cond_numeric=cond_numeric,
		currency=cur,
		journal_entry_type=je_type,
		beam_size=beam_size,
		alpha=alpha,
		max_lines=max_lines,
		tau=tau,
		query_text=desc_norm if retr_mem is None and index_dir and ids_uri and embeddings_uri else None,
		index_dir=index_dir,
		ids_uri=ids_uri,
		embeddings_uri=embeddings_uri,
		tokenizer_loc=encoder,
		encoder_loc=encoder,
		top_k_retrieval=5,
	)

	# Postprocess
	filtered = postprocess_candidates(
		candidates,
		duplicate_policy=duplicate_policy,
		max_dup_per_account=max_dup_per_account,
		require_both_sides=True,
		min_lines=2,
	)
	return filtered


def main():
	parser = argparse.ArgumentParser(description="Inference wrapper for JE prediction.")
	parser.add_argument("--description", type=str, required=True)
	parser.add_argument("--accounts-artifact", type=str, required=True)
	parser.add_argument("--encoder", type=str, default="bert-base-multilingual-cased")
	parser.add_argument("--hidden-dim", type=int, default=256)
	parser.add_argument("--max-length", type=int, default=128)
	parser.add_argument("--max-lines", type=int, default=8)
	parser.add_argument("--currency", type=str, default="")
	parser.add_argument("--journal-entry-type", type=str, default="")
	parser.add_argument("--year", type=int, default=0)
	parser.add_argument("--month", type=int, default=0)
	parser.add_argument("--day", type=int, default=0)
	parser.add_argument("--dow", type=int, default=0)
	parser.add_argument("--month-sin", type=float, default=0.0)
	parser.add_argument("--month-cos", type=float, default=0.0)
	parser.add_argument("--day-sin", type=float, default=0.0)
	parser.add_argument("--day-cos", type=float, default=0.0)
	parser.add_argument("--beam-size", type=int, default=20)
	parser.add_argument("--alpha", type=float, default=0.7)
	parser.add_argument("--tau", type=float, default=0.5)
	parser.add_argument("--duplicate-policy", type=str, default="collapse_unique_pairs", choices=["allow", "collapse_unique_pairs", "limit_per_account"])
	parser.add_argument("--max-dup-per-account", type=int, default=None)
	# Optional retrieval artifacts
	parser.add_argument("--index-dir", type=str, default=None)
	parser.add_argument("--ids-uri", type=str, default=None)
	parser.add_argument("--embeddings-uri", type=str, default=None)
	args = parser.parse_args()

	date_feats = dict(
		year=args.year,
		month=args.month,
		day=args.day,
		dow=args.dow,
		month_sin=args.month_sin,
		month_cos=args.month_cos,
		day_sin=args.day_sin,
		day_cos=args.day_cos,
	)
	result = infer(
		description=args.description,
		accounts_artifact=args.accounts_artifact,
		encoder=args.encoder,
		hidden_dim=args.hidden_dim,
		max_length=args.max_length,
		max_lines=args.max_lines,
		currency=args.currency,
		journal_entry_type=args.journal_entry_type,
		date_features=date_feats,
		beam_size=args.beam_size,
		alpha=args.alpha,
		tau=args.tau,
		duplicate_policy=args.duplicate_policy,
		max_dup_per_account=args.max_dup_per_account,
		index_dir=args.index_dir,
		ids_uri=args.ids_uri,
		embeddings_uri=args.embeddings_uri,
	)
	print(json.dumps(result, indent=2))


if __name__ == "__main__":
	main()


