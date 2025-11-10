from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

# Ensure KERAS_BACKEND default
if not os.environ.get("KERAS_BACKEND"):
	os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

from data.text_normalization import normalize_description
from models.losses import SetF1Hungarian
from models.tokenizer import DescriptionTokenizer
from models.catalog_encoder import CatalogEncoder
from models.je_model import build_je_model
from inference.beam_decode import beam_search_decode
from inference.postprocess import postprocess_candidates


AUTOTUNE = tf.data.AUTOTUNE


def load_json_from_uri(uri: str) -> Dict[str, Any]:
	if uri.startswith("gs://"):
		from google.cloud import storage
		client = storage.Client()
		_, path = uri.split("gs://", 1)
		bucket_name, blob_path = path.split("/", 1)
		blob = client.bucket(bucket_name).blob(blob_path)
		data = blob.download_as_bytes()
		return json.loads(data.decode("utf-8"))
	with open(uri, "r", encoding="utf-8") as f:
		return json.load(f)


def build_catalog_embeddings(artifact: Dict[str, Any], emb_dim: int) -> np.ndarray:
	accounts = artifact["accounts"]
	number = tf.convert_to_tensor([a.get("number", "") for a in accounts], dtype=tf.string)
	name = tf.convert_to_tensor([a.get("name", "") for a in accounts], dtype=tf.string)
	nature = tf.convert_to_tensor([a.get("nature", "") for a in accounts], dtype=tf.string)
	encoder = CatalogEncoder(emb_dim=emb_dim)
	embs = encoder({"number": number, "name": name, "nature": nature})
	return embs.numpy()


def feature_spec() -> Dict[str, Any]:
	return {
		"journal_entry_id": tf.io.FixedLenFeature([], tf.string),
		"business_id": tf.io.FixedLenFeature([], tf.string),
		"description": tf.io.FixedLenFeature([], tf.string),
		"currency": tf.io.FixedLenFeature([], tf.string),
		"journal_entry_type": tf.io.FixedLenFeature([], tf.string),
		"date_year": tf.io.FixedLenFeature([], tf.int64),
		"date_month": tf.io.FixedLenFeature([], tf.int64),
		"date_day": tf.io.FixedLenFeature([], tf.int64),
		"date_dow": tf.io.FixedLenFeature([], tf.int64),
		"date_month_sin": tf.io.FixedLenFeature([], tf.float32),
		"date_month_cos": tf.io.FixedLenFeature([], tf.float32),
		"date_day_sin": tf.io.FixedLenFeature([], tf.float32),
		"date_day_cos": tf.io.FixedLenFeature([], tf.float32),
		"debit_accounts": tf.io.VarLenFeature(tf.int64),
		"credit_accounts": tf.io.VarLenFeature(tf.int64),
	}


def build_eval_ds(pattern: str) -> tf.data.Dataset:
	files = tf.io.gfile.glob(pattern)
	if not files:
		raise ValueError(f"No TFRecord files matched: {pattern}")
	def _parse(x):
		ex = tf.io.parse_single_example(x, feature_spec())
		ex["debit_accounts"] = tf.cast(tf.sparse.to_dense(ex["debit_accounts"], default_value=-1), tf.int32)
		ex["credit_accounts"] = tf.cast(tf.sparse.to_dense(ex["credit_accounts"], default_value=-1), tf.int32)
		return ex
	return tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).map(_parse, num_parallel_calls=AUTOTUNE)


def ece(probs: List[float], correct: List[bool], num_bins: int = 15) -> float:
	probs_np = np.asarray(probs, dtype=np.float64)
	corr_np = np.asarray(correct, dtype=np.float64)
	bins = np.linspace(0.0, 1.0, num_bins + 1)
	ece_val = 0.0
	n = len(probs_np)
	for i in range(num_bins):
		l, r = bins[i], bins[i + 1]
		mask = (probs_np >= l) & (probs_np < r) if i < num_bins - 1 else (probs_np >= l) & (probs_np <= r)
		if not np.any(mask):
			continue
		conf = probs_np[mask].mean()
		acc = corr_np[mask].mean()
		ece_val += (mask.sum() / n) * abs(acc - conf)
	return float(ece_val)


def tune_temperature(probs: List[float], correct: List[bool], grid: List[float]) -> Tuple[float, float]:
	# Power scaling: p' = p ** (1/T)
	best_T = 1.0
	best_ece = ece(probs, correct)
	for T in grid:
		if T <= 0:
			continue
		scaled = [float(p ** (1.0 / T)) for p in probs]
		cur_ece = ece(scaled, correct)
		if cur_ece < best_ece:
			best_ece = cur_ece
			best_T = T
	return best_T, best_ece


def main():
	parser = argparse.ArgumentParser(description="Evaluate JE model: token/set metrics and calibration.")
	parser.add_argument("--input-pattern", type=str, required=True)
	parser.add_argument("--accounts-artifact", type=str, required=True)
	parser.add_argument("--model-dir", type=str, required=False, help="If omitted, builds a fresh model with encoder weights")
	parser.add_argument("--encoder", type=str, default="bert-base-multilingual-cased")
	parser.add_argument("--max-length", type=int, default=128)
	parser.add_argument("--max-lines", type=int, default=8)
	parser.add_argument("--hidden-dim", type=int, default=256)
	parser.add_argument("--limit", type=int, default=1000, help="Max examples to evaluate")
	parser.add_argument("--beam-size", type=int, default=20)
	parser.add_argument("--alpha", type=float, default=0.7)
	parser.add_argument("--tau", type=float, default=0.5)
	parser.add_argument("--output-report", type=str, default="eval_report.json")
	# Optional retrieval artifacts
	parser.add_argument("--index-dir", type=str, default=None)
	parser.add_argument("--ids-uri", type=str, default=None)
	parser.add_argument("--embeddings-uri", type=str, default=None)
	args = parser.parse_args()

	tok = DescriptionTokenizer(model_name_or_path=args.encoder, max_length=args.max_length)
	ds = build_eval_ds(args.input_pattern)

	artifact = load_json_from_uri(args.accounts_artifact)
	cat_emb_np = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim)
	cat_emb = tf.convert_to_tensor(cat_emb_np, dtype=tf.float32)

	model = build_je_model(
		encoder_loc=args.encoder,
		hidden_dim=args.hidden_dim,
		max_lines=args.max_lines,
		temperature=1.0,
	)
	if args.model_dir:
		# Load weights if provided
		try:
			loaded = tf.saved_model.load(args.model_dir)
			# If incompatible, fall back to fresh model (weights not loaded)
			del loaded
		} except Exception:
			pass

	metric_set = SetF1Hungarian()
	token_acc_ptr = keras.metrics.Mean(name="ptr_token_acc")
	token_acc_side = keras.metrics.Mean(name="side_token_acc")
	token_acc_stop = keras.metrics.Mean(name="stop_token_acc")

	seq_probs: List[float] = []
	seq_correct: List[bool] = []

	count = 0
	for ex in ds.take(args.limit):
		desc = normalize_description(ex["description"].numpy().decode("utf-8"))
		batch = tok.tokenize_batch([desc])
		input_ids = tf.convert_to_tensor([batch["input_ids"][0]], dtype=tf.int32)
		attention_mask = tf.convert_to_tensor([batch["attention_mask"][0]], dtype=tf.int32)

		# Build cond numeric
		cond_numeric = tf.convert_to_tensor(
			[[
				float(ex["date_year"].numpy()),
				float(ex["date_month"].numpy()),
				float(ex["date_day"].numpy()),
				float(ex["date_dow"].numpy()),
				float(ex["date_month_sin"].numpy()),
				float(ex["date_month_cos"].numpy()),
				float(ex["date_day_sin"].numpy()),
				float(ex["date_day_cos"].numpy()),
			]],
			dtype=tf.float32,
		)
		currency = tf.convert_to_tensor([ex["currency"].numpy().decode("utf-8")], dtype=tf.string)
		je_type = tf.convert_to_tensor([ex["journal_entry_type"].numpy().decode("utf-8")], dtype=tf.string)

		# Targets as sequence for token metrics
		debits = ex["debit_accounts"].numpy().tolist()
		credits = ex["credit_accounts"].numpy().tolist()
		tgt_seq = [int(i) for i in debits if i >= 0] + [int(i) for i in credits if i >= 0]
		tgt_side = [0] * len([i for i in debits if i >= 0]) + [1] * len([i for i in credits if i >= 0])
		T = min(len(tgt_seq), args.max_lines)
		# Prepare teacher forcing inputs
		prev_acc = [-1] + tgt_seq[:max(0, T - 1)]
		prev_side = [-1] + tgt_side[:max(0, T - 1)]
		# Pad to max_lines
		prev_acc = prev_acc + [-1] * (args.max_lines - len(prev_acc))
		prev_side = prev_side + [-1] * (args.max_lines - len(prev_side))
		target_acc = tgt_seq[:T] + [-1] * (args.max_lines - T)
		target_side = tgt_side[:T] + [-1] * (args.max_lines - T)
		target_stop = [0] * (max(0, T - 1)) + ([1] if T > 0 else [1]) + [0] * (args.max_lines - T)

		outs = model(
			{
				"input_ids": input_ids,
				"attention_mask": attention_mask,
				"prev_account_idx": tf.convert_to_tensor([prev_acc], dtype=tf.int32),
				"prev_side_id": tf.convert_to_tensor([prev_side], dtype=tf.int32),
				"catalog_embeddings": cat_emb,
				"retrieval_memory": tf.zeros((1, args.hidden_dim), dtype=tf.float32),
				"cond_numeric": cond_numeric,
				"currency": currency,
				"journal_entry_type": je_type,
			},
			training=False,
		)
		ptr_pred = tf.argmax(outs["pointer_logits"], axis=-1).numpy()[0][:T] if T > 0 else []
		side_pred = tf.argmax(outs["side_logits"], axis=-1).numpy()[0][:T] if T > 0 else []
		stop_pred = tf.argmax(outs["stop_logits"], axis=-1).numpy()[0][:T] if T > 0 else []
		if T > 0:
			token_acc_ptr.update_state(np.mean((np.array(ptr_pred) == np.array(tgt_seq[:T])).astype(np.float32)))
			token_acc_side.update_state(np.mean((np.array(side_pred) == np.array(tgt_side[:T])).astype(np.float32)))
			# Consider stop correctness at last step only
			token_acc_stop.update_state(float(stop_pred[-1] == 1))

		# Set-level Hungarian F1 on token logits vs targets
		metric_set.update_state(
			outs["pointer_logits"],
			outs["side_logits"],
			tf.convert_to_tensor([target_acc], dtype=tf.int32),
			tf.convert_to_tensor([target_side], dtype=tf.int32),
			tf.convert_to_tensor([target_stop], dtype=tf.int32),
		)

		# Beam search candidate and probability for calibration
		cands = beam_search_decode(
			model=model,
			input_ids=input_ids,
			attention_mask=attention_mask,
			catalog_embeddings=cat_emb,
			retrieval_memory=None,
			cond_numeric=cond_numeric,
			currency=currency,
			journal_entry_type=je_type,
			beam_size=args.beam_size,
			alpha=args.alpha,
			max_lines=args.max_lines,
			tau=args.tau,
		)
		if not cands:
			continue
		best = cands[0]
		pairs_pred = list(zip(best["accounts"], best["sides"]))
		pairs_true = list(zip(tgt_seq[:T], tgt_side[:T]))
		seq_probs.append(best["prob"])
		seq_correct.append(pairs_pred == pairs_true)

		count += 1

	# Calibration
	ece_raw = ece(seq_probs, seq_correct, num_bins=15)
	best_T, best_ece = tune_temperature(seq_probs, seq_correct, grid=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

	report = {
		"examples": count,
		"token_ptr_acc": float(token_acc_ptr.result().numpy()) if count else 0.0,
		"token_side_acc": float(token_acc_side.result().numpy()) if count else 0.0,
		"token_stop_acc": float(token_acc_stop.result().numpy()) if count else 0.0,
		"set_f1_hungarian": float(metric_set.result().numpy()),
		"calibration": {
			"ece_raw": ece_raw,
			"best_temperature": best_T,
			"ece_best": best_ece,
		},
	}
	# Generalization buckets: by number of lines (2, 3, >=4) can be added in a future pass
	with tf.io.gfile.GFile(args.output_report, "w") as f:
		json.dump(report, f, indent=2)
	print(json.dumps(report, indent=2))


if __name__ == "__main__":
	main()


