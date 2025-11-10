from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import keras
import numpy as np
import tensorflow as tf
from google.cloud import storage

from data.text_normalization import normalize_description
from models.catalog_encoder import CatalogEncoder
from models.je_model import build_je_model
from models.losses import (SetF1Hungarian, SetF1Metric, coverage_penalty,
                           flow_aux_loss, pointer_loss, side_loss, stop_loss)
from models.tokenizer import DescriptionTokenizer

AUTOTUNE = tf.data.AUTOTUNE


def load_json_from_uri(uri: str) -> Dict[str, Any]:
	"""
	Load JSON from local path or GCS (gs://).
	"""
	if uri.startswith("gs://"):
		client = storage.Client()
		bucket_name, blob_path = uri[5:].split("/", 1)
		b = client.bucket(bucket_name).blob(blob_path)
		data = b.download_as_bytes()
		return json.loads(data.decode("utf-8"))
	with open(uri, "r", encoding="utf-8") as f:
		return json.load(f)


def build_catalog_embeddings(artifact: Dict[str, Any], emb_dim: int) -> np.ndarray:
	"""
	Build catalog embeddings from accounts artifact using CatalogEncoder.
	Returns numpy array [C, emb_dim].
	"""
	accounts = artifact["accounts"]
	number = tf.convert_to_tensor([a.get("number", "") for a in accounts], dtype=tf.string)
	name = tf.convert_to_tensor([a.get("name", "") for a in accounts], dtype=tf.string)
	nature = tf.convert_to_tensor([a.get("nature", "") for a in accounts], dtype=tf.string)

	encoder = CatalogEncoder(emb_dim=emb_dim)
	embs = encoder({"number": number, "name": name, "nature": nature})  # [C, emb_dim]
	return embs.numpy()


def make_feature_spec() -> Dict[str, Any]:
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
		"debit_amounts_norm": tf.io.VarLenFeature(tf.float32),
		"credit_amounts_norm": tf.io.VarLenFeature(tf.float32),
	}


def build_targets_from_sets(debits: tf.Tensor, credits: tf.Tensor, max_lines: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
	"""
	Build sequence targets and teacher-forcing inputs from unordered sets.
	Order used: debits first, then credits (deterministic). Truncate/pad to max_lines.
	Returns:
	  prev_accounts [T], prev_sides [T], target_accounts [T], target_sides [T], stop [T]
	"""
	d_len = tf.shape(debits)[0]
	c_len = tf.shape(credits)[0]
	seq_acc = tf.concat([debits, credits], axis=0)  # [L]
	seq_side = tf.concat([tf.zeros((d_len,), tf.int32), tf.ones((c_len,), tf.int32)], axis=0)  # 0=Debit,1=Credit
	L = tf.shape(seq_acc)[0]
	Lc = tf.minimum(L, max_lines)

	seq_acc = seq_acc[:Lc]
	seq_side = seq_side[:Lc]

	# Pad to max_lines with -1 for accounts and sides
	pad_acc = tf.fill([max_lines - Lc], tf.constant(-1, tf.int32))
	pad_side = tf.fill([max_lines - Lc], tf.constant(-1, tf.int32))
	target_accounts = tf.concat([seq_acc, pad_acc], axis=0)  # [T]
	target_sides = tf.concat([seq_side, pad_side], axis=0)  # [T]

	# Teacher forcing prev arrays: shift right, BOS=-1
	prev_accounts = tf.concat([tf.constant([-1], tf.int32), target_accounts[:-1]], axis=0)
	prev_sides = tf.concat([tf.constant([-1], tf.int32), target_sides[:-1]], axis=0)

	# Stop target: 1 at last valid step, else 0, pad 0
	stop = tf.zeros((max_lines,), tf.int32)
	def set_stop():
		idx = tf.maximum(Lc - 1, 0)
		return tf.tensor_scatter_nd_update(stop, indices=tf.reshape(idx, [1, 1]), updates=tf.constant([1], tf.int32))
	stop = tf.cond(Lc > 0, set_stop, lambda: tf.tensor_scatter_nd_update(stop, [[0]], [1]))
	return prev_accounts, prev_sides, target_accounts, target_sides, stop


def build_dataset(pattern: str, tokenizer: DescriptionTokenizer, max_length: int, max_lines: int, batch_size: int) -> tf.data.Dataset:
	spec = make_feature_spec()

	def _parse(rec):
		ex = tf.io.parse_single_example(rec, spec)
		# Densify varlen
		ex["debit_accounts"] = tf.cast(tf.sparse.to_dense(ex["debit_accounts"], default_value=-1), tf.int32)
		ex["credit_accounts"] = tf.cast(tf.sparse.to_dense(ex["credit_accounts"], default_value=-1), tf.int32)
		ex["debit_amounts_norm"] = tf.sparse.to_dense(ex["debit_amounts_norm"], default_value=0.0)
		ex["credit_amounts_norm"] = tf.sparse.to_dense(ex["credit_amounts_norm"], default_value=0.0)
		return ex

	def _map_py(ex):
		# Normalize + tokenize
		desc = normalize_description(ex["description"].numpy().decode("utf-8"))
		tok = tokenizer.tokenize_batch([desc])
		input_ids = tok["input_ids"][0]
		attention_mask = tok["attention_mask"][0]
		return (
			np.asarray(input_ids, dtype=np.int32),
			np.asarray(attention_mask, dtype=np.int32),
		)

	def _map(ex):
		# Tokenize via py_function
		input_ids, attention_mask = tf.py_function(
			func=_map_py,
			inp=[ex],
			Tout=(tf.int32, tf.int32),
		)
		input_ids.set_shape([max_length])
		attention_mask.set_shape([max_length])

		# Build conditioning numeric vector [8]: year, month, day, dow, month_sin, month_cos, day_sin, day_cos
		cond_numeric = tf.stack([
			tf.cast(ex["date_year"], tf.float32),
			tf.cast(ex["date_month"], tf.float32),
			tf.cast(ex["date_day"], tf.float32),
			tf.cast(ex["date_dow"], tf.float32),
			ex["date_month_sin"],
			ex["date_month_cos"],
			ex["date_day_sin"],
			ex["date_day_cos"],
		], axis=0)

		prev_acc, prev_side, tgt_acc, tgt_side, tgt_stop = build_targets_from_sets(
			ex["debit_accounts"], ex["credit_accounts"], max_lines=max_lines
		)
		features = {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"prev_account_idx": prev_acc,
			"prev_side_id": prev_side,
			"cond_numeric": cond_numeric,
			"currency": ex["currency"],
			"journal_entry_type": ex["journal_entry_type"],
		}
		targets = {
			"target_account_idx": tgt_acc,
			"target_side_id": tgt_side,
			"target_stop_id": tgt_stop,
			# Flow supervision targets
			"debit_indices": ex["debit_accounts"],
			"debit_weights": ex["debit_amounts_norm"],
			"credit_indices": ex["credit_accounts"],
			"credit_weights": ex["credit_amounts_norm"],
		}
		return features, targets

	ds = tf.data.TFRecordDataset(tf.io.gfile.glob(pattern), num_parallel_reads=AUTOTUNE)
	ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)
	ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
	ds = ds.shuffle(4096).batch(batch_size).prefetch(AUTOTUNE)
	return ds


def main():
	parser = argparse.ArgumentParser(description="Train JE pointer model.")
	parser.add_argument("--input-pattern", type=str, required=True, help="gs:// or local glob of TFRecords")
	parser.add_argument("--accounts-artifact", type=str, required=True, help="gs://.../artifacts/accounts_*.json matching the TFRecords catalog")
	parser.add_argument("--output-dir", type=str, required=True, help="gs:// or local path to save checkpoints")
	parser.add_argument("--encoder", type=str, default="bert-base-multilingual-cased")
	parser.add_argument("--max-length", type=int, default=128)
	parser.add_argument("--max-lines", type=int, default=8)
	parser.add_argument("--hidden-dim", type=int, default=256)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=1)
	args = parser.parse_args()

	tokenizer = DescriptionTokenizer(model_name_or_path=args.encoder, max_length=args.max_length)

	# Dataset
	ds = build_dataset(args.input_pattern, tokenizer, args.max_length, args.max_lines, args.batch_size)

	# Catalog embeddings
	artifact = load_json_from_uri(args.accounts_artifact)
	cat_emb_np = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim)  # [C, H]
	cat_emb = tf.convert_to_tensor(cat_emb_np, dtype=tf.float32)

	# Retrieval memory placeholder (zeros; can be replaced by actual retrieval contexts)
	retr_mem = tf.zeros((1, args.hidden_dim), dtype=tf.float32)  # [K=1, H]

	# Model
	model = build_je_model(
		encoder_loc=args.encoder,
		hidden_dim=args.hidden_dim,
		max_lines=args.max_lines,
		temperature=1.0,
	)

	optimizer = keras.optimizers.Adam(learning_rate=2e-4)
	metric_set_f1 = SetF1Hungarian()

	@tf.function
	def train_step(batch_features, batch_targets):
		with tf.GradientTape() as tape:
			outputs = model(
				{
					"input_ids": batch_features["input_ids"],
					"attention_mask": batch_features["attention_mask"],
					"prev_account_idx": batch_features["prev_account_idx"],
					"prev_side_id": batch_features["prev_side_id"],
					"catalog_embeddings": cat_emb,
					"retrieval_memory": retr_mem,
					"cond_numeric": batch_features["cond_numeric"],
					"currency": batch_features["currency"],
					"journal_entry_type": batch_features["journal_entry_type"],
				},
				training=True,
			)
			pl = pointer_loss(outputs["pointer_logits"], batch_targets["target_account_idx"], ignore_index=-1)
			sl = side_loss(outputs["side_logits"], batch_targets["target_side_id"], ignore_index=-1)
			stl = stop_loss(outputs["stop_logits"], batch_targets["target_stop_id"], ignore_index=-1)
			cov = coverage_penalty(outputs["pointer_logits"])
			flow = flow_aux_loss(
				outputs["pointer_logits"],
				outputs["side_logits"],
				batch_targets["debit_indices"],
				batch_targets["debit_weights"],
				batch_targets["credit_indices"],
				batch_targets["credit_weights"],
			)
			total = pl + sl + stl + 0.01 * cov + 0.1 * flow
		grads = tape.gradient(total, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		metric_set_f1.update_state(
			outputs["pointer_logits"], outputs["side_logits"],
			batch_targets["target_account_idx"], batch_targets["target_side_id"],
			batch_targets["target_stop_id"],
		)
		return {"loss": total, "pointer_loss": pl, "side_loss": sl, "stop_loss": stl, "cov_pen": cov, "flow_loss": flow}

	# Training loop
	global_step = 0
	for epoch in range(args.epochs):
		for features, targets in ds:
			stats = train_step(features, targets)
			global_step += 1
			if global_step % 50 == 0:
				tf.print(
					"step", global_step,
					"loss", stats["loss"],
					"ptr", stats["pointer_loss"],
					"side", stats["side_loss"],
					"stop", stats["stop_loss"],
					"cov", stats["cov_pen"],
					"setF1", metric_set_f1.result(),
				)

	# Save model
	if args.output_dir.startswith("gs://"):
		tmp_dir = "/tmp/je_model"
		tf.saved_model.save(model, tmp_dir)
		client = storage.Client()
		for root, _, files in os.walk(tmp_dir):
			for f in files:
				local_path = os.path.join(root, f)
				rel = os.path.relpath(local_path, tmp_dir)
				bucket_name, blob_path = args.output_dir[5:].split("/", 1)
				blob = client.bucket(bucket_name).blob(os.path.join(blob_path, rel))
				blob.upload_from_filename(local_path)
	else:
		tf.saved_model.save(model, args.output_dir)


