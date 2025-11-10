from __future__ import annotations

import os
from typing import Any, Dict

# Ensure Keras uses TensorFlow backend by default
if not os.environ.get("KERAS_BACKEND"):
	os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import sqlalchemy as sa
import tensorflow as tf
from google.cloud import storage

from data.build_tfrecords import (AccountCatalog, create_engine_with_connector,
                                  ensure_gcs_client, reflect_or_define_tables,
                                  write_examples_to_gcs, write_gcs_json)
from models.je_model import build_je_model
from models.losses import (SetF1Hungarian, coverage_penalty, flow_aux_loss,
                           pointer_loss, side_loss, stop_loss)
from models.tokenizer import DescriptionTokenizer
from train.train import (build_catalog_embeddings, build_dataset,
                         load_json_from_uri)

# =========================
# Configuration constants
# =========================

# GCS bucket used across ingestion, training data, and model export
YOUR_BUCKET: str = "YOUR_BUCKET_NAME"

# TFRecords to train on (glob). Example: "gs://your-bucket/je_data/tfrecords/*.tfrecord"
INPUT_PATTERN: str = ""  # If empty, will be derived from ingestion GCS_OUTPUT_URI/tfrecords/*.tfrecord

# Accounts artifact produced by data/build_tfrecords.py (JSON with snapshot)
ACCOUNTS_ARTIFACT_URI: str = ""  # If empty, use the artifact produced by this run

# Where to save the trained SavedModel (local dir or gs://)
OUTPUT_MODEL_DIR: str = f"gs://{YOUR_BUCKET}/models/je_pointer_v1"

# Ingestion (Cloud SQL Postgres over Cloud SQL Python Connector)
DB_INSTANCE_CONNECTION_NAME: str = "PROJECT:REGION:INSTANCE"
DB_NAME: str = "postgres"
DB_USER: str = "service-account@PROJECT.iam"  # IAM DB auth enabled user or DB user
USE_PRIVATE_IP: bool = False

# Ingestion filters and output
GCS_OUTPUT_URI: str = f"gs://{YOUR_BUCKET}/prefix"  # Base dir where TFRecords and artifacts will be written
BUSINESS_ID: str | None = None
START_DATE: str | None = None  # "YYYY-MM-DD" or None
END_DATE: str | None = None    # "YYYY-MM-DD" or None
SHARD_SIZE: int = 10000

# Text encoder (HF name or path)
ENCODER_LOC: str = "bert-base-multilingual-cased"

# Tokenization/model sizes
MAX_LENGTH: int = 128
MAX_LINES: int = 8
HIDDEN_DIM: int = 256

# Training loop
BATCH_SIZE: int = 32
EPOCHS: int = 1
LEARNING_RATE: float = 2e-4

# Loss weights
COVERAGE_WEIGHT: float = 0.01
FLOW_LOSS_WEIGHT: float = 0.10


def run_ingestion() -> tuple[str, str]:
	"""
	Run DB → TFRecord ingestion and account artifact snapshot.
	Returns:
	  - accounts_artifact_uri
	  - tfrecord_pattern (prefix/tfrecords/*.tfrecord)
	"""
	import time
	if not GCS_OUTPUT_URI.startswith("gs://"):
		raise ValueError("GCS_OUTPUT_URI must start with gs://")

	engine = create_engine_with_connector(
		instance_connection_name=DB_INSTANCE_CONNECTION_NAME,
		db_name=DB_NAME,
		db_user=DB_USER,
		enable_private_ip=bool(USE_PRIVATE_IP),
		enable_iam_auth=True,
	)
	# reflect_or_define_tables constructs SQLAlchemy Table objects
	tables = reflect_or_define_tables(sa.MetaData())

	with engine.connect() as conn:
		# Build account catalog snapshot
		catalog = AccountCatalog.build(conn, tables, business_id=BUSINESS_ID)
		gcs_client = ensure_gcs_client()
		timestamp = time.strftime("%Y%m%d-%H%M%S")
		accounts_artifact_uri = f"{GCS_OUTPUT_URI.rstrip('/')}/artifacts/accounts_{timestamp}.json"
		write_gcs_json(accounts_artifact_uri, catalog.to_artifact(), gcs_client)

		# Write TFRecords
		_manifest = write_examples_to_gcs(
			conn=conn,
			tables=tables,
			catalog= catalog,
			gcs_output_uri=GCS_OUTPUT_URI,
			shard_size=SHARD_SIZE,
			filters={
				"business_id": BUSINESS_ID,
				"start_date": None if START_DATE is None else tf.constant(START_DATE).numpy() if False else START_DATE,  # start/end handled inside write_examples_to_gcs
				"end_date": None if END_DATE is None else END_DATE,
			},
		)
		# Pattern for training
		tfrecord_pattern = f"{GCS_OUTPUT_URI.rstrip('/')}/tfrecords/*.tfrecord"
		return accounts_artifact_uri, tfrecord_pattern


def save_saved_model(model: keras.Model, output_dir: str) -> None:
	"""
	Save a Keras/TF model as SavedModel to a local directory or to a GCS URI.
	"""
	if output_dir.startswith("gs://"):
		tmp_dir = "/tmp/je_saved_model"
		tf.saved_model.save(model, tmp_dir)
		client = storage.Client()
		_, path = output_dir.split("gs://", 1)
		bucket_name, prefix = path.split("/", 1)
		bucket = client.bucket(bucket_name)
		for root, _, files in os.walk(tmp_dir):
			for f in files:
				local_path = os.path.join(root, f)
				rel = os.path.relpath(local_path, tmp_dir)
				blob = bucket.blob(f"{prefix.rstrip('/')}/{rel}")
				blob.upload_from_filename(local_path)
	else:
		tf.saved_model.save(model, output_dir)


def main() -> None:
	# Step 1: Ingest data from DB to GCS (accounts artifact + TFRecords)
	accounts_uri, tfrecord_pattern = run_ingestion()

	# Tokenizer
	tokenizer = DescriptionTokenizer(model_name_or_path=ENCODER_LOC, max_length=MAX_LENGTH)

	# Dataset
	ds = build_dataset(
		pattern=tfrecord_pattern if not INPUT_PATTERN else INPUT_PATTERN,
		tokenizer=tokenizer,
		max_length=MAX_LENGTH,
		max_lines=MAX_LINES,
		batch_size=BATCH_SIZE,
	)

	# Catalog embeddings
	artifact: Dict[str, Any] = load_json_from_uri(accounts_uri if ACCOUNTS_ARTIFACT_URI == "" else ACCOUNTS_ARTIFACT_URI)
	cat_emb_np = build_catalog_embeddings(artifact, emb_dim=HIDDEN_DIM)  # [C, H]
	cat_emb = tf.convert_to_tensor(cat_emb_np, dtype=tf.float32)

	# Retrieval memory placeholder (zeros; can be wired later)
	retr_mem = tf.zeros((1, HIDDEN_DIM), dtype=tf.float32)  # [K=1, H]

	# Model
	model = build_je_model(
		encoder_loc=ENCODER_LOC,
		hidden_dim=HIDDEN_DIM,
		max_lines=MAX_LINES,
		temperature=1.0,
	)

	optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
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
			total = tf.add_n([
				pl,
				sl,
				stl,
				tf.convert_to_tensor(COVERAGE_WEIGHT, dtype=tf.float32) * cov,
				tf.convert_to_tensor(FLOW_LOSS_WEIGHT, dtype=tf.float32) * flow,
			])
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
	for epoch in range(EPOCHS):
		for features, targets in ds:
			stats = train_step(features, targets)
			global_step += 1
			if global_step % 50 == 0:
				tf.print(
					"epoch", epoch, "step", global_step,
					"loss", stats["loss"],
					"ptr", stats["pointer_loss"],
					"side", stats["side_loss"],
					"stop", stats["stop_loss"],
					"cov", stats["cov_pen"],
					"flow", stats["flow_loss"],
					"setF1", metric_set_f1.result(),
				)

	# Save to OUTPUT_MODEL_DIR
	save_saved_model(model, OUTPUT_MODEL_DIR)
	print(f"SavedModel exported to {OUTPUT_MODEL_DIR}")


if __name__ == "__main__":
	main()


