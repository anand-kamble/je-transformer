from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from data.ingest_to_parquet import ingest_to_parquet
from models.catalog_encoder import CatalogEncoder
from models.je_model import build_je_model
from models.losses import (SetF1Hungarian, coverage_penalty, flow_aux_loss,
                           pointer_loss, side_loss, stop_loss)
from train.dataset import JEConfig, ParquetJEDataset, collate_fn


def load_json_from_uri(uri: str) -> Dict[str, Any]:
	if uri.startswith("gs://"):
		from google.cloud import storage
		client = storage.Client()
		_, path = uri.split("gs://", 1)
		bucket_name, blob_path = path.split("/", 1)
		blob = client.bucket(bucket_name).blob(blob_path)
		data = blob.download_as_bytes()
		import json
		return json.loads(data.decode("utf-8"))
	import json
	with open(uri, "r", encoding="utf-8") as f:
		return json.load(f)


def build_catalog_embeddings(artifact: Dict[str, Any], emb_dim: int, device: torch.device) -> torch.Tensor:
	accounts = artifact["accounts"]
	numbers = [a.get("number", "") for a in accounts]
	names = [a.get("name", "") for a in accounts]
	nature = [a.get("nature", "") for a in accounts]
	encoder = CatalogEncoder(emb_dim=emb_dim)
	embs = encoder({"number": numbers, "name": names, "nature": nature})  # [C, emb_dim]
	return embs.to(device)


def main() -> None:
	parser = argparse.ArgumentParser(description="End-to-end: Ingest Parquet and train JE model (PyTorch)")
	# Ingestion
	parser.add_argument("--project", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
	parser.add_argument("--instance", type=str, default=os.environ.get("DB_INSTANCE_CONNECTION_NAME"))
	parser.add_argument("--db", type=str, default=os.environ.get("DB_NAME", "postgres"))
	parser.add_argument("--db-user", type=str, default=os.environ.get("DB_USER"))
	parser.add_argument("--db-user-secret", type=str, default=os.environ.get("DB_USER_SECRET_NAME"))
	parser.add_argument("--private-ip", action="store_true")
	parser.add_argument("--gcs-output-uri", type=str, default=os.environ.get("GCS_OUTPUT_URI"))
	parser.add_argument("--gcs-output-uri-secret", type=str, default=os.environ.get("GCS_OUTPUT_URI_SECRET_NAME"))
	parser.add_argument("--business-id", type=str, default=os.environ.get("BUSINESS_ID"))
	parser.add_argument("--start-date", type=str, default=os.environ.get("START_DATE"))
	parser.add_argument("--end-date", type=str, default=os.environ.get("END_DATE"))
	parser.add_argument("--shard-size", type=int, default=int(os.environ.get("SHARD_SIZE", "10000")))
	parser.add_argument("--secrets-project", type=str, default=os.environ.get("SECRETS_PROJECT_ID") or os.environ.get("GOOGLE_CLOUD_PROJECT"))
	# Training
	parser.add_argument("--encoder", type=str, default="bert-base-multilingual-cased")
	parser.add_argument("--max-length", type=int, default=128)
	parser.add_argument("--max-lines", type=int, default=8)
	parser.add_argument("--hidden-dim", type=int, default=256)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--epochs", type=int, default=1)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--output-dir", type=str, required=True)
	args = parser.parse_args()

	# Ingest to Parquet
	ing = ingest_to_parquet(
		instance_connection_name=args.instance,
		db_name=args.db,
		db_user=args.db_user,
		gcs_output_uri=args.gcs_output_uri,
		business_id=args.business_id,
		start_date=args.start_date,
		end_date=args.end_date,
		shard_size=args.shard_size,
		secrets_project=args.secrets_project,
		db_user_secret=args.db_user_secret,
		gcs_output_uri_secret=args.gcs_output_uri_secret,
		use_private_ip=bool(args.private_ip),
	)
	parquet_glob = os.path.join(os.path.dirname(ing["manifest_uri"]), "..", "parquet", "*.parquet")
	parquet_glob = os.path.normpath(parquet_glob)
	accounts_artifact = ing["accounts_artifact_uri"]

	# Train
	device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
	cfg = JEConfig(tokenizer_loc=args.encoder, max_length=args.max_length, max_lines=args.max_lines)
	paths = sorted(glob.glob(parquet_glob))
	if not paths:
		raise ValueError(f"No Parquet files matched after ingestion: {parquet_glob}")
	ds = ParquetJEDataset(paths, cfg)
	dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

	artifact = load_json_from_uri(accounts_artifact)
	cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)
	retr_mem = torch.zeros((1, args.hidden_dim), dtype=torch.float32, device=device)

	model = build_je_model(
		encoder_loc=args.encoder,
		hidden_dim=args.hidden_dim,
		max_lines=args.max_lines,
		temperature=1.0,
	).to(device)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
	metric_set_f1 = SetF1Hungarian()

	model.train()
	global_step = 0
	for epoch in range(args.epochs):
		for features, targets in dl:
			global_step += 1
			features = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()}
			targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in targets.items()}
			optimizer.zero_grad()
			outputs = model({
				"input_ids": features["input_ids"],
				"attention_mask": features["attention_mask"],
				"prev_account_idx": features["prev_account_idx"],
				"prev_side_id": features["prev_side_id"],
				"catalog_embeddings": cat_emb,
				"retrieval_memory": retr_mem,
				"cond_numeric": features["cond_numeric"],
				"currency": features["currency"],
				"journal_entry_type": features["journal_entry_type"],
			})
			pl = pointer_loss(outputs["pointer_logits"], targets["target_account_idx"], ignore_index=-1)
			sl = side_loss(outputs["side_logits"], targets["target_side_id"], ignore_index=-1)
			stl = stop_loss(outputs["stop_logits"], targets["target_stop_id"], ignore_index=-1)
			cov = coverage_penalty(outputs["pointer_logits"])
			flow = flow_aux_loss(
				outputs["pointer_logits"],
				outputs["side_logits"],
				targets["debit_indices"],
				targets["debit_weights"],
				targets["credit_indices"],
				targets["credit_weights"],
			)
			total = pl + sl + stl + 0.01 * cov + 0.1 * flow
			total.backward()
			optimizer.step()
			if global_step % 50 == 0:
				metric_set_f1.update_state(
					outputs["pointer_logits"].detach().cpu(),
					outputs["side_logits"].detach().cpu(),
					targets["target_account_idx"].detach().cpu(),
					targets["target_side_id"].detach().cpu(),
					targets["target_stop_id"].detach().cpu(),
				)
				print(
					f"step {global_step} loss {total.item():.4f} ptr {pl.item():.4f} side {sl.item():.4f} stop {stl.item():.4f} cov {cov.item():.4f} setF1 {metric_set_f1.result():.4f}"
				)

	# Save final
	os.makedirs(args.output_dir, exist_ok=True)
	torch.save({"model": model.state_dict()}, os.path.join(args.output_dir, "model_state.pt"))
	print(f"Saved model state_dict to {args.output_dir}")


if __name__ == "__main__":
	main()


