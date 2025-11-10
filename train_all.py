from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from data.ingest_to_parquet import (AccountCatalog,
                                    create_engine_with_connector,
                                    reflect_or_define_tables, write_gcs_json,
                                    write_parquet_shards)
from models.catalog_encoder import CatalogEncoder
from models.je_model_torch import JEModel
from models.losses import (SetF1Metric, coverage_penalty, flow_aux_loss,
                           pointer_loss, side_loss, stop_loss)
from train.dataset import ParquetJEDataset, collate_fn


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
    parser.add_argument("--flow-weight", type=float, default=0.10)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Ingest to Parquet
    if not args.gcs_output_uri or not args.instance or not args.db_user:
        raise ValueError("--instance, --db-user and --gcs-output-uri are required (or via env/secrets)")
    engine = create_engine_with_connector(
        instance_connection_name=args.instance,
        db_name=args.db,
        db_user=args.db_user,
        db_password=os.environ.get("DB_PASSWORD"),
        enable_private_ip=bool(args.private_ip),
        enable_iam_auth=True,
    )
    metadata = __import__("sqlalchemy").MetaData()
    tables = reflect_or_define_tables(metadata)

    with engine.connect() as conn:
        catalog = AccountCatalog.build(conn, tables, business_id=args.business_id)
        from google.cloud import storage
        client = storage.Client()
        import time
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        accounts_artifact_uri = f"{args.gcs_output_uri.rstrip('/')}/artifacts/accounts_{timestamp}.json"
        write_gcs_json(accounts_artifact_uri, catalog.to_artifact(), client)
        manifest = write_parquet_shards(
            conn=conn,
            tables=tables,
            catalog=catalog,
            gcs_output_uri=args.gcs_output_uri,
            shard_size=args.shard_size,
            filters={
                "business_id": args.business_id,
                "start_date": None if not args.start_date else __import__("datetime").date.fromisoformat(args.start_date),
                "end_date": None if not args.end_date else __import__("datetime").date.fromisoformat(args.end_date),
            },
        )
        parquet_glob = f"{args.gcs_output_uri.rstrip('/')}/parquet/*.parquet"

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    ds = ParquetJEDataset(parquet_glob, tokenizer_loc=args.encoder, max_length=args.max_length, max_lines=args.max_lines)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    artifact = load_json_from_uri(accounts_artifact_uri)
    cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)
    retr_mem = torch.zeros((1, args.hidden_dim), dtype=torch.float32, device=device)

    model = JEModel(encoder_loc=args.encoder, hidden_dim=args.hidden_dim, max_lines=args.max_lines, temperature=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metric_set_f1 = SetF1Metric()

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        for features, targets in dl:
            global_step += 1
            input_ids = features["input_ids"].to(device)
            attention_mask = features["attention_mask"].to(device)
            prev_account_idx = features["prev_account_idx"].to(device)
            prev_side_id = features["prev_side_id"].to(device)
            cond_numeric = features["cond_numeric"].to(device)
            currency = features["currency"]
            journal_entry_type = features["journal_entry_type"]

            target_account_idx = targets["target_account_idx"].to(device)
            target_side_id = targets["target_side_id"].to(device)
            target_stop_id = targets["target_stop_id"].to(device)
            debit_indices = targets["debit_indices"].to(device)
            debit_weights = targets["debit_weights"].to(device)
            credit_indices = targets["credit_indices"].to(device)
            credit_weights = targets["credit_weights"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prev_account_idx=prev_account_idx,
                prev_side_id=prev_side_id,
                catalog_embeddings=cat_emb,
                retrieval_memory=retr_mem,
                cond_numeric=cond_numeric,
                currency=currency,
                journal_entry_type=journal_entry_type,
            )
            pl = pointer_loss(outputs["pointer_logits"], target_account_idx, ignore_index=-1)
            sl = side_loss(outputs["side_logits"], target_side_id, ignore_index=-1)
            stl = stop_loss(outputs["stop_logits"], target_stop_id, ignore_index=-1)
            cov = coverage_penalty(outputs["pointer_logits"]) * 0.01
            flow = flow_aux_loss(
                outputs["pointer_logits"],
                outputs["side_logits"],
                debit_indices,
                debit_weights,
                credit_indices,
                credit_weights,
            ) * float(args.flow_weight)
            total = pl + sl + stl + cov + flow
            total.backward()
            optimizer.step()
            if global_step % 50 == 0:
                metric_set_f1.update_state(
                    outputs["pointer_logits"].detach().cpu(),
                    outputs["side_logits"].detach().cpu(),
                    target_account_idx.detach().cpu(),
                    target_side_id.detach().cpu(),
                    target_stop_id.detach().cpu(),
                )
                print(
                    f"step {global_step} loss {total.item():.4f} ptr {pl.item():.4f} side {sl.item():.4f} stop {stl.item():.4f} cov {cov.item():.4f} flow {flow.item():.4f} setF1 {metric_set_f1.result().item():.4f}"
                )

    # Save final
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({"model": model.state_dict()}, os.path.join(args.output_dir, "model_state.pt"))
    print(f"Saved model state_dict to {args.output_dir}")



