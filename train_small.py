from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset

from data.ingest_to_parquet import (AccountCatalog, build_join_statement,
                                    create_engine_with_psql, date_features,
                                    normalize_amounts,
                                    reflect_or_define_tables, write_gcs_json,
                                    write_parquet_shards)
from models.catalog_encoder import CatalogEncoder
from models.je_model_torch import JEModel
from models.losses import (SetF1Metric, coverage_penalty, flow_aux_loss,
                           pointer_loss, side_loss, stop_loss)
from train.dataset import ParquetJEDataset, collate_fn

load_dotenv(override=True)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json_from_uri(uri: str) -> Dict[str, Any]:
    print(f"Loading JSON from URI: {uri}")
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


def build_catalog_embeddings(artifact: Dict[str, Any], emb_dim: int, device: torch.device) -> torch.Tensor:
    accounts = artifact["accounts"]
    number = [a.get("number", "") for a in accounts]
    name = [a.get("name", "") for a in accounts]
    nature = [a.get("nature", "") for a in accounts]
    encoder = CatalogEncoder(emb_dim=emb_dim)
    embs = encoder({"number": number, "name": name, "nature": nature})  # [C, emb_dim]
    return embs.to(device=device, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick small training on a subset of Parquet shards (PyTorch)")
    parser.add_argument(
        "--parquet-pattern",
        type=str,
        default=os.environ.get("PARQUET_PATTERN", "./data/parquet/*.parquet"),
        help="Local or gs:// glob of Parquet shards (default: ./data/parquet/*.parquet or $PARQUET_PATTERN)",
    )
    parser.add_argument(
        "--accounts-artifact",
        type=str,
        default=os.environ.get("ACCOUNTS_ARTIFACT"),
        help="Accounts JSON (local or gs://) (default: ./data/artifacts/accounts.json or $ACCOUNTS_ARTIFACT)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "./out_small"),
        help="Dir (or gs://) to write checkpoints (default: ./out_small or $OUTPUT_DIR)",
    )
    # Optional: end-to-end ingestion (Cloud SQL -> Parquet on GCS) before training
    parser.add_argument("--project", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--instance", type=str, default=os.environ.get("DB_INSTANCE_CONNECTION_NAME", "lb01-438216:us-central1:db-3-postgres"))
    parser.add_argument("--db", type=str, default=os.environ.get("DB_NAME", "liebre_dev"))
    parser.add_argument("--db-user", type=str, default=os.environ.get("DB_USER","postgres"))
    parser.add_argument("--db-password", type=str, default=os.environ.get("DB_PASSWORD","PC=?gB>i6LB5]T9n"))
    parser.add_argument("--private-ip", action="store_true")
    parser.add_argument(
        "--gcs-output-uri",
        type=str,
        default=os.environ.get("GCS_OUTPUT_URI", "./out_small_ingest"),
        help="Where to write Parquet/artifacts. If starts with gs://, writes to GCS else to local dir (default: ./out_small_ingest)",
    )
    parser.add_argument("--business-id", type=str, default=os.environ.get("BUSINESS_ID", "bu-651"))
    parser.add_argument("--start-date", type=str, default=os.environ.get("START_DATE"))
    parser.add_argument("--end-date", type=str, default=os.environ.get("END_DATE"))
    parser.add_argument("--shard-size", type=int, default=int(os.environ.get("SHARD_SIZE", "2000")))
    # Tiny defaults for quick runs
    parser.add_argument("--encoder", type=str, default="prajjwal1/bert-tiny", help="Small HF encoder for speed")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-lines", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--flow-weight", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=200, help="Train on first N examples only")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed:
        set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

    # Optional end-to-end ingestion if GCS output is provided (recommended for full flow)
    parquet_pattern = args.parquet_pattern
    accounts_artifact_path = args.accounts_artifact
    if args.gcs_output_uri:
        if not args.instance or not args.db_user:
            raise ValueError("--instance and --db-user are required (or set DB_INSTANCE_CONNECTION_NAME/DB_USER) to run ingestion")
        engine = create_engine_with_psql(
            database_name=args.db,
            db_user=args.db_user,
            db_password=args.db_password,
        )
        import json as _json
        import time

        import pandas as _pd
        import sqlalchemy as sa
        metadata = sa.MetaData()
        tables = reflect_or_define_tables(metadata, engine=engine, schema=os.environ.get("DB_SCHEMA", "public"))
        with engine.connect() as conn:
            catalog = AccountCatalog.build(conn, tables, business_id=args.business_id)
            ts = time.strftime("%Y%m%d-%H%M%S")
            if args.gcs_output_uri.startswith("gs://"):
                from google.cloud import storage
                client = storage.Client()
                accounts_artifact_path = f"{args.gcs_output_uri.rstrip('/')}/artifacts/accounts_{ts}.json"
                write_gcs_json(accounts_artifact_path, catalog.to_artifact(), client)
                _ = write_parquet_shards(
                    conn=conn,
                    tables=tables,
                    catalog=catalog,
                    gcs_output_uri=args.gcs_output_uri,
                    shard_size=int(args.shard_size),
                    filters={
                        "business_id": args.business_id,
                        "start_date": None if not args.start_date else __import__("datetime").date.fromisoformat(args.start_date),
                        "end_date": None if not args.end_date else __import__("datetime").date.fromisoformat(args.end_date),
                    },
                )
                parquet_pattern = f"{args.gcs_output_uri.rstrip('/')}/parquet/*.parquet"
            else:
                # Local directory output
                base_dir = args.gcs_output_uri.rstrip("/")
                os.makedirs(os.path.join(base_dir, "artifacts"), exist_ok=True)
                os.makedirs(os.path.join(base_dir, "parquet"), exist_ok=True)
                accounts_artifact_path = os.path.join(base_dir, "artifacts", f"accounts_{ts}.json")
                with open(accounts_artifact_path, "w", encoding="utf-8") as f:
                    _json.dump(catalog.to_artifact(), f, ensure_ascii=False, indent=2)
                # Stream rows and write parquet shards locally
                shard_size = int(args.shard_size)
                shard_index = 0
                shard_records = 0
                current_rows: List[Dict[str, Any]] = []
                all_entries: List[Dict[str, Any]] = []
                stmt = build_join_statement(
                    tables=tables,
                    business_id=args.business_id,
                    start_date=None if not args.start_date else __import__("datetime").date.fromisoformat(args.start_date),
                    end_date=None if not args.end_date else __import__("datetime").date.fromisoformat(args.end_date),
                )
                result = conn.execute(stmt.execution_options(stream_results=True))
                current_je_id: Optional[str] = None
                current_je: Dict[str, Any] = {}
                debit_accounts: List[int] = []
                credit_accounts: List[int] = []
                debit_amounts: List[float] = []
                credit_amounts: List[float] = []

                def write_shard(rows: List[Dict[str, Any]], si: int) -> str:
                    out_path = os.path.join(base_dir, "parquet", f"part-{si:05d}-{ts}.parquet")
                    df = _pd.DataFrame(rows)
                    for col in ("debit_accounts", "credit_accounts", "debit_amounts_norm", "credit_amounts_norm"):
                        if col in df.columns:
                            df[col] = df[col].apply(lambda x: list(x) if isinstance(x, list) else (list(x) if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)) else []))
                    df.to_parquet(out_path, engine="pyarrow", index=False)
                    return out_path

                def flush_current():
                    nonlocal shard_index, shard_records, current_rows
                    if current_je_id is None:
                        return
                    row = {
                        "journal_entry_id": current_je.get("journal_entry_id"),
                        "business_id": current_je.get("business_id"),
                        "description": current_je.get("je_description", ""),
                        "currency": current_je.get("currency", ""),
                        "journal_entry_type": current_je.get("journal_entry_type", ""),
                    }
                    row.update(date_features(current_je["date"]))
                    row["debit_accounts"] = [int(x) for x in debit_accounts]
                    row["credit_accounts"] = [int(x) for x in credit_accounts]
                    row["debit_amounts_norm"] = normalize_amounts(debit_amounts)
                    row["credit_amounts_norm"] = normalize_amounts(credit_amounts)
                    current_rows.append(row)
                    all_entries.append(row)
                    shard_records += 1
                    if shard_records >= shard_size:
                        _ = write_shard(current_rows, shard_index)
                        shard_index += 1
                        shard_records = 0
                        current_rows = []

                for r in result:
                    je_id = str(r.journal_entry_id)
                    if current_je_id is None:
                        current_je_id = je_id
                        current_je = {
                            "journal_entry_id": je_id,
                            "business_id": r.business_id,
                            "date": r.date,
                            "je_description": r.je_description or "",
                            "currency": r.currency or "",
                            "journal_entry_type": r.journal_entry_type or "",
                            "journal_entry_sub_type": r.journal_entry_sub_type or "",
                        }
                    if je_id != current_je_id:
                        flush_current()
                        current_je_id = je_id
                        debit_accounts, credit_accounts = [], []
                        debit_amounts, credit_amounts = [], []
                        current_je = {
                            "journal_entry_id": je_id,
                            "business_id": r.business_id,
                            "date": r.date,
                            "je_description": r.je_description or "",
                            "currency": r.currency or "",
                            "journal_entry_type": r.journal_entry_type or "",
                            "journal_entry_sub_type": r.journal_entry_sub_type or "",
                        }
                    account_id = None if r.ledger_account_id is None else str(r.ledger_account_id)
                    if account_id is None:
                        continue
                    idx = catalog.id_to_index.get(account_id)
                    if idx is None:
                        continue
                    debit_val = float(r.debit or 0.0)
                    credit_val = float(r.credit or 0.0)
                    if debit_val > 0.0:
                        debit_accounts.append(int(idx))
                        debit_amounts.append(debit_val)
                    elif credit_val > 0.0:
                        credit_accounts.append(int(idx))
                        credit_amounts.append(credit_val)

                flush_current()
                if current_rows:
                    _ = write_shard(current_rows, shard_index)
                parquet_pattern = os.path.join(base_dir, "parquet", "*.parquet")
                # Write the full set of journal entries as a JSON artifact for training/inspection
                entries_json_path = os.path.join(base_dir, "artifacts", f"journal_entries_{ts}.json")
                with open(entries_json_path, "w", encoding="utf-8") as jf:
                    _json.dump(all_entries, jf, ensure_ascii=False, indent=2)

    # Dataset and small subset
    full_ds = ParquetJEDataset(parquet_pattern, tokenizer_loc=args.encoder, max_length=args.max_length, max_lines=args.max_lines)
    n = len(full_ds)
    lim = max(1, min(int(args.limit), n))
    idx: List[int] = list(range(lim))
    ds = Subset(full_ds, idx)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    artifact = load_json_from_uri(accounts_artifact_path)
    cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)

    model = JEModel(encoder_loc=args.encoder, hidden_dim=args.hidden_dim, max_lines=args.max_lines, temperature=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metric_set_f1 = SetF1Metric()

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for features, targets in dl:
            step += 1
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

            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prev_account_idx=prev_account_idx,
                prev_side_id=prev_side_id,
                catalog_embeddings=cat_emb,
                retrieval_memory=None,
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
            optimizer.step()

            if step % 20 == 0:
                metric_set_f1.update_state(
                    outputs["pointer_logits"].detach().cpu(),
                    outputs["side_logits"].detach().cpu(),
                    target_account_idx.detach().cpu(),
                    target_side_id.detach().cpu(),
                    target_stop_id.detach().cpu(),
                )
                print(
                    f"step {step} loss={total.item():.4f} ptr={pl.item():.4f} side={sl.item():.4f} stop={stl.item():.4f} cov={cov.item():.4f} flow={flow.item():.4f} setF1={metric_set_f1.result().item():.4f}"
                )

    # Save final checkpoint
    if args.output_dir.startswith("gs://"):
        tmp = "/tmp/model_state.pt"
        torch.save({"model": model.state_dict()}, tmp)
        from google.cloud import storage

        client = storage.Client()
        _, path = args.output_dir.split("gs://", 1)
        bucket_name, prefix = path.split("/", 1)
        blob = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/model_state.pt")
        blob.upload_from_filename(tmp)
        print(f"Uploaded checkpoint to {args.output_dir}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        ckpt = os.path.join(args.output_dir, "model_state.pt")
        torch.save({"model": model.state_dict()}, ckpt)
        print(f"Checkpoint written to {ckpt}")


if __name__ == "__main__":
    main()
