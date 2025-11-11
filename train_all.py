from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time
import uuid
from typing import Any, Dict

import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from data.ingest_to_parquet import (AccountCatalog, create_engine_with_psql,
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
    parser = argparse.ArgumentParser(description="End-to-end: Ingest Parquet and train JE model (PyTorch)")
    # Ingestion
    parser.add_argument("--project", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--instance", type=str, default=os.environ.get("DB_INSTANCE_CONNECTION_NAME", "lb01-438216:us-central1:db-3-postgres"))
    parser.add_argument("--db", type=str, default=os.environ.get("DB_NAME", "postgres"))
    parser.add_argument("--db-user", type=str, default=os.environ.get("DB_USER","postgres"))
    parser.add_argument("--db-password", type=str, default=os.environ.get("DB_PASSWORD","PC=?gB>i6LB5]T9n"))
    parser.add_argument("--db-user-secret", type=str, default=os.environ.get("DB_USER_SECRET_NAME","lb01-438216-db-3-postgres-user"))
    parser.add_argument("--private-ip", action="store_true")
    parser.add_argument("--gcs-output-uri", type=str, default=os.environ.get("GCS_OUTPUT_URI", "./out_all_ingest"))
    parser.add_argument("--gcs-output-uri-secret", type=str, default=os.environ.get("GCS_OUTPUT_URI_SECRET_NAME"))
    parser.add_argument("--business-id", type=str, default=os.environ.get("BUSINESS_ID", "bu-651"))
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
    parser.add_argument("--output-dir", type=str, default=os.environ.get("OUTPUT_DIR", "./out_all"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="je-transformer", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--wandb-entity", type=str, default="liebre-ai", help="W&B entity/team name (optional)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    args = parser.parse_args()

    if args.seed:
        set_seed(int(args.seed))

    # Unique run directory/prefix (used for GCS artifacts and local checkpoints)
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    runs_prefix = f"{args.gcs_output_uri.rstrip('/')}/runs/{run_id}"

    # Ingest to Parquet
    if not args.gcs_output_uri or not args.instance or not args.db_user:
        raise ValueError("--db-user and --gcs-output-uri are required (or via env/secrets)")
    engine = create_engine_with_psql(
        database_name=args.db,
        db_user=args.db_user,
        db_password=args.db_password,
    )
    import sqlalchemy as sa
    metadata = sa.MetaData()
    tables = reflect_or_define_tables(metadata, engine=engine, schema=os.environ.get("DB_SCHEMA", "public"))

    with engine.connect() as conn:
        catalog = AccountCatalog.build(conn, tables, business_id=args.business_id)
        ts = time.strftime("%Y%m%d-%H%M%S")
        from google.cloud import storage
        client = storage.Client()
        # Write accounts artifact into run-specific directory
        accounts_artifact_uri = f"{runs_prefix}/artifacts/accounts_{ts}.json"
        write_gcs_json(accounts_artifact_uri, catalog.to_artifact(), client)
        # Write parquet shards under run-specific prefix
        manifest = write_parquet_shards(
            conn=conn,
            tables=tables,
            catalog=catalog,
            gcs_output_uri=runs_prefix,
            shard_size=int(args.shard_size),
            filters={
                "business_id": args.business_id,
                "start_date": None if not args.start_date else __import__("datetime").date.fromisoformat(args.start_date),
                "end_date": None if not args.end_date else __import__("datetime").date.fromisoformat(args.end_date),
            },
        )
        # Save manifest JSON alongside accounts artifact
        manifest_uri = f"{runs_prefix}/artifacts/manifest_{ts}.json"
        write_gcs_json(manifest_uri, manifest, client)
        parquet_glob = f"{runs_prefix}/parquet/*.parquet"

    # Initialize wandb (after ingestion, before training)
    wandb_enabled = not args.no_wandb
    if wandb_enabled:
        try:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                entity=args.wandb_entity,
                config={
                    "encoder": args.encoder,
                    "hidden_dim": args.hidden_dim,
                    "max_lines": args.max_lines,
                    "max_length": args.max_length,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                    "flow_weight": args.flow_weight,
                    "seed": args.seed,
                    "business_id": args.business_id,
                    "start_date": args.start_date,
                    "end_date": args.end_date,
                    "shard_size": args.shard_size,
                    "run_id": run_id,
                },
                job_type="training",
            )
            # Define custom metrics for better visualization
            wandb.define_metric("step")
            wandb.define_metric("train/*", step_metric="step")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            # Log ingestion metadata (catalog is available from ingestion step above)
            try:
                artifact_data = load_json_from_uri(accounts_artifact_uri)
                accounts_count = len(artifact_data.get("accounts", []))
                wandb.log({
                    "ingestion/accounts_count": accounts_count,
                    "ingestion/parquet_shards": manifest.get("num_shards", 0) if manifest else 0,
                })
            except Exception as e:
                print(f"Warning: Failed to log ingestion metadata: {e}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            wandb_enabled = False

    # Train
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    ds = ParquetJEDataset(parquet_glob, tokenizer_loc=args.encoder, max_length=args.max_length, max_lines=args.max_lines)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    artifact = load_json_from_uri(accounts_artifact_uri)
    cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)

    model = JEModel(encoder_loc=args.encoder, hidden_dim=args.hidden_dim, max_lines=args.max_lines, temperature=1.0).to(device)
    
    # Monitor model with wandb.watch() (always enabled when wandb is enabled)
    if wandb_enabled:
        try:
            wandb.watch(model, log="gradients", log_freq=100)
        except Exception as e:
            print(f"Warning: Failed to enable wandb.watch(): {e}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metric_set_f1 = SetF1Metric()

    model.train()
    global_step = 0
    for epoch in range(args.epochs):
        # Track last metrics from epoch for checkpoint
        last_metrics = None
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
            total.backward(retain_graph=True)
            optimizer.step()
            
            # Store last metrics for checkpoint
            last_metrics = {
                "loss": total.item(),
                "pointer_loss": pl.item(),
                "side_loss": sl.item(),
                "stop_loss": stl.item(),
                "coverage_penalty": cov.item(),
                "flow_loss": flow.item(),
                "set_f1": metric_set_f1.result().item() if global_step % 50 == 0 else (last_metrics["set_f1"] if last_metrics else 0.0),
            }
            
            if global_step % 50 == 0:
                metric_set_f1.update_state(
                    outputs["pointer_logits"].detach().cpu(),
                    outputs["side_logits"].detach().cpu(),
                    target_account_idx.detach().cpu(),
                    target_side_id.detach().cpu(),
                    target_stop_id.detach().cpu(),
                )
                print(
                    f"step {global_step} loss={total.item():.4f} ptr={pl.item():.4f} side={sl.item():.4f} stop={stl.item():.4f} cov={cov.item():.4f} flow={flow.item():.4f} setF1={metric_set_f1.result().item():.4f}"
                )
                # Log metrics to wandb
                if wandb_enabled:
                    try:
                        wandb.log({
                            "step": global_step,
                            "epoch": epoch,
                            "train/loss": total.item(),
                            "train/pointer_loss": pl.item(),
                            "train/side_loss": sl.item(),
                            "train/stop_loss": stl.item(),
                            "train/coverage_penalty": cov.item(),
                            "train/flow_loss": flow.item(),
                            "train/set_f1": metric_set_f1.result().item(),
                            "train/learning_rate": args.lr,
                        })
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")
        
        # Save comprehensive epoch checkpoint
        # Use last metrics from epoch, or default if none available
        if last_metrics is None:
            # Fallback if no metrics were recorded
            last_metrics = {
                "loss": 0.0,
                "pointer_loss": 0.0,
                "side_loss": 0.0,
                "stop_loss": 0.0,
                "coverage_penalty": 0.0,
                "flow_loss": 0.0,
                "set_f1": metric_set_f1.result().item() if hasattr(metric_set_f1, 'result') else 0.0,
            }
        current_metrics = last_metrics
        
        # Create temporary directory for checkpoint files
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Save PyTorch checkpoint
            ckpt_path = os.path.join(tmpdir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None,  # No scheduler in train_all
                "epoch": epoch + 1,
                "step": global_step,
                "metrics": current_metrics,
            }, ckpt_path)
            
            # 2. Save model config
            config_path = os.path.join(tmpdir, "config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "encoder": args.encoder,
                    "hidden_dim": args.hidden_dim,
                    "max_lines": args.max_lines,
                    "max_length": args.max_length,
                    "temperature": 1.0,  # Model default
                }, f, indent=2)
            
            # 3. Save accounts artifact (copy from source)
            accounts_path = os.path.join(tmpdir, "accounts_artifact.json")
            artifact_data = load_json_from_uri(accounts_artifact_uri)
            with open(accounts_path, "w", encoding="utf-8") as f:
                json.dump(artifact_data, f, indent=2, ensure_ascii=False)
            
            # 4. Save training metadata
            metadata_path = os.path.join(tmpdir, "training_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "step": global_step,
                    "total_steps": len(dl) * (epoch + 1),
                    "hyperparameters": {
                        "encoder": args.encoder,
                        "hidden_dim": args.hidden_dim,
                        "max_lines": args.max_lines,
                        "max_length": args.max_length,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "flow_weight": args.flow_weight,
                        "seed": args.seed,
                        "business_id": args.business_id,
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                        "shard_size": args.shard_size,
                    },
                    "metrics": current_metrics,
                }, f, indent=2)
            
            # 5. Create wandb artifact
            if wandb_enabled:
                try:
                    artifact = wandb.Artifact(
                        name=f"model-epoch-{epoch+1}",
                        type="model",
                        metadata={
                            "epoch": epoch + 1,
                            "step": global_step,
                            "set_f1": current_metrics["set_f1"],
                            "loss": current_metrics["loss"],
                        }
                    )
                    artifact.add_file(ckpt_path, name="checkpoint.pt")
                    artifact.add_file(config_path, name="config.json")
                    artifact.add_file(accounts_path, name="accounts_artifact.json")
                    artifact.add_file(metadata_path, name="training_metadata.json")
                    wandb.log_artifact(artifact)
                except Exception as e:
                    print(f"Warning: Failed to log wandb artifact: {e}")
            
            # 6. Also save to local/gs:// for backward compatibility
            os.makedirs(args.output_dir, exist_ok=True) if not args.output_dir.startswith("gs://") else None
            local_ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt") if not args.output_dir.startswith("gs://") else f"/tmp/checkpoint_epoch_{epoch+1}.pt"
            
            # Copy checkpoint to final location
            shutil.copy2(ckpt_path, local_ckpt_path)
            
            if args.output_dir.startswith("gs://"):
                from google.cloud import storage

                client = storage.Client()
                _, path = args.output_dir.split("gs://", 1)
                bucket_name, prefix = path.split("/", 1)
                blob = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/checkpoint_epoch_{epoch+1}.pt")
                blob.upload_from_filename(local_ckpt_path)

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
    
    # Log final summary to wandb
    if wandb_enabled:
        try:
            wandb.summary.update({
                "final_epoch": args.epochs,
                "final_step": global_step,
                "final_set_f1": metric_set_f1.result().item(),
            })
            wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to update wandb summary: {e}")


if __name__ == "__main__":
    main()

