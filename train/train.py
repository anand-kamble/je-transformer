from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from typing import Any, Dict

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train JE pointer model (PyTorch)")
    parser.add_argument("--parquet-pattern", type=str, required=True, help="gs:// or local glob of Parquet shards")
    parser.add_argument("--accounts-artifact", type=str, required=True, help="Accounts snapshot JSON (gs:// or local)")
    parser.add_argument("--output-dir", type=str, required=True, help="Local or gs:// dir for checkpoints")
    parser.add_argument("--encoder", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-lines", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max grad norm; 0 to disable")
    parser.add_argument("--flow-weight", type=float, default=0.10)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="je-transformer", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team name (optional)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    args = parser.parse_args()

    if args.seed:
        set_seed(int(args.seed))

    # Initialize wandb
    wandb_enabled = not args.no_wandb and wandb is not None
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
                        "use_amp": args.use_amp,
                        "grad_clip": args.grad_clip,
                        "flow_weight": args.flow_weight,
                        "seed": args.seed,
                    },
                    job_type="training",
                )
            # Define custom metrics for better visualization
            wandb.define_metric("step")
            wandb.define_metric("train/*", step_metric="step")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            wandb_enabled = False

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    ds = ParquetJEDataset(args.parquet_pattern, tokenizer_loc=args.encoder, max_length=args.max_length, max_lines=args.max_lines)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    artifact = load_json_from_uri(args.accounts_artifact)
    cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)

    model = JEModel(encoder_loc=args.encoder, hidden_dim=args.hidden_dim, max_lines=args.max_lines, temperature=1.0).to(device)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Monitor model with wandb.watch() (always enabled when wandb is enabled)
    if wandb_enabled:
        try:
            wandb.watch(model, log="gradients", log_freq=100)
        except Exception as e:
            print(f"Warning: Failed to enable wandb.watch(): {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(dl) * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.use_amp and device.type == "cuda"))

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
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
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
            scaler.scale(total).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            metric_set_f1.update_state(outputs["pointer_logits"].detach().cpu(), outputs["side_logits"].detach().cpu(), target_account_idx.detach().cpu(), target_side_id.detach().cpu(), target_stop_id.detach().cpu())

            if step % 50 == 0:
                print(
                    f"step {step} loss={total.item():.4f} ptr={pl.item():.4f} side={sl.item():.4f} stop={stl.item():.4f} cov={cov.item():.4f} flow={flow.item():.4f} setF1={metric_set_f1.result().item():.4f}"
                )
                # Log metrics to wandb
                if wandb_enabled:
                    try:
                        wandb.log({
                            "step": step,
                            "epoch": epoch,
                            "train/loss": total.item(),
                            "train/pointer_loss": pl.item(),
                            "train/side_loss": sl.item(),
                            "train/stop_loss": stl.item(),
                            "train/coverage_penalty": cov.item(),
                            "train/flow_loss": flow.item(),
                            "train/set_f1": metric_set_f1.result().item(),
                            "train/learning_rate": scheduler.get_last_lr()[0] if scheduler else args.lr,
                        })
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")
        # Save comprehensive epoch checkpoint
        # Store current metrics for checkpoint
        current_metrics = {
            "loss": total.item(),
            "pointer_loss": pl.item(),
            "side_loss": sl.item(),
            "stop_loss": stl.item(),
            "coverage_penalty": cov.item(),
            "flow_loss": flow.item(),
            "set_f1": metric_set_f1.result().item(),
        }
        
        # Create temporary directory for checkpoint files
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Save PyTorch checkpoint
            ckpt_path = os.path.join(tmpdir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch + 1,
                "step": step,
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
            artifact_data = load_json_from_uri(args.accounts_artifact)
            with open(accounts_path, "w", encoding="utf-8") as f:
                json.dump(artifact_data, f, indent=2, ensure_ascii=False)
            
            # 4. Save training metadata
            metadata_path = os.path.join(tmpdir, "training_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump({
                    "epoch": epoch + 1,
                    "step": step,
                    "total_steps": len(dl) * (epoch + 1),
                    "hyperparameters": {
                        "encoder": args.encoder,
                        "hidden_dim": args.hidden_dim,
                        "max_lines": args.max_lines,
                        "max_length": args.max_length,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "use_amp": args.use_amp,
                        "grad_clip": args.grad_clip,
                        "flow_weight": args.flow_weight,
                        "seed": args.seed,
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
                            "step": step,
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
    ckpt_final = os.path.join(args.output_dir, "model_state.pt") if not args.output_dir.startswith("gs://") else "/tmp/model_state.pt"
    torch.save({"model": model.state_dict()}, ckpt_final)
    if args.output_dir.startswith("gs://"):
        from google.cloud import storage

        client = storage.Client()
        _, path = args.output_dir.split("gs://", 1)
        bucket_name, prefix = path.split("/", 1)
        blob = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/model_state.pt")
        blob.upload_from_filename(ckpt_final)
        print(f"Uploaded checkpoint to {args.output_dir}")
    else:
        print(f"Checkpoint written to {ckpt_final}")
    
    # Log final summary to wandb
    if wandb_enabled:
        try:
            wandb.summary.update({
                "final_epoch": args.epochs,
                "final_step": step,
                "final_set_f1": metric_set_f1.result().item(),
            })
            wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to update wandb summary: {e}")


if __name__ == "__main__":
    main()



