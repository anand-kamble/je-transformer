from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from models.catalog_encoder import CatalogEncoder
from models.je_model_torch import JEModel
from models.losses import (SetF1Metric, coverage_penalty, pointer_loss,
                           side_loss, stop_loss)
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    ds = ParquetJEDataset(args.parquet_pattern, tokenizer_loc=args.encoder, max_length=args.max_length, max_lines=args.max_lines)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)

    artifact = load_json_from_uri(args.accounts_artifact)
    cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)

    model = JEModel(encoder_loc=args.encoder, hidden_dim=args.hidden_dim, max_lines=args.max_lines, temperature=1.0).to(device)

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
                total = pl + sl + stl + cov
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
                    f"step {step} loss={total.item():.4f} ptr={pl.item():.4f} side={sl.item():.4f} stop={stl.item():.4f} cov={cov.item():.4f} setF1={metric_set_f1.result().item():.4f}"
                )

    # Save checkpoint (state_dict)
    os.makedirs(args.output_dir, exist_ok=True) if not args.output_dir.startswith("gs://") else None
    ckpt_path = os.path.join(args.output_dir, "model_state.pt") if not args.output_dir.startswith("gs://") else "/tmp/model_state.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)

    if args.output_dir.startswith("gs://"):
        from google.cloud import storage

        client = storage.Client()
        _, path = args.output_dir.split("gs://", 1)
        bucket_name, prefix = path.split("/", 1)
        blob = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/model_state.pt")
        blob.upload_from_filename(ckpt_path)
        print(f"Uploaded checkpoint to {args.output_dir}")
    else:
        print(f"Checkpoint written to {ckpt_path}")


if __name__ == "__main__":
    main()



