from __future__ import annotations


import argparse
import json
import os
import random
import shutil
import tempfile
from typing import Any, Dict, List, Optional


import numpy as np
import torch
import wandb
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Subset


# Ingestion is now handled by data/ingest_to_parquet.py (external step)
from models.catalog_encoder import CatalogEncoder
from models.je_model_torch import JEModel, mean_pool
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
    return embs.to(device=device, dtype=torch.float32).detach()



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
    parser.add_argument("--max-lines", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--flow-weight", type=float, default=0.05)
    # Pointer/logit scaling and catalog options
    parser.add_argument("--pointer-temp", type=float, default=0.1, help="Pointer softmax temperature (smaller -> larger logits)")
    parser.add_argument("--pointer-scale-init", type=float, default=10.0, help="Initial multiplicative scale for pointer logits")
    parser.add_argument("--learnable-pointer-scale", action="store_true", help="Make pointer logit scale learnable")
    parser.add_argument("--no-pointer-norm", action="store_true", help="Disable L2 normalization in pointer layer")
    parser.add_argument("--trainable-catalog", action="store_true", help="Make catalog embeddings trainable inside model")
    # Flow warmup
    parser.add_argument("--flow-warmup-epochs", type=int, default=3, help="Epochs to apply warmup multiplier to flow loss")
    parser.add_argument("--flow-warmup-multiplier", type=float, default=5.0, help="Multiplier for flow loss during warmup")
    # Optional retrieval memory integration
    parser.add_argument("--retrieval-index-dir", type=str, default=os.environ.get("RETRIEVAL_INDEX_DIR"), help="ScaNN index dir (gs:// or local). If provided with ids and embeddings, retrieval is enabled.")
    parser.add_argument("--retrieval-ids-uri", type=str, default=os.environ.get("RETRIEVAL_IDS_URI"), help="Text file with ids (gs:// or local)")
    parser.add_argument("--retrieval-embeddings-uri", type=str, default=os.environ.get("RETRIEVAL_EMBEDDINGS_URI"), help="NumPy .npy of neighbor embeddings aligned to ids (gs:// or local)")
    parser.add_argument("--retrieval-top-k", type=int, default=int(os.environ.get("RETRIEVAL_TOP_K", "5")))
    parser.add_argument("--retrieval-use-cls", action="store_true", help="Use [CLS] for pooling the query (defaults to mean pooling)")
    parser.add_argument("--limit", type=int, default=20000, help="Train on first N examples only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", type=str, default="je-transformer", help="W&B project name")
    parser.add_argument("--wandb-name", type=str, default=None, help="W&B run name (optional)")
    parser.add_argument("--wandb-entity", type=str, default="liebre-ai", help="W&B entity/team name (optional)")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    args = parser.parse_args()


    if args.seed:
        set_seed(int(args.seed))

    # Generate unique run name/ID EARLY (before any file operations)
    wandb_enabled = not args.no_wandb
    run_id: Optional[str] = None
    if wandb_enabled:
        # Pre-generate run ID so we can use it for directory naming
        try:
            import uuid
            run_id = uuid.uuid4().hex
        except Exception:
            run_id = None
        run_name = args.wandb_name or run_id
    else:
        # If wandb disabled, fallback to timestamp
        import time
        run_name = time.strftime("%Y%m%d-%H%M%S")
    # Ensure run_name is a non-empty string
    if not run_name:
        import time as _time
        run_name = _time.strftime("%Y%m%d-%H%M%S")
    
    # Update ALL output directories to use run_name
    # 1. Update main output directory
    if not args.output_dir.startswith("gs://"):
        args.output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # For GCS, append run_name to path
        args.output_dir = f"{args.output_dir.rstrip('/')}/{run_name}"
    
    # 2. Update ingestion output directory
    if args.gcs_output_uri:
        if not args.gcs_output_uri.startswith("gs://"):
            args.gcs_output_uri = os.path.join(args.gcs_output_uri, run_name)
            os.makedirs(args.gcs_output_uri, exist_ok=True)
        else:
            args.gcs_output_uri = f"{args.gcs_output_uri.rstrip('/')}/{run_name}"

    # Initialize wandb with the pre-generated run name/ID
    if wandb_enabled:
        try:
            wandb.init(
                project=args.wandb_project,
                name=run_name,
                id=run_id,
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
                    "pointer_temp": args.pointer_temp,
                    "pointer_scale_init": args.pointer_scale_init,
                    "learnable_pointer_scale": args.learnable_pointer_scale,
                    "pointer_use_norm": not args.no_pointer_norm,
                    "trainable_catalog": args.trainable_catalog,
                    "flow_warmup_epochs": args.flow_warmup_epochs,
                    "flow_warmup_multiplier": args.flow_warmup_multiplier,
                    "retrieval_index_dir": args.retrieval_index_dir,
                    "retrieval_ids_uri": args.retrieval_ids_uri,
                    "retrieval_embeddings_uri": args.retrieval_embeddings_uri,
                    "retrieval_top_k": args.retrieval_top_k,
                    "retrieval_use_cls": args.retrieval_use_cls,
                    "limit": args.limit,
                    "seed": args.seed,
                },
                job_type="training",
            )
            # Verify run_name matches
            _rn = getattr(wandb, "run", None)
            if _rn is not None and getattr(_rn, "name", None):
                run_name = _rn.name  # type: ignore[assignment]
            print(f"Initialized wandb run: {run_name}")
            print(f"All outputs will be saved to: {args.output_dir}")

            # Define custom metrics for better visualization
            wandb.define_metric("step")
            wandb.define_metric("train/*", step_metric="step")
            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            wandb_enabled = False


    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))


    # Upstream ingestion has moved to data/ingest_to_parquet.py.
    # Expect prebuilt Parquet shards and accounts artifact via CLI.
    parquet_pattern = args.parquet_pattern
    accounts_artifact_path = args.accounts_artifact


    # Dataset and small subset
    full_ds = ParquetJEDataset(parquet_pattern, tokenizer_loc=args.encoder, max_length=args.max_length, max_lines=args.max_lines)
    n = len(full_ds)
    lim = max(1, min(int(args.limit), n))
    subset_indices: List[int] = list(range(lim))
    ds = Subset(full_ds, subset_indices)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)


    artifact = load_json_from_uri(accounts_artifact_path)
    cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)


    model = JEModel(
        encoder_loc=args.encoder,
        hidden_dim=args.hidden_dim,
        max_lines=args.max_lines,
        temperature=float(args.pointer_temp),
        pointer_scale_init=float(args.pointer_scale_init),
        pointer_learnable_scale=bool(args.learnable_pointer_scale),
        use_pointer_norm=not args.no_pointer_norm,
        learn_catalog=bool(args.trainable_catalog),
    ).to(device)
    # Optional: load retrieval artifacts once and pre-project embeddings to model hidden dim
    retrieval_enabled = bool(args.retrieval_index_dir and args.retrieval_ids_uri and args.retrieval_embeddings_uri)
    _retr_searcher = None
    _retr_proj_embs = None  # torch.Tensor on CPU [N, H_hidden]
    if retrieval_enabled:
        try:
            import scann  # type: ignore
            import numpy as _np
            # Helper loaders supporting gs:// and local
            def _load_bytes(path: str) -> bytes:
                if path.startswith("gs://"):
                    from google.cloud import storage  # type: ignore
                    client = storage.Client()
                    _, p = path.split("gs://", 1)
                    bkt, blob_name = p.split("/", 1)
                    blob = client.bucket(bkt).blob(blob_name)
                    return blob.download_as_bytes()
                with open(path, "rb") as f:
                    return f.read()
            def _download_scann_dir(src: str) -> str:
                import tempfile as _tf, os as _os
                if not src.startswith("gs://"):
                    return src
                tmpdir = _tf.mkdtemp(prefix="scann_idx_")
                from google.cloud import storage  # type: ignore
                client = storage.Client()
                _, path = src.split("gs://", 1)
                bucket_name, prefix = path.split("/", 1)
                bucket = client.bucket(bucket_name)
                for blob in bucket.list_blobs(prefix=prefix.rstrip("/")):
                    if blob.name.endswith("/"):
                        continue
                    lp = _os.path.join(tmpdir, _os.path.relpath(blob.name, start=prefix.rstrip("/")))
                    _os.makedirs(_os.path.dirname(lp), exist_ok=True)
                    blob.download_to_filename(lp)
                return tmpdir
            # Load searcher and embeddings
            _idx_dir_local = _download_scann_dir(args.retrieval_index_dir)
            _retr_searcher = scann.scann_ops_pybind.load_searcher(_idx_dir_local)
            # Load embeddings and ids
            ids_bytes = _load_bytes(args.retrieval_ids_uri)
            ids_text = ids_bytes.decode("utf-8")
            _ = [line for line in ids_text.splitlines() if line.strip()]  # kept for alignment checks if needed later
            emb_bytes = _load_bytes(args.retrieval_embeddings_uri)
            emb_np = _np.load(__import__("io").BytesIO(emb_bytes))  # [N, H_enc]
            # Project to hidden_dim using the model's enc_proj and tanh, on CPU and no grad
            with torch.no_grad():
                emb_t = torch.tensor(emb_np, dtype=torch.float32)  # CPU
                proj = torch.tanh(model.enc_proj(emb_t.to(device=model.enc_proj.weight.device)))  # [N, H_hidden] on model device
                _retr_proj_embs = proj.detach().cpu()  # keep on CPU, move to device per batch
            print(f"Loaded retrieval artifacts. Embeddings: {_retr_proj_embs.shape}, top_k={int(args.retrieval_top_k)}")
        except Exception as e:
            print(f"Warning: retrieval disabled due to load error: {e}")
            retrieval_enabled = False
    # Small helper to build retrieval memory per batch
    def _build_retrieval_memory_batch(_inp_ids: torch.Tensor, _attn: torch.Tensor) -> Optional[torch.Tensor]:
        if not retrieval_enabled:
            return None
        assert _retr_searcher is not None and _retr_proj_embs is not None
        with torch.no_grad():
            enc_out = model.encoder(input_ids=_inp_ids, attention_mask=_attn)
            pooled = enc_out.last_hidden_state[:, 0, :] if args.retrieval_use_cls else mean_pool(enc_out.last_hidden_state, _attn)
            q_cpu = pooled.detach().cpu()
            q_cpu = q_cpu / (q_cpu.norm(dim=1, keepdim=True) + 1e-8)
            nbrs, _ = _retr_searcher.search_batched(q_cpu.numpy(), final_num_neighbors=int(args.retrieval_top_k))
            mem_list = []
            for i in range(len(nbrs)):
                idx_tensor = torch.tensor(nbrs[i], dtype=torch.long)
                mem_i = _retr_proj_embs.index_select(0, idx_tensor)  # [K, H]
                mem_list.append(mem_i)
            mem = torch.stack(mem_list, dim=0).to(device)  # [B, K, H]
        return mem
    if args.trainable_catalog:
        try:
            model.set_catalog_embeddings(cat_emb)
        except Exception as e:
            print(f"Warning: failed to register trainable catalog embeddings: {e}")
    
    # Monitor model with wandb.watch() (always enabled when wandb is enabled)
    if wandb_enabled:
        try:
            wandb.watch(model, log="gradients", log_freq=100)
        except Exception as e:
            print(f"Warning: Failed to enable wandb.watch(): {e}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metric_set_f1 = SetF1Metric()


    model.train()
    step = 0
    best_loss = float("inf")
    best_epoch = 0
    best_metrics: Optional[Dict[str, float]] = None
    for epoch in range(args.epochs):
        # Track last metrics from epoch for checkpoint
        last_metrics = None
        for features, targets in dl:
            step += 1
            optimizer.zero_grad(set_to_none=True)
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
            # Optional retrieval memory
            retr_mem = _build_retrieval_memory_batch(input_ids, attention_mask) if retrieval_enabled else None
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
            
            # Compute softmax ONCE
            pointer_probs = torch.softmax(outputs["pointer_logits"], dim=-1)
            side_probs = torch.softmax(outputs["side_logits"], dim=-1)


            # Pass probabilities instead of logits where applicable
            pl = pointer_loss(outputs["pointer_logits"], target_account_idx, ignore_index=-1)
            sl = side_loss(outputs["side_logits"], target_side_id, ignore_index=-1)
            stl = stop_loss(outputs["stop_logits"], target_stop_id, ignore_index=-1)


            # These use the pre-computed probabilities
            cov = coverage_penalty(outputs["pointer_logits"], probs=pointer_probs) * 0.01     # Modify function
            # Flow loss with warmup schedule
            flow_weight_now = float(args.flow_weight) * (float(args.flow_warmup_multiplier) if epoch < int(args.flow_warmup_epochs) else 1.0)
            flow = flow_aux_loss(
                outputs["pointer_logits"],
                outputs["side_logits"],
                debit_indices,
                debit_weights,
                credit_indices,
                credit_weights,
                pointer_probs=pointer_probs,
                side_probs=side_probs,
            ) * flow_weight_now


            total = pl + sl + stl + cov + flow
            total.backward()  # No retain_graph needed!
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Store last metrics for checkpoint
            last_metrics = {
                "loss": total.item(),
                "pointer_loss": pl.item(),
                "side_loss": sl.item(),
                "stop_loss": stl.item(),
                "coverage_penalty": cov.item(),
                "flow_loss": flow.item(),
                "set_f1": metric_set_f1.result().item() if step % 20 == 0 else (last_metrics["set_f1"] if last_metrics else 0.0),
            }


            if step % 100 == 0:
                # Pointer logit stats for diagnostics
                try:
                    p_logits = outputs["pointer_logits"].detach()
                    p_max = float(p_logits.max().item())
                    p_min = float(p_logits.min().item())
                    p_std = float(p_logits.std().item())
                except Exception:
                    p_max = p_min = p_std = 0.0
                metric_set_f1.update_state(
                    outputs["pointer_logits"].detach().cpu(),
                    outputs["side_logits"].detach().cpu(),
                    target_account_idx.detach().cpu(),
                    target_side_id.detach().cpu(),
                    target_stop_id.detach().cpu(),
                )
                print(
                    f"step {step} loss={total.item():.4f} ptr={pl.item():.4f} side={sl.item():.4f} stop={stl.item():.4f} cov={cov.item():.4f} flow={flow.item():.4f} setF1={metric_set_f1.result().item():.4f} pmax={p_max:.3f} pstd={p_std:.3f}"
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
                            "train/learning_rate": args.lr,
                            "train/pointer_logit_max": p_max,
                            "train/pointer_logit_min": p_min,
                            "train/pointer_logit_std": p_std,
                            "train/flow_weight_now": flow_weight_now,
                        })
                    except Exception as e:
                        print(f"Warning: Failed to log to wandb: {e}")
        
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

        # Save only if this is the best so far (by lowest loss)
        current_loss = float(current_metrics.get("loss", float("inf")))
        if current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch + 1
            best_metrics = dict(current_metrics)
            with tempfile.TemporaryDirectory() as tmpdir:
                # 1. Save model_state.pt (for inference)
                model_state_path = os.path.join(tmpdir, "model_state.pt")
                torch.save({"model": model.state_dict()}, model_state_path)
                # 2. Save model config
                config_path = os.path.join(tmpdir, "config.json")
                with open(config_path, "w") as f:
                    json.dump({
                        "encoder": args.encoder,
                        "hidden_dim": args.hidden_dim,
                        "max_lines": args.max_lines,
                        "max_length": args.max_length,
                        "temperature": float(args.pointer_temp),
                        "pointer_scale_init": float(args.pointer_scale_init),
                        "learnable_pointer_scale": bool(args.learnable_pointer_scale),
                        "pointer_use_norm": not args.no_pointer_norm,
                        "trainable_catalog": bool(args.trainable_catalog),
                        "retrieval_index_dir": args.retrieval_index_dir,
                        "retrieval_ids_uri": args.retrieval_ids_uri,
                        "retrieval_embeddings_uri": args.retrieval_embeddings_uri,
                        "retrieval_top_k": int(args.retrieval_top_k),
                        "retrieval_use_cls": bool(args.retrieval_use_cls),
                    }, f, indent=2)
                # 3. Save accounts artifact
                accounts_path = os.path.join(tmpdir, "accounts_artifact.json")
                artifact_data = load_json_from_uri(accounts_artifact_path)
                with open(accounts_path, "w", encoding="utf-8") as f:
                    json.dump(artifact_data, f, indent=2, ensure_ascii=False)
                # 4. Save training metadata
                metadata_path = os.path.join(tmpdir, "training_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump({
                        "run_name": run_name,
                        "best_epoch": best_epoch,
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
                            "flow_weight": args.flow_weight,
                            "pointer_temp": args.pointer_temp,
                            "pointer_scale_init": args.pointer_scale_init,
                            "learnable_pointer_scale": args.learnable_pointer_scale,
                            "pointer_use_norm": not args.no_pointer_norm,
                            "trainable_catalog": args.trainable_catalog,
                            "flow_warmup_epochs": args.flow_warmup_epochs,
                            "flow_warmup_multiplier": args.flow_warmup_multiplier,
                            "retrieval_index_dir": args.retrieval_index_dir,
                            "retrieval_ids_uri": args.retrieval_ids_uri,
                            "retrieval_embeddings_uri": args.retrieval_embeddings_uri,
                            "retrieval_top_k": args.retrieval_top_k,
                            "retrieval_use_cls": args.retrieval_use_cls,
                            "limit": args.limit,
                            "seed": args.seed,
                        },
                        "metrics": best_metrics,
                    }, f, indent=2)

                # Log wandb artifact for best only
                if wandb_enabled:
                    try:
                        artifact = wandb.Artifact(
                            name="model-best",
                            type="model",
                            metadata={
                                "epoch": best_epoch,
                                "step": step,
                                "set_f1": best_metrics.get("set_f1", 0.0),
                                "loss": best_metrics.get("loss", 0.0),
                            }
                        )
                        artifact.add_file(model_state_path, name="model_state.pt")
                        artifact.add_file(config_path, name="config.json")
                        artifact.add_file(accounts_path, name="accounts_artifact.json")
                        artifact.add_file(metadata_path, name="training_metadata.json")
                        wandb.log_artifact(artifact)
                    except Exception as e:
                        print(f"Warning: Failed to log wandb artifact: {e}")

                # Save best files to output dir (local or GCS)
                if args.output_dir.startswith("gs://"):
                    from google.cloud import storage
                    client = storage.Client()
                    _, path = args.output_dir.split("gs://", 1)
                    bucket_name, prefix = path.split("/", 1)
                    for fname, fpath in [
                        ("model_state.pt", model_state_path),
                        ("config.json", config_path),
                        ("accounts_artifact.json", accounts_path),
                        ("training_metadata.json", metadata_path),
                    ]:
                        blob = client.bucket(bucket_name).blob(f"{prefix.rstrip('/')}/{fname}")
                        blob.upload_from_filename(fpath)
                    print(f"New best (epoch {best_epoch}, loss={best_loss:.4f}) uploaded to {args.output_dir}")
                else:
                    for fname, fpath in [
                        ("model_state.pt", model_state_path),
                        ("config.json", config_path),
                        ("accounts_artifact.json", accounts_path),
                        ("training_metadata.json", metadata_path),
                    ]:
                        shutil.copy2(fpath, os.path.join(args.output_dir, fname))
                    print(f"New best (epoch {best_epoch}, loss={best_loss:.4f}) saved to {args.output_dir}")

    # Only the best checkpoint is saved during training; nothing else to do here.
    
    # Log final summary to wandb
    if wandb_enabled:
        try:
            wandb.summary.update({
                "final_epoch": args.epochs,
                "final_step": step,
                "final_set_f1": metric_set_f1.result().item(),
                "output_directory": args.output_dir,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "best_set_f1": (best_metrics.get("set_f1") if isinstance(best_metrics, dict) else None) if best_metrics is not None else None,
            })
            wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to update wandb summary: {e}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Run name: {run_name}")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"{'='*60}")



if __name__ == "__main__":
    main()
