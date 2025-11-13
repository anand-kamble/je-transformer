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
from transformers import get_cosine_schedule_with_warmup


from models.catalog_encoder import HierarchicalCatalogEncoder
from models.hierarchy_utils import AccountHierarchy
from models.je_model_torch import JEModel, mean_pool
from models.losses import (
    SetF1Metric,
    coverage_penalty,
    flow_aux_loss,
    hierarchical_pointer_loss,
    side_loss,
    stop_loss,
    WindowedSetF1Metric,
)
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


def build_catalog_embeddings(
    artifact: Dict[str, Any], emb_dim: int, device: torch.device
) -> torch.Tensor:
    accounts = artifact["accounts"]
    number = [a.get("number", "") for a in accounts]
    name = [a.get("name", "") for a in accounts]
    nature = [a.get("nature", "") for a in accounts]
    encoder = HierarchicalCatalogEncoder(emb_dim=emb_dim)
    embs = encoder({"number": number, "name": name, "nature": nature})
    return embs.to(device=device, dtype=torch.float32).detach()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick small training on a subset of Parquet shards (PyTorch)"
    )
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

    parser.add_argument(
        "--project", type=str, default=os.environ.get("GOOGLE_CLOUD_PROJECT")
    )
    parser.add_argument(
        "--instance",
        type=str,
        default=os.environ.get(
            "DB_INSTANCE_CONNECTION_NAME", "lb01-438216:us-central1:db-3-postgres"
        ),
    )
    parser.add_argument(
        "--db", type=str, default=os.environ.get("DB_NAME", "liebre_dev")
    )
    parser.add_argument(
        "--db-user", type=str, default=os.environ.get("DB_USER", "postgres")
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default=os.environ.get("DB_PASSWORD", "PC=?gB>i6LB5]T9n"),
    )
    parser.add_argument("--private-ip", action="store_true")
    parser.add_argument(
        "--gcs-output-uri",
        type=str,
        default=os.environ.get("GCS_OUTPUT_URI", "./out_small_ingest"),
        help="Where to write Parquet/artifacts. If starts with gs://, writes to GCS else to local dir (default: ./out_small_ingest)",
    )
    parser.add_argument(
        "--business-id", type=str, default=os.environ.get("BUSINESS_ID", "bu-651")
    )
    parser.add_argument("--start-date", type=str, default=os.environ.get("START_DATE"))
    parser.add_argument("--end-date", type=str, default=os.environ.get("END_DATE"))
    parser.add_argument(
        "--shard-size", type=int, default=int(os.environ.get("SHARD_SIZE", "2000"))
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="prajjwal1/bert-tiny",
        help="Small HF encoder for speed",
    )
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-lines", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--flow-weight", type=float, default=0.01)

    parser.add_argument(
        "--pointer-temp",
        type=float,
        default=0.5,
        help="Pointer softmax temperature (larger -> softer distribution)",
    )
    parser.add_argument(
        "--pointer-scale-init",
        type=float,
        default=5.0,
        help="Initial multiplicative scale for pointer logits",
    )
    parser.add_argument(
        "--learnable-pointer-scale",
        action="store_true",
        help="Make pointer logit scale learnable",
    )
    parser.add_argument(
        "--no-pointer-norm",
        action="store_true",
        help="Disable L2 normalization in pointer layer",
    )
    parser.add_argument(
        "--trainable-catalog",
        action="store_true",
        help="Make catalog embeddings trainable inside model",
    )

    parser.add_argument(
        "--flow-warmup-epochs",
        type=int,
        default=3,
        help="Epochs to apply warmup multiplier to flow loss",
    )
    parser.add_argument(
        "--flow-warmup-multiplier",
        type=float,
        default=5.0,
        help="Multiplier for flow loss during warmup",
    )

    parser.add_argument(
        "--max-grad-norm", type=float, default=0.5, help="Gradient clipping max norm"
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        type=int,
        default=2,
        help="Epochs for LR warmup in cosine schedule",
    )

    parser.add_argument(
        "--retrieval-index-dir",
        type=str,
        default=os.environ.get("RETRIEVAL_INDEX_DIR"),
        help="ScaNN index dir (gs://). Enables retrieval when provided",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=int(os.environ.get("RETRIEVAL_TOP_K", "5")),
    )
    parser.add_argument(
        "--retrieval-use-cls",
        action="store_true",
        help="Use [CLS] for pooling the query (defaults to mean pooling)",
    )

    parser.add_argument(
        "--setf1-window-steps",
        type=int,
        default=100,
        help="Window size for training SetF1 metric",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Epochs with no val/loss improvement before stopping (0=disable)",
    )
    parser.add_argument(
        "--log-hist-every",
        type=int,
        default=0,
        help="Log pointer logits histogram every N steps (0=disabled)",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of training subset reserved for validation",
    )
    parser.add_argument(
        "--max-val-batches",
        type=int,
        default=0,
        help="Max validation batches per epoch (0=all)",
    )
    parser.add_argument(
        "--log-retrieval-heatmaps",
        action="store_true",
        help="Log retrieval fusion weight heatmaps to W&B",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=4,
        help="Num samples to visualize for retrieval heatmaps",
    )
    parser.add_argument(
        "--limit", type=int, default=20000, help="Train on first N examples only"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wandb-project", type=str, default="je-transformer", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-name", type=str, default=None, help="W&B run name (optional)"
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default="liebre-ai",
        help="W&B entity/team name (optional)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    args = parser.parse_args()

    if args.seed:
        set_seed(int(args.seed))

    wandb_enabled = not args.no_wandb
    run_id: Optional[str] = None
    if wandb_enabled:
        try:
            import uuid

            run_id = uuid.uuid4().hex
        except Exception:
            run_id = None
        run_name = args.wandb_name or run_id
    else:
        import time

        run_name = time.strftime("%Y%m%d-%H%M%S")

    if not run_name:
        import time as _time

        run_name = _time.strftime("%Y%m%d-%H%M%S")

    if not args.output_dir.startswith("gs://"):
        args.output_dir = os.path.join(args.output_dir, run_name)
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        args.output_dir = f"{args.output_dir.rstrip('/')}/{run_name}"

    if args.gcs_output_uri:
        if not args.gcs_output_uri.startswith("gs://"):
            args.gcs_output_uri = os.path.join(args.gcs_output_uri, run_name)
            os.makedirs(args.gcs_output_uri, exist_ok=True)
        else:
            args.gcs_output_uri = f"{args.gcs_output_uri.rstrip('/')}/{run_name}"

    def get_loss_weights(epoch, total_epochs):
        progress = epoch / total_epochs

        flow_weight = args.flow_weight * (2.0 - progress)

        coverage_weight = 0.01 * (1.0 + progress)

        if epoch < 3:
            flow_warmup = args.flow_warmup_multiplier
        else:
            flow_warmup = 1.0

        return {
            "flow": flow_weight * flow_warmup,
            "coverage": coverage_weight,
        }

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
                    "retrieval_top_k": args.retrieval_top_k,
                    "retrieval_use_cls": args.retrieval_use_cls,
                    "limit": args.limit,
                    "seed": args.seed,
                },
                job_type="training",
            )

            _rn = getattr(wandb, "run", None)
            if _rn is not None and getattr(_rn, "name", None):
                run_name = _rn.name
            print(f"Initialized wandb run: {run_name}")
            print(f"All outputs will be saved to: {args.output_dir}")

            wandb.define_metric("epoch")
            wandb.define_metric("train/*", step_metric="epoch")
            wandb.define_metric("val/*", step_metric="epoch")

            try:
                wandb.define_metric("val/loss", summary="min")
                wandb.define_metric("val/set_f1", summary="max")
                wandb.define_metric("train/loss", summary="min")
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            wandb_enabled = False

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else (
            "mps"
            if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cpu"
        )
    )

    parquet_pattern = args.parquet_pattern
    accounts_artifact_path = args.accounts_artifact

    full_ds = ParquetJEDataset(
        parquet_pattern,
        tokenizer_loc=args.encoder,
        max_length=args.max_length,
        max_lines=args.max_lines,
    )
    n = len(full_ds)
    lim = max(1, min(int(args.limit), n))
    subset_indices: List[int] = list(range(lim))

    rng = np.random.RandomState(int(args.seed) if args.seed is not None else 0)
    rng.shuffle(subset_indices)
    val_count = (
        int(max(1, round(float(args.val_ratio) * len(subset_indices))))
        if args.val_ratio > 0
        else 0
    )
    val_indices = subset_indices[:val_count] if val_count > 0 else []
    train_indices = subset_indices[val_count:] if val_count > 0 else subset_indices
    train_ds = Subset(full_ds, train_indices)
    val_ds = Subset(full_ds, val_indices) if val_indices else None
    dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    dl_val = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        if val_ds is not None
        else None
    )

    artifact = load_json_from_uri(accounts_artifact_path)
    hierarchy = AccountHierarchy(artifact["accounts"])
    cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)

    catalog_size = cat_emb.size(0)

    jeid_to_accountset: Dict[str, set] = {}
    try:
        df_local = full_ds.df

        def _safe_to_list(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (bytes, bytearray)):
                try:
                    s_text_bytes: str = value.decode("utf-8").strip()
                except Exception:
                    return []
                if len(s_text_bytes) >= 2 and (
                    (s_text_bytes[0] == "[" and s_text_bytes[-1] == "]")
                    or (s_text_bytes[0] == "(" and s_text_bytes[-1] == ")")
                ):
                    import json as _json

                    try:
                        s_json = (
                            "[" + s_text_bytes[1:-1] + "]"
                            if (s_text_bytes[0] == "(" and s_text_bytes[-1] == ")")
                            else s_text_bytes
                        )
                        parsed = _json.loads(s_json)
                        return list(parsed) if isinstance(parsed, (list, tuple)) else []
                    except Exception:
                        return []
                return []
            if isinstance(value, str):
                s_text = value.strip()
                if len(s_text) >= 2 and (
                    (s_text[0] == "[" and s_text[-1] == "]")
                    or (s_text[0] == "(" and s_text[-1] == ")")
                ):
                    import json as _json

                    try:
                        s_json = (
                            "[" + s_text[1:-1] + "]"
                            if (s_text[0] == "(" and s_text[-1] == ")")
                            else s_text
                        )
                        parsed = _json.loads(s_json)
                        return list(parsed) if isinstance(parsed, (list, tuple)) else []
                    except Exception:
                        return []
                return []
            return []

        for _, r in df_local.iterrows():
            jid = str(r.get("journal_entry_id", "") or "")
            if not jid:
                continue
            deb_l = [int(x) for x in _safe_to_list(r.get("debit_accounts"))]
            cre_l = [int(x) for x in _safe_to_list(r.get("credit_accounts"))]
            jeid_to_accountset[jid] = set(deb_l + cre_l)
    except Exception as _emap_e:
        print(
            f"[retrieval-metrics] Warning: failed building jeid->account_set mapping: {_emap_e}"
        )

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

    retrieval_enabled = bool(args.retrieval_index_dir)
    _retr_searcher = None
    _retr_proj_embs = None
    _normalize_queries = True
    _retr_index_ids: Optional[List[str]] = None
    if not retrieval_enabled:
        print("[retrieval] Disabled: --retrieval-index-dir not provided")
    if retrieval_enabled:
        try:
            import scann
            import numpy as _np

            def _load_bytes(path: str) -> bytes:
                if path.startswith("gs://"):
                    from google.cloud import storage

                    client = storage.Client()
                    _, p = path.split("gs://", 1)
                    bkt, blob_name = p.split("/", 1)
                    blob = client.bucket(bkt).blob(blob_name)
                    return blob.download_as_bytes()
                with open(path, "rb") as f:
                    return f.read()

            def _download_scann_dir(src: str) -> str:
                import tempfile as _tf
                import os as _os

                if not src.startswith("gs://"):
                    raise RuntimeError(
                        "Retrieval is GCS-only; --retrieval-index-dir must be a gs:// URI"
                    )
                tmpdir = _tf.mkdtemp(prefix="scann_idx_")
                from google.cloud import storage

                client = storage.Client()
                _, path = src.split("gs://", 1)
                bucket_name, prefix = path.split("/", 1)
                bucket = client.bucket(bucket_name)
                print(
                    f"[retrieval] Downloading ScaNN index from gs://{bucket_name}/{prefix.rstrip('/')}"
                )

                required_files = [
                    "serialized_partitioner.pb",
                    "scann_config.pb",
                    "scann_assets.pbtxt",
                    "ah_codebook.pb",
                ]
                optional_files = [
                    "dataset.npy",
                    "hashed_dataset.npy",
                    "datapoint_to_token.npy",
                ]
                for fname in required_files:
                    blob = bucket.blob(f"{prefix.rstrip('/')}/{fname}")
                    exists = blob.exists(client)
                    print(
                        f"[retrieval]   required '{fname}': {'FOUND' if exists else 'MISSING'} at gs://{bucket_name}/{prefix.rstrip('/')}/{fname}"
                    )
                    if not exists:
                        raise RuntimeError(
                            f"Missing required ScaNN file '{fname}' at {src}. "
                            "Ensure the index directory contains the serialized ScaNN files at its root."
                        )
                    local_path = _os.path.join(tmpdir, fname)
                    blob.download_to_filename(local_path)
                    try:
                        sz = _os.path.getsize(local_path)
                    except Exception:
                        sz = -1
                    print(
                        f"[retrieval]   downloaded '{fname}' to {local_path} ({sz} bytes)"
                    )
                    if _os.path.getsize(local_path) == 0:
                        raise RuntimeError(
                            f"Downloaded zero-byte ScaNN file '{fname}' from {src}"
                        )

                for fname in optional_files:
                    blob = bucket.blob(f"{prefix.rstrip('/')}/{fname}")
                    if blob.exists(client):
                        local_path = _os.path.join(tmpdir, fname)
                        blob.download_to_filename(local_path)
                        try:
                            sz = _os.path.getsize(local_path)
                        except Exception:
                            sz = -1
                        print(
                            f"[retrieval]   downloaded optional '{fname}' ({sz} bytes)"
                        )

                try:
                    assets_path = _os.path.join(tmpdir, "scann_assets.pbtxt")
                    if _os.path.exists(assets_path):
                        with open(assets_path, "r", encoding="utf-8") as af:
                            txt = af.read()
                        before = txt

                        import re as _re

                        def _basename_repl(match):
                            path_str = match.group(1)
                            base = _os.path.basename(path_str)
                            return f'"{base}"'

                        txt = _re.sub(r"\"(/tmp/[^\"]+)\"", _basename_repl, txt)

                        def _basename_repl_generic(match):
                            path_str = match.group(1)
                            if path_str.startswith("/"):
                                base = _os.path.basename(path_str)
                                return f'"{base}"'
                            return f'"{path_str}"'

                        txt = _re.sub(
                            r"\"([^\"]+/[^\"]+)\"", _basename_repl_generic, txt
                        )
                        if txt != before:
                            with open(assets_path, "w", encoding="utf-8") as af:
                                af.write(txt)
                            print(
                                "[retrieval] Rewrote scann_assets.pbtxt to use relative paths."
                            )
                except Exception as e:
                    print(
                        f"[retrieval] Warning: failed to rewrite scann_assets.pbtxt: {e}"
                    )
                return tmpdir

            print(f"[retrieval] Using retrieval index dir: {args.retrieval_index_dir}")
            _idx_dir_local = _download_scann_dir(args.retrieval_index_dir)

            print(f"[retrieval] Loading ScaNN searcher from {_idx_dir_local} ...")
            try:
                _retr_searcher = scann.scann_ops_pybind.load_searcher(_idx_dir_local)
            except Exception as e_load:
                import os as _os

                print(f"[retrieval] ERROR loading ScaNN searcher: {e_load}")
                try:
                    print("[retrieval] Local index dir contents:")
                    for f in sorted(_os.listdir(_idx_dir_local)):
                        fp = _os.path.join(_idx_dir_local, f)
                        try:
                            sz = _os.path.getsize(fp)
                        except Exception:
                            sz = -1
                        print(f"    {f}  ({sz} bytes)")
                except Exception as e_ls:
                    print(f"[retrieval] Failed to list local index dir: {e_ls}")
                raise

            manifest = None
            manifest_path = os.path.join(_idx_dir_local, "index_manifest.json")
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    manifest = json.load(mf)
                _normalize_queries = bool(manifest.get("l2_normalized", True))
                try:
                    print(
                        f"[retrieval] Loaded manifest: {json.dumps(manifest, indent=2)[:500]}{'...' if len(json.dumps(manifest)) > 500 else ''}"
                    )
                except Exception:
                    print(
                        "[retrieval] Loaded manifest (printing skipped due to size/encoding)"
                    )

            emb_np = None
            candidates: List[str] = []
            ids_rel = None
            if manifest and isinstance(manifest.get("files", {}), dict):
                emb_rel = manifest["files"].get("embeddings")
                ids_rel = manifest["files"].get("ids")
                if emb_rel:
                    candidates.append(os.path.join(_idx_dir_local, emb_rel))
            candidates.extend(
                [
                    os.path.join(_idx_dir_local, "embeddings.npy"),
                    os.path.join(_idx_dir_local, "dataset.npy"),
                ]
            )
            print("[retrieval] Embedding candidates (in order):")
            for c in candidates:
                print(f"    {c}")
            for p in candidates:
                if os.path.exists(p):
                    emb_np = _np.load(p)
                    print(
                        f"[retrieval] Using embeddings from: {p}, shape={getattr(emb_np, 'shape', None)}"
                    )
                    break
            if emb_np is None:
                raise RuntimeError(
                    "Could not locate retrieval embeddings (tried manifest → embeddings.npy → dataset.npy)."
                )

            _ids_path = os.path.join(_idx_dir_local, "ids.txt")
            if os.path.exists(_ids_path):
                with open(_ids_path, "r", encoding="utf-8") as f:
                    _retr_index_ids = [line.strip() for line in f if line.strip()]
                print(f"[retrieval] Loaded ids.txt with {len(_retr_index_ids)} ids")
            elif manifest and isinstance(manifest.get("files", {}), dict) and ids_rel:
                _alt = os.path.join(_idx_dir_local, ids_rel)
                if os.path.exists(_alt):
                    with open(_alt, "r", encoding="utf-8") as f:
                        _retr_index_ids = [line.strip() for line in f if line.strip()]
                    print(
                        f"[retrieval] Loaded ids via manifest path with {len(_retr_index_ids)} ids"
                    )
            else:
                print(
                    "[retrieval] Warning: ids.txt not found; retrieval metrics will be limited"
                )

            with torch.no_grad():
                emb_t = torch.tensor(emb_np, dtype=torch.float32)
                proj = torch.tanh(
                    model.enc_proj(emb_t.to(device=model.enc_proj.weight.device))
                )
                _retr_proj_embs = proj.detach().cpu()
            print(
                f"[retrieval] Loaded. proj_embeddings={tuple(_retr_proj_embs.shape)}, top_k={int(args.retrieval_top_k)}, normalize_queries={_normalize_queries}"
            )
        except Exception as e:
            print(f"Warning: retrieval disabled due to load error: {e}")
            retrieval_enabled = False

    def _build_retrieval_memory_batch(
        _inp_ids: torch.Tensor, _attn: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not retrieval_enabled:
            return None
        assert _retr_searcher is not None and _retr_proj_embs is not None
        with torch.no_grad():
            enc_out = model.encoder(input_ids=_inp_ids, attention_mask=_attn)
            pooled = (
                enc_out.last_hidden_state[:, 0, :]
                if args.retrieval_use_cls
                else mean_pool(enc_out.last_hidden_state, _attn)
            )
            q_cpu = pooled.detach().cpu()
            if _normalize_queries:
                q_cpu = q_cpu / (q_cpu.norm(dim=1, keepdim=True) + 1e-8)
            nbrs, _ = _retr_searcher.search_batched(
                q_cpu.numpy(), final_num_neighbors=int(args.retrieval_top_k)
            )
            mem_list = []
            for i in range(len(nbrs)):
                idx_tensor = torch.tensor(nbrs[i], dtype=torch.long)
                mem_i = _retr_proj_embs.index_select(0, idx_tensor)
                mem_list.append(mem_i)
            mem = torch.stack(mem_list, dim=0).to(device)
        return mem

    def _neighbors_and_memory_batch(
        _inp_ids: torch.Tensor, _attn: torch.Tensor
    ) -> Optional[tuple]:
        if not retrieval_enabled:
            return None
        assert _retr_searcher is not None and _retr_proj_embs is not None
        with torch.no_grad():
            enc_out = model.encoder(input_ids=_inp_ids, attention_mask=_attn)
            pooled = (
                enc_out.last_hidden_state[:, 0, :]
                if args.retrieval_use_cls
                else mean_pool(enc_out.last_hidden_state, _attn)
            )
            q_cpu = pooled.detach().cpu()
            if _normalize_queries:
                q_cpu = q_cpu / (q_cpu.norm(dim=1, keepdim=True) + 1e-8)
            nbrs, _ = _retr_searcher.search_batched(
                q_cpu.numpy(), final_num_neighbors=int(args.retrieval_top_k)
            )
            mem_list = []
            for i in range(len(nbrs)):
                idx_tensor = torch.tensor(nbrs[i], dtype=torch.long)
                mem_i = _retr_proj_embs.index_select(0, idx_tensor)
                mem_list.append(mem_i)
            mem = torch.stack(mem_list, dim=0).to(device)
            return nbrs, mem
        return None

    def clip_grad_by_component(model, max_norm):
        param_groups = {
            "pointer": [
                p
                for n, p in model.named_parameters()
                if "pointer" in n and p.grad is not None
            ],
            "side": [
                p
                for n, p in model.named_parameters()
                if "side" in n and p.grad is not None
            ],
            "stop": [
                p
                for n, p in model.named_parameters()
                if "stop" in n and p.grad is not None
            ],
            "encoder": [
                p
                for n, p in model.named_parameters()
                if "encoder" in n and p.grad is not None
            ],
        }

        for name, params in param_groups.items():
            if params:
                torch.nn.utils.clip_grad_norm_(params, max_norm)

    if args.trainable_catalog:
        try:
            model.set_catalog_embeddings(cat_emb)
        except Exception as e:
            print(f"Warning: failed to register trainable catalog embeddings: {e}")

    pointer_params = [p for n, p in model.named_parameters() if "pointer" in n]
    other_params = [p for n, p in model.named_parameters() if "pointer" not in n]

    optimizer = torch.optim.AdamW(
        [
            {"params": other_params, "lr": args.lr},
            {"params": pointer_params, "lr": args.lr * 0.5},
        ],
        lr=args.lr,
    )

    total_steps = max(1, len(dl) * int(args.epochs))
    try:
        warmup_steps = int(len(dl) * int(args.lr_warmup_epochs))
    except Exception:
        warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    metric_set_f1 = WindowedSetF1Metric(window_size=int(args.setf1_window_steps))

    model.train()
    step = 0
    best_loss = float("inf")
    best_epoch = 0
    best_metrics: Optional[Dict[str, float]] = None
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    for epoch in range(args.epochs):
        last_metrics = None

        epoch_train_examples = 0
        train_total_sum = 0.0
        train_pl_sum = 0.0
        train_sl_sum = 0.0
        train_stl_sum = 0.0
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

            with torch.no_grad():
                m = target_account_idx >= 0
                if m.any():
                    max_idx = int(target_account_idx[m].max().item())
                    if max_idx >= catalog_size:
                        raise ValueError(
                            f"Account index {max_idx} >= catalog size {catalog_size}. "
                            "Check that dataset indices align with accounts artifact order."
                        )

            optimizer.zero_grad(set_to_none=True)

            retr_mem = (
                _build_retrieval_memory_batch(input_ids, attention_mask)
                if retrieval_enabled
                else None
            )
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

            pointer_probs = torch.softmax(outputs["pointer_logits"], dim=-1)
            side_probs = torch.softmax(outputs["side_logits"], dim=-1)

            pl = hierarchical_pointer_loss(
                outputs["pointer_logits"],
                target_account_idx,
                hierarchy=hierarchy,
                alpha=0.7,
                ignore_index=-1
            )

            sl = side_loss(outputs["side_logits"], target_side_id, ignore_index=-1)
            stl = stop_loss(outputs["stop_logits"], target_stop_id, ignore_index=-1)

            cov = coverage_penalty(outputs["pointer_logits"], probs=pointer_probs)

            flow = flow_aux_loss(
                outputs["pointer_logits"],
                outputs["side_logits"],
                debit_indices,
                debit_weights,
                credit_indices,
                credit_weights,
                pointer_probs=pointer_probs,
                side_probs=side_probs,
            )

            loss_weights = get_loss_weights(epoch, args.epochs)

            total = (
                pl
                + sl
                + stl
                + cov * loss_weights["coverage"]
                + flow * loss_weights["flow"]
            )

            def _finite(x: torch.Tensor) -> bool:
                try:
                    return bool(torch.isfinite(x).all())
                except Exception:
                    return True

            all_finite = (
                _finite(total)
                and _finite(pl)
                and _finite(sl)
                and _finite(stl)
                and _finite(cov)
                and _finite(flow)
            )
            if not all_finite:
                print(
                    "[anomaly] Non-finite loss component detected; skipping optimizer step."
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            try:
                B = int(input_ids.size(0))
            except Exception:
                B = 1
            epoch_train_examples += B
            train_total_sum += float(total.item()) * B
            train_pl_sum += float(pl.item()) * B
            train_sl_sum += float(sl.item()) * B
            train_stl_sum += float(stl.item()) * B
            total.backward()
            clip_grad_by_component(model, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            metric_set_f1.update_state(
                outputs["pointer_logits"].detach().cpu(),
                outputs["side_logits"].detach().cpu(),
                target_account_idx.detach().cpu(),
                target_side_id.detach().cpu(),
                target_stop_id.detach().cpu(),
            )

            last_metrics = {
                "loss": total.item(),
                "pointer_loss": pl.item(),
                "side_loss": sl.item(),
                "stop_loss": stl.item(),
                "coverage_penalty": cov.item(),
                "flow_loss": flow.item(),
                "set_f1": metric_set_f1.result().item()
                if step % 100 == 0
                else (last_metrics["set_f1"] if last_metrics else 0.0),
            }

        train_ex_den = float(max(epoch_train_examples, 1))
        train_logs = {
            "epoch": epoch,
            "train/loss": (train_total_sum / train_ex_den),
            "train/pointer_loss": (train_pl_sum / train_ex_den),
            "train/side_loss": (train_sl_sum / train_ex_den),
            "train/stop_loss": (train_stl_sum / train_ex_den),
        }

        if dl_val is not None:
            model.eval()
            val_steps_done = 0

            val_examples = 0
            val_total_sum = 0.0
            val_pl_sum = 0.0
            val_sl_sum = 0.0
            val_stl_sum = 0.0
            val_metric = SetF1Metric()

            retr_query_ids: List[str] = []
            retr_topk_indices: List[List[int]] = []
            first_heatmaps_logged = False
            with torch.no_grad():
                for v_features, v_targets in dl_val:
                    input_ids_v = v_features["input_ids"].to(device)
                    attn_v = v_features["attention_mask"].to(device)
                    prev_acc_v = v_features["prev_account_idx"].to(device)
                    prev_side_v = v_features["prev_side_id"].to(device)
                    cond_num_v = v_features["cond_numeric"].to(device)
                    curr_v = v_features["currency"]
                    jtyp_v = v_features["journal_entry_type"]
                    tgt_acc_v = v_targets["target_account_idx"].to(device)
                    tgt_side_v = v_targets["target_side_id"].to(device)
                    tgt_stop_v = v_targets["target_stop_id"].to(device)
                    deb_idx_v = v_targets["debit_indices"].to(device)
                    deb_w_v = v_targets["debit_weights"].to(device)
                    cre_idx_v = v_targets["credit_indices"].to(device)
                    cre_w_v = v_targets["credit_weights"].to(device)

                    retr_mem_v = None
                    nbrs_v = None
                    if retrieval_enabled:
                        nbrs_mem = _neighbors_and_memory_batch(input_ids_v, attn_v)
                        if nbrs_mem is not None:
                            nbrs_v, retr_mem_v = nbrs_mem
                    outs_v = model(
                        input_ids=input_ids_v,
                        attention_mask=attn_v,
                        prev_account_idx=prev_acc_v,
                        prev_side_id=prev_side_v,
                        catalog_embeddings=cat_emb,
                        retrieval_memory=retr_mem_v,
                        cond_numeric=cond_num_v,
                        currency=curr_v,
                        journal_entry_type=jtyp_v,
                        return_retrieval_weights=bool(
                            args.log_retrieval_heatmaps
                            and retrieval_enabled
                            and not first_heatmaps_logged
                        ),
                    )
                    pp_v = torch.softmax(outs_v["pointer_logits"], dim=-1)
                    sp_v = torch.softmax(outs_v["side_logits"], dim=-1)
                    pl_v = pointer_loss(
                        outs_v["pointer_logits"], tgt_acc_v, ignore_index=-1
                    )
                    sl_v = side_loss(outs_v["side_logits"], tgt_side_v, ignore_index=-1)
                    stl_v = stop_loss(
                        outs_v["stop_logits"], tgt_stop_v, ignore_index=-1
                    )
                    cov_v = coverage_penalty(outs_v["pointer_logits"], probs=pp_v)
                    flow_v = flow_aux_loss(
                        outs_v["pointer_logits"],
                        outs_v["side_logits"],
                        deb_idx_v,
                        deb_w_v,
                        cre_idx_v,
                        cre_w_v,
                        pointer_probs=pp_v,
                        side_probs=sp_v,
                    )

                    loss_weights_v = get_loss_weights(epoch, args.epochs)
                    total_v = (
                        pl_v
                        + sl_v
                        + stl_v
                        + cov_v * loss_weights_v["coverage"]
                        + flow_v * loss_weights_v["flow"]
                    )

                    Bv = int(input_ids_v.size(0))
                    val_examples += Bv
                    val_total_sum += float(total_v.item()) * Bv
                    val_pl_sum += float(pl_v.item()) * Bv
                    val_sl_sum += float(sl_v.item()) * Bv
                    val_stl_sum += float(stl_v.item()) * Bv

                    val_metric.update_state(
                        outs_v["pointer_logits"].detach().cpu(),
                        outs_v["side_logits"].detach().cpu(),
                        tgt_acc_v.detach().cpu(),
                        tgt_side_v.detach().cpu(),
                        tgt_stop_v.detach().cpu(),
                    )

                    if retrieval_enabled and nbrs_v is not None:
                        retr_topk_indices.extend([list(x) for x in nbrs_v])
                        qids_batch = [
                            str(x)
                            for x in v_features.get(
                                "journal_entry_id", [""] * len(nbrs_v)
                            )
                        ]
                        retr_query_ids.extend(qids_batch)

                    val_steps_done += 1
                    if int(args.max_val_batches) > 0 and val_steps_done >= int(
                        args.max_val_batches
                    ):
                        break

            ex_den = float(max(val_examples, 1))
            val_logs = {
                "epoch": epoch,
                "val/loss": val_total_sum / ex_den,
                "val/pointer_loss": val_pl_sum / ex_den,
                "val/side_loss": val_sl_sum / ex_den,
                "val/stop_loss": val_stl_sum / ex_den,
                "val/set_f1": float(val_metric.result().item())
                if hasattr(val_metric, "result")
                else 0.0,
            }

            if wandb_enabled:
                try:
                    _combined_logs = dict(train_logs)
                    _combined_logs.update(val_logs)
                    wandb.log(_combined_logs)
                except Exception as _vlog_e:
                    print(f"[wandb] Warning: failed logging epoch metrics: {_vlog_e}")

            current_selection_loss = float(val_logs["val/loss"])
            if current_selection_loss < best_val_loss:
                best_val_loss = current_selection_loss
                best_epoch = epoch + 1
                best_metrics = dict(val_logs)
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_state_path = os.path.join(tmpdir, "model_state.pt")
                    torch.save({"model": model.state_dict()}, model_state_path)

                    config_path = os.path.join(tmpdir, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(
                            {
                                "encoder": args.encoder,
                                "hidden_dim": args.hidden_dim,
                                "max_lines": args.max_lines,
                                "max_length": args.max_length,
                                "temperature": float(args.pointer_temp),
                                "pointer_scale_init": float(args.pointer_scale_init),
                                "learnable_pointer_scale": bool(
                                    args.learnable_pointer_scale
                                ),
                                "pointer_use_norm": not args.no_pointer_norm,
                                "trainable_catalog": bool(args.trainable_catalog),
                                "retrieval_index_dir": args.retrieval_index_dir,
                                "retrieval_top_k": int(args.retrieval_top_k),
                                "retrieval_use_cls": bool(args.retrieval_use_cls),
                            },
                            f,
                            indent=2,
                        )

                    accounts_path = os.path.join(tmpdir, "accounts_artifact.json")
                    artifact_data = load_json_from_uri(accounts_artifact_path)
                    with open(accounts_path, "w", encoding="utf-8") as f:
                        json.dump(artifact_data, f, indent=2, ensure_ascii=False)

                    catalog_emb_path = os.path.join(tmpdir, "catalog_embeddings.pt")
                    try:
                        cat_src = (
                            model.catalog_param.detach().cpu()
                            if (
                                bool(args.trainable_catalog)
                                and hasattr(model, "catalog_param")
                            )
                            else cat_emb.detach().cpu()
                        )
                        torch.save({"catalog_embeddings": cat_src}, catalog_emb_path)
                    except Exception as _ce_save_e:
                        print(
                            f"Warning: failed to save catalog embeddings: {_ce_save_e}"
                        )

                    metadata_path = os.path.join(tmpdir, "training_metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(
                            {
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
                                    "retrieval_top_k": args.retrieval_top_k,
                                    "retrieval_use_cls": args.retrieval_use_cls,
                                    "limit": args.limit,
                                    "seed": args.seed,
                                    "setf1_window_steps": args.setf1_window_steps,
                                    "early_stopping_patience": args.early_stopping_patience,
                                    "log_hist_every": args.log_hist_every,
                                },
                                "metrics": {
                                    "val_loss": best_val_loss,
                                    "val_set_f1": val_logs.get("val/set_f1", 0.0),
                                },
                            },
                            f,
                            indent=2,
                        )

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
                            ("catalog_embeddings.pt", catalog_emb_path),
                        ]:
                            blob = client.bucket(bucket_name).blob(
                                f"{prefix.rstrip('/')}/{fname}"
                            )
                            blob.upload_from_filename(fpath)
                        print(
                            f"New best (epoch {best_epoch}, val_loss={best_val_loss:.4f}) uploaded to {args.output_dir}"
                        )
                    else:
                        for fname, fpath in [
                            ("model_state.pt", model_state_path),
                            ("config.json", config_path),
                            ("accounts_artifact.json", accounts_path),
                            ("training_metadata.json", metadata_path),
                            ("catalog_embeddings.pt", catalog_emb_path),
                        ]:
                            shutil.copy2(fpath, os.path.join(args.output_dir, fname))
                        print(
                            f"New best (epoch {best_epoch}, val_loss={best_val_loss:.4f}) saved to {args.output_dir}"
                        )
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
            model.train()

        if dl_val is None:
            if wandb_enabled:
                try:
                    wandb.log(train_logs)
                except Exception as _tlog_e:
                    print(
                        f"[wandb] Warning: failed logging train epoch metrics: {_tlog_e}"
                    )
            if last_metrics is None:
                last_metrics = {
                    "loss": 0.0,
                    "pointer_loss": 0.0,
                    "side_loss": 0.0,
                    "stop_loss": 0.0,
                    "coverage_penalty": 0.0,
                    "flow_loss": 0.0,
                    "set_f1": metric_set_f1.result().item()
                    if hasattr(metric_set_f1, "result")
                    else 0.0,
                }
            current_loss = float(last_metrics.get("loss", float("inf")))
            if current_loss < best_loss:
                best_loss = current_loss
                best_epoch = epoch + 1
                best_metrics = dict(last_metrics)
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_state_path = os.path.join(tmpdir, "model_state.pt")
                    torch.save({"model": model.state_dict()}, model_state_path)
                    config_path = os.path.join(tmpdir, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(
                            {
                                "encoder": args.encoder,
                                "hidden_dim": args.hidden_dim,
                                "max_lines": args.max_lines,
                                "max_length": args.max_length,
                                "temperature": float(args.pointer_temp),
                                "pointer_scale_init": float(args.pointer_scale_init),
                                "learnable_pointer_scale": bool(
                                    args.learnable_pointer_scale
                                ),
                                "pointer_use_norm": not args.no_pointer_norm,
                                "trainable_catalog": bool(args.trainable_catalog),
                                "retrieval_index_dir": args.retrieval_index_dir,
                                "retrieval_top_k": int(args.retrieval_top_k),
                                "retrieval_use_cls": bool(args.retrieval_use_cls),
                            },
                            f,
                            indent=2,
                        )
                    accounts_path = os.path.join(tmpdir, "accounts_artifact.json")
                    artifact_data = load_json_from_uri(accounts_artifact_path)
                    with open(accounts_path, "w", encoding="utf-8") as f:
                        json.dump(artifact_data, f, indent=2, ensure_ascii=False)
                    catalog_emb_path = os.path.join(tmpdir, "catalog_embeddings.pt")
                    try:
                        cat_src = (
                            model.catalog_param.detach().cpu()
                            if (
                                bool(args.trainable_catalog)
                                and hasattr(model, "catalog_param")
                            )
                            else cat_emb.detach().cpu()
                        )
                        torch.save({"catalog_embeddings": cat_src}, catalog_emb_path)
                    except Exception as _ce_save_e:
                        print(
                            f"Warning: failed to save catalog embeddings: {_ce_save_e}"
                        )
                    metadata_path = os.path.join(tmpdir, "training_metadata.json")
                    with open(metadata_path, "w") as f:
                        json.dump(
                            {
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
                                    "retrieval_top_k": args.retrieval_top_k,
                                    "retrieval_use_cls": args.retrieval_use_cls,
                                    "limit": args.limit,
                                    "seed": args.seed,
                                    "setf1_window_steps": args.setf1_window_steps,
                                    "early_stopping_patience": args.early_stopping_patience,
                                    "log_hist_every": args.log_hist_every,
                                },
                                "metrics": best_metrics,
                            },
                            f,
                            indent=2,
                        )

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
                            ("catalog_embeddings.pt", catalog_emb_path),
                        ]:
                            blob = client.bucket(bucket_name).blob(
                                f"{prefix.rstrip('/')}/{fname}"
                            )
                            blob.upload_from_filename(fpath)
                        print(
                            f"New best (epoch {best_epoch}, loss={best_loss:.4f}) uploaded to {args.output_dir}"
                        )
                    else:
                        for fname, fpath in [
                            ("model_state.pt", model_state_path),
                            ("config.json", config_path),
                            ("accounts_artifact.json", accounts_path),
                            ("training_metadata.json", metadata_path),
                            ("catalog_embeddings.pt", catalog_emb_path),
                        ]:
                            shutil.copy2(fpath, os.path.join(args.output_dir, fname))
                        print(
                            f"New best (epoch {best_epoch}, loss={best_loss:.4f}) saved to {args.output_dir}"
                        )

        if dl_val is not None and int(args.early_stopping_patience) > 0:
            if epochs_since_improvement >= int(args.early_stopping_patience):
                print(
                    f"Early stopping: no val/loss improvement for {epochs_since_improvement} epochs at epoch {epoch + 1}."
                )
                break

    if wandb_enabled:
        try:
            wandb.summary.update(
                {
                    "final_epoch": args.epochs,
                    "final_step": step,
                    "final_set_f1": metric_set_f1.result().item(),
                    "output_directory": args.output_dir,
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss
                    if best_val_loss < float("inf")
                    else None,
                    "best_train_loss": best_loss if best_loss < float("inf") else None,
                    "best_set_f1": (
                        best_metrics.get("val/set_f1")
                        if isinstance(best_metrics, dict)
                        else None
                    )
                    if best_metrics is not None
                    else None,
                }
            )
            wandb.finish()
        except Exception as e:
            print(f"Warning: Failed to update wandb summary: {e}")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Run name: {run_name}")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
