from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import onnx
import onnxruntime as ort
import torch

from models.catalog_encoder import CatalogEncoder
from models.je_model_torch import JEModel


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
    parser = argparse.ArgumentParser(description="Export JEModel to ONNX and optionally validate with ONNXRuntime")
    parser.add_argument("--output-onnx", type=str, required=True)
    parser.add_argument("--accounts-artifact", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False, help="Optional state_dict checkpoint path or gs://")
    parser.add_argument("--encoder", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-lines", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu")

    model = JEModel(encoder_loc=args.encoder, hidden_dim=args.hidden_dim, max_lines=args.max_lines, temperature=1.0).to(device)

    if args.checkpoint:
        # load from local or gs://
        if args.checkpoint.startswith("gs://"):
            from google.cloud import storage

            client = storage.Client()
            _, path = args.checkpoint.split("gs://", 1)
            bucket_name, blob_path = path.split("/", 1)
            blob = client.bucket(bucket_name).blob(blob_path)
            tmp = "/tmp/model_state.pt"
            blob.download_to_filename(tmp)
            sd = torch.load(tmp, map_location=device)
        else:
            sd = torch.load(args.checkpoint, map_location=device)
        state = sd.get("model", sd)
        model.load_state_dict(state, strict=False)

    model.eval()

    # Catalog embeddings for export
    artifact = load_json_from_uri(args.accounts_artifact)
    cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)

    # Dummy inputs (dynamic axes will relax sizes)
    B = 2
    L = args.max_length
    T = args.max_lines
    C = cat_emb.shape[0]
    input_ids = torch.ones((B, L), dtype=torch.long, device=device)
    attention_mask = torch.ones((B, L), dtype=torch.long, device=device)
    prev_account_idx = torch.full((B, T), -1, dtype=torch.long, device=device)
    prev_side_id = torch.full((B, T), -1, dtype=torch.long, device=device)
    cond_numeric = torch.zeros((B, 8), dtype=torch.float32, device=device)
    # For ONNX, pass categorical ids as tensors (not strings)
    currency_ids = torch.zeros((B,), dtype=torch.long, device=device)
    type_ids = torch.zeros((B,), dtype=torch.long, device=device)

    # Wrapper for export to map to model signature
    class Wrapper(torch.nn.Module):
        def __init__(self, m: JEModel, cat: torch.Tensor) -> None:
            super().__init__()
            self.m = m
            self.cat = cat

        def forward(
            self,
            input_ids, attention_mask, prev_account_idx, prev_side_id, cond_numeric, currency_ids, type_ids
        ):
            out = self.m(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prev_account_idx=prev_account_idx,
                prev_side_id=prev_side_id,
                catalog_embeddings=self.cat,
                retrieval_memory=None,
                cond_numeric=cond_numeric,
                currency=currency_ids,
                journal_entry_type=type_ids,
            )
            return out["pointer_logits"], out["side_logits"], out["stop_logits"]

    wrapped = Wrapper(model, cat_emb)

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "prev_account_idx": {0: "batch", 1: "time"},
        "prev_side_id": {0: "batch", 1: "time"},
        "cond_numeric": {0: "batch"},
        "currency_ids": {0: "batch"},
        "type_ids": {0: "batch"},
        "pointer_logits": {0: "batch", 1: "time", 2: "catalog"},
        "side_logits": {0: "batch", 1: "time"},
        "stop_logits": {0: "batch", 1: "time"},
    }

    torch.onnx.export(
        wrapped,
        (input_ids, attention_mask, prev_account_idx, prev_side_id, cond_numeric, currency_ids, type_ids),
        args.output_onnx,
        input_names=[
            "input_ids",
            "attention_mask",
            "prev_account_idx",
            "prev_side_id",
            "cond_numeric",
            "currency_ids",
            "type_ids",
        ],
        output_names=["pointer_logits", "side_logits", "stop_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=int(args.opset),
    )
    print(f"ONNX exported to {args.output_onnx}")

    if args.validate:
        sess = ort.InferenceSession(args.output_onnx, providers=["CPUExecutionProvider"])
        outs = sess.run(
            None,
            {
                "input_ids": input_ids.cpu().numpy(),
                "attention_mask": attention_mask.cpu().numpy(),
                "prev_account_idx": prev_account_idx.cpu().numpy(),
                "prev_side_id": prev_side_id.cpu().numpy(),
                "cond_numeric": cond_numeric.cpu().numpy(),
                "currency_ids": currency_ids.cpu().numpy(),
                "type_ids": type_ids.cpu().numpy(),
            },
        )
        print("ONNXRuntime validation outputs shapes:", [tuple(o.shape) for o in outs])


if __name__ == "__main__":
    main()



