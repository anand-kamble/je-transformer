#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict

import torch

from inference.beam_decode import beam_search_decode
from models.catalog_encoder import CatalogEncoder
from models.je_model_torch import JEModel
from models.tokenizer import DescriptionTokenizer


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
	return encoder({"number": number, "name": name, "nature": nature}).to(device=device, dtype=torch.float32)


def main() -> None:
	parser = argparse.ArgumentParser(description="Quick inference test for JE model (PyTorch)")
	parser.add_argument("--description", type=str, default="CAPITAL SOCIAL", help="Input description to infer on")
	parser.add_argument("--accounts-artifact", type=str, default="./out_small_ingest/artifacts/accounts_20251109-230650.json", help="Accounts JSON (local or gs://)")
	parser.add_argument("--checkpoint", type=str, default="./out_small/model_state.pt", help="Optional model_state .pt to load")
	parser.add_argument("--encoder", type=str, default="prajjwal1/bert-tiny")
	parser.add_argument("--hidden-dim", type=int, default=128)
	parser.add_argument("--max-length", type=int, default=64)
	parser.add_argument("--max-lines", type=int, default=4)
	parser.add_argument("--beam-size", type=int, default=20)
	parser.add_argument("--alpha", type=float, default=0.7)
	parser.add_argument("--tau", type=float, default=0.2)
	parser.add_argument("--currency", type=str, default="mxn", help="Optional currency string")
	parser.add_argument("--je-type", type=str, default="general", help="Optional journal entry type")
	parser.add_argument("--debug", action="store_true", default=True, help="Print debug shapes and logits")
	parser.add_argument("--min-lines", type=int, default=1, help="Force at least this many lines via fallback if beam returns empty")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

	# Tokenize description
	tok = DescriptionTokenizer(model_name_or_path=args.encoder, max_length=args.max_length)
	desc = args.description.strip()
	batch = tok.tokenize_batch([desc])
	input_ids = torch.tensor([batch["input_ids"][0]], dtype=torch.long, device=device)
	attention_mask = torch.tensor([batch["attention_mask"][0]], dtype=torch.long, device=device)

	# Catalog embeddings
	artifact = load_json_from_uri(args.accounts_artifact)
	cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)
	if args.debug:
		print(f"[infer] catalog_embeddings shape={tuple(cat_emb.shape)} num_accounts={int(cat_emb.shape[0])}")

	# Model
	model = JEModel(
		encoder_loc=args.encoder,
		hidden_dim=args.hidden_dim,
		max_lines=args.max_lines,
		temperature=1.0,
	).to(device)
	if args.checkpoint:
		state = torch.load(args.checkpoint, map_location=device)
		state = state.get("model", state)
		try:
			model.load_state_dict(state, strict=True)
			if args.debug:
				print("[infer] Loaded checkpoint strictly (no missing/unexpected keys).")
		except Exception as e:
			print(f"[infer] Strict load failed: {e}. Falling back to strict=False")
			model.load_state_dict(state, strict=False)
	else:
		if args.debug:
			print("[infer] No checkpoint provided; using randomly initialized model.")
	model.eval()

	# Conditioning defaults (zeros)
	cond_numeric = torch.zeros((1, 8), dtype=torch.float32, device=device)
	currency = [args.currency]
	je_type = [args.je_type]

	# Retrieval memory default to small zeros
	retr_mem = torch.zeros((1, args.hidden_dim), dtype=torch.float32, device=device)

	# Optional: print first-step logits to see if stop dominates
	if args.debug:
		with torch.no_grad():
			prev_acc0 = torch.full((1, args.max_lines), -1, dtype=torch.long, device=device)
			prev_side0 = torch.full((1, args.max_lines), -1, dtype=torch.long, device=device)
			outs0 = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				prev_account_idx=prev_acc0,
				prev_side_id=prev_side0,
				catalog_embeddings=cat_emb,
				retrieval_memory=retr_mem,
				cond_numeric=cond_numeric,
				currency=currency,
				journal_entry_type=je_type,
			)
			ptr0 = torch.softmax(outs0["pointer_logits"][0, 0], dim=-1)
			side0 = torch.softmax(outs0["side_logits"][0, 0], dim=-1)
			stop0 = torch.softmax(outs0["stop_logits"][0, 0], dim=-1)
			topv, topi = torch.topk(ptr0, k=min(5, ptr0.numel()))
			print(f"[infer] step0 stop_probs={stop0.tolist()} side_probs={side0.tolist()}")
			print(f"[infer] step0 pointer topk indices={topi.tolist()} probs={topv.tolist()}")

	# Decode
	with torch.no_grad():
		cands = beam_search_decode(
			model=model,
			input_ids=input_ids,
			attention_mask=attention_mask,
			catalog_embeddings=cat_emb,
			retrieval_memory=retr_mem,
			cond_numeric=cond_numeric,
			currency=currency,
			journal_entry_type=je_type,
			beam_size=args.beam_size,
			alpha=args.alpha,
			max_lines=args.max_lines,
			tau=args.tau,
		)
	if args.debug:
		print(f"[infer] beam candidates returned={len(cands) if cands else 0}")
	# Fallback: if beam returns empty or length 0 and min-lines requested, force 1 line greedily
	if (not cands or (cands and cands[0].get("length", 0) == 0)) and int(args.min_lines) > 0:
		with torch.no_grad():
			prev_acc0 = torch.full((1, args.max_lines), -1, dtype=torch.long, device=device)
			prev_side0 = torch.full((1, args.max_lines), -1, dtype=torch.long, device=device)
			outs0 = model(
				input_ids=input_ids,
				attention_mask=attention_mask,
				prev_account_idx=prev_acc0,
				prev_side_id=prev_side0,
				catalog_embeddings=cat_emb,
				retrieval_memory=retr_mem,
				cond_numeric=cond_numeric,
				currency=currency,
				journal_entry_type=je_type,
			)
			ptr0 = torch.softmax(outs0["pointer_logits"][0, 0], dim=-1)
			side0 = torch.softmax(outs0["side_logits"][0, 0], dim=-1)
			acc_idx = int(torch.argmax(ptr0).item()) if ptr0.numel() > 0 else -1
			side_idx = int(torch.argmax(side0).item()) if side0.numel() > 0 else 0
			if acc_idx >= 0:
				cands = [{"accounts": [acc_idx], "sides": [side_idx], "length": 1, "score": float(torch.log(torch.clamp(ptr0[acc_idx], min=1e-9)).item()), "prob": float(ptr0[acc_idx].item()), "logprob": float(torch.log(torch.clamp(ptr0[acc_idx], min=1e-9)).item())}]
			else:
				cands = []
	if not cands:
		print(json.dumps({"candidates": []}, indent=2))
		return
	# Print top-3
	top = cands[:3]
	print(json.dumps({"candidates": top}, indent=2))


if __name__ == "__main__":
	main()


