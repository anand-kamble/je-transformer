#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import io
from typing import Any, Dict, Optional

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
	return encoder({"number": number, "name": name, "nature": nature}).to(device=device, dtype=torch.float32).detach()


def _uri_join(base: str, name: str) -> str:
	if base.startswith("gs://"):
		return f"{base.rstrip('/')}/{name}"
	return os.path.join(base, name)


def _uri_dirname(uri: str) -> str:
	if uri.startswith("gs://"):
		# Trim last path segment
		return uri.rsplit("/", 1)[0]
	return os.path.dirname(uri)


def _uri_exists(uri: str) -> bool:
	if uri.startswith("gs://"):
		try:
			from google.cloud import storage
			client = storage.Client()
			_, path = uri.split("gs://", 1)
			bucket_name, blob_path = path.split("/", 1)
			blob = client.bucket(bucket_name).blob(blob_path)
			return blob.exists(client)
		except Exception:
			return False
	return os.path.exists(uri)


def load_torch_from_uri(uri: str, map_location: Optional[torch.device] = None) -> Any:
	if uri.startswith("gs://"):
		from google.cloud import storage
		client = storage.Client()
		_, path = uri.split("gs://", 1)
		bucket_name, blob_path = path.split("/", 1)
		blob = client.bucket(bucket_name).blob(blob_path)
		data = blob.download_as_bytes()
		return torch.load(io.BytesIO(data), map_location=map_location)
	return torch.load(uri, map_location=map_location)

def _list_runs_local(root: str):
	try:
		entries = []
		for name in os.listdir(root):
			p = os.path.join(root, name)
			if os.path.isdir(p):
				try:
					mtime = os.path.getmtime(p)
				except Exception:
					mtime = 0.0
				entries.append((name, mtime))
		return sorted(entries, key=lambda x: x[1], reverse=True)
	except FileNotFoundError:
		return []

def _list_runs_gcs(root: str):
	from collections import defaultdict
	try:
		from google.cloud import storage
		client = storage.Client()
		_, path = root.split("gs://", 1)
		bucket_name, prefix = path.split("/", 1)
		if not prefix.endswith("/"):
			prefix = prefix + "/"
		bucket = client.bucket(bucket_name)
		# Gather first-level subdirs and their latest updated time
		latest_ts_by_run = defaultdict(float)
		for blob in client.list_blobs(bucket, prefix=prefix):
			# Expect names like prefix + run_name + "/..."
			name = blob.name
			if not name.startswith(prefix):
				continue
			rem = name[len(prefix):]
			parts = rem.split("/", 1)
			if not parts or parts[0] == "":
				continue
			rn = parts[0]
			if getattr(blob, "updated", None):
				ts = blob.updated.timestamp()
				if ts > latest_ts_by_run[rn]:
					latest_ts_by_run[rn] = ts
		runs = [(rn, ts) for rn, ts in latest_ts_by_run.items()]
		return sorted(runs, key=lambda x: x[1], reverse=True)
	except Exception:
		return []

def list_runs(root: str):
	if root.startswith("gs://"):
		return _list_runs_gcs(root)
	return _list_runs_local(root)

def main() -> None:
	parser = argparse.ArgumentParser(description="Quick inference test for JE model (PyTorch)")
	parser.add_argument("--description", type=str, default="CAPITAL SOCIAL", help="Input description to infer on")
	parser.add_argument("--accounts-artifact", type=str, default=os.environ.get("ACCOUNTS_ARTIFACT"), help="Accounts JSON (local dir/file or gs://). If omitted, will try to load accounts_artifact.json next to checkpoint.")
	parser.add_argument("--checkpoint", type=str, default="./out_small/model_state.pt", help="Path to model_state.pt or a directory (local or gs://)")
	parser.add_argument("--runs-root", type=str, default=os.environ.get("OUTPUT_DIR", "./out_small"), help="Root directory or gs:// where run subdirectories are saved.")
	parser.add_argument("--run-name", type=str, default=None, help="Which run subdirectory to use under runs-root.")
	parser.add_argument("--use-latest-run", action="store_true", help="Automatically pick the most recently updated run under runs-root.")
	parser.add_argument("--list-runs", action="store_true", help="List available runs under runs-root and exit.")
	parser.add_argument("--encoder", type=str, default="prajjwal1/bert-tiny")
	parser.add_argument("--hidden-dim", type=int, default=512)
	parser.add_argument("--max-length", type=int, default=64)
	parser.add_argument("--max-lines", type=int, default=40)
	parser.add_argument("--beam-size", type=int, default=20)
	parser.add_argument("--alpha", type=float, default=0.7)
	parser.add_argument("--tau", type=float, default=0.2)
	parser.add_argument("--currency", type=str, default="mxn", help="Optional currency string")
	parser.add_argument("--je-type", type=str, default="general", help="Optional journal entry type")
	parser.add_argument("--debug", action="store_true", default=False, help="Print debug shapes and logits")
	parser.add_argument("--min-lines", type=int, default=1, help="Force at least this many lines via fallback if beam returns empty")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

	# Optional: list runs and exit
	if args.list_runs:
		runs = list_runs(args.runs_root)
		if not runs:
			print("No runs found.")
			return
		for name, ts in runs:
			print(f"{name}\t{ts}")
		return

	# Resolve checkpoint path/URI and base directory/URI
	ckpt_uri = args.checkpoint
	# If a run name is provided or latest-run requested, construct checkpoint from runs-root/run-name
	if args.run_name or args.use_latest_run:
		base_uri_sel = None
		if args.run_name:
			base_uri_sel = _uri_join(args.runs_root, args.run_name)
		else:
			runs = list_runs(args.runs_root)
			if runs:
				base_uri_sel = _uri_join(args.runs_root, runs[0][0])
			else:
				raise FileNotFoundError(f"No runs found under {args.runs_root}")
		ckpt_uri = _uri_join(base_uri_sel, "model_state.pt")
	if ckpt_uri.startswith("gs://"):
		if ckpt_uri.endswith(".pt"):
			pass
		else:
			ckpt_uri = _uri_join(ckpt_uri, "model_state.pt")
		base_uri = _uri_dirname(ckpt_uri)
	else:
		if os.path.isdir(ckpt_uri):
			ckpt_uri = os.path.join(ckpt_uri, "model_state.pt")
		base_uri = os.path.dirname(ckpt_uri)

	# Load config.json from alongside checkpoint if present and use to override core params
	config_uri = _uri_join(base_uri, "config.json")
	if _uri_exists(config_uri):
		try:
			cfg = load_json_from_uri(config_uri)
			# Override core hyperparameters from training config
			args.encoder = cfg.get("encoder", args.encoder)
			args.hidden_dim = int(cfg.get("hidden_dim", args.hidden_dim))
			args.max_lines = int(cfg.get("max_lines", args.max_lines))
			args.max_length = int(cfg.get("max_length", args.max_length))
			if args.debug:
				print(f"[infer] Loaded config from {config_uri}")
		except Exception as e:
			if args.debug:
				print(f"[infer] Failed to load config.json from {config_uri}: {e}")

	# Tokenize description
	tok = DescriptionTokenizer(model_name_or_path=args.encoder, max_length=args.max_length)
	desc = args.description.strip()
	batch = tok.tokenize_batch([desc])
	input_ids = torch.tensor([batch["input_ids"][0]], dtype=torch.long, device=device)
	attention_mask = torch.tensor([batch["attention_mask"][0]], dtype=torch.long, device=device)

	# Catalog embeddings
	accounts_uri: Optional[str] = args.accounts_artifact
	if not accounts_uri:
		cand = _uri_join(base_uri, "accounts_artifact.json")
		if _uri_exists(cand):
			accounts_uri = cand
		else:
			raise FileNotFoundError("Accounts artifact not provided and accounts_artifact.json not found next to checkpoint.")
	artifact = load_json_from_uri(accounts_uri)
	cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)
	if args.debug:
		print(f"[infer] catalog_embeddings shape={tuple(cat_emb.shape)} num_accounts={int(cat_emb.shape[0])} from={accounts_uri}")

	# Model
	model = JEModel(
		encoder_loc=args.encoder,
		hidden_dim=args.hidden_dim,
		max_lines=args.max_lines,
		temperature=1.0,
	).to(device)
 
	if args.trainable_catalog:
	    model.set_catalog_embeddings(cat_emb)
     
	if ckpt_uri:
		state = load_torch_from_uri(ckpt_uri, map_location=device)
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
		print(f"[infer] fallback to min-lines={args.min_lines}")
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


