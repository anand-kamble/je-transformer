
from __future__ import annotations

import argparse
import json
import os
import io
import math
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

import torch

from inference.beam_decode import beam_search_decode
from models.catalog_encoder import CatalogEncoder
from models.je_model_torch import JEModel
from models.tokenizer import DescriptionTokenizer


def _dbg(enabled: bool, msg: str) -> None:
	if enabled:
		print(f"[infer][debug] {msg}")


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
		except Exception as e:
			_dbg(True, f"GCS exists() check failed for {uri}: {type(e).__name__}: {e}")
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
		return torch.load(io.BytesIO(data), map_location=map_location, weights_only=False)
	return torch.load(uri, map_location=map_location, weights_only=False)

def _list_runs_local(root: str, debug: bool = False) -> List[Tuple[str, float]]:
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
		_dbg(debug, f"Local runs discovered under {root}: {len(entries)}")
		return sorted(entries, key=lambda x: x[1], reverse=True)
	except FileNotFoundError:
		_dbg(debug, f"Local runs root not found: {root}")
		return []

def _list_runs_gcs(root: str, debug: bool = False) -> List[Tuple[str, float]]:
	from collections import defaultdict
	try:
		from google.cloud import storage
		client = storage.Client()
		_, path = root.split("gs://", 1)
		bucket_name, prefix = path.split("/", 1)
		if not prefix.endswith("/"):
			prefix = prefix + "/"
		bucket = client.bucket(bucket_name)
		_dbg(debug, f"Listing GCS runs: bucket={bucket_name} prefix={prefix}")
		
		latest_ts_by_run = defaultdict(float)
		count = 0
		for blob in client.list_blobs(bucket, prefix=prefix):
			count += 1
			
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
		_dbg(debug, f"GCS list_blobs scanned objects={count}, identified runs={len(latest_ts_by_run)}")
		runs = [(rn, ts) for rn, ts in latest_ts_by_run.items()]
		return sorted(runs, key=lambda x: x[1], reverse=True)
	except Exception as e:
		_dbg(True, f"GCS listing failed for root={root}: {type(e).__name__}: {e}")
		return []

def list_runs(root: str, debug: bool = False):
	if root.startswith("gs://"):
		return _list_runs_gcs(root, debug=debug)
	return _list_runs_local(root, debug=debug)

def main() -> None:
	parser = argparse.ArgumentParser(description="Quick inference test for JE model (PyTorch)")
	parser.add_argument("--description", type=str, default="CAPITAL SOCIAL", help="Input description to infer on")
	parser.add_argument("--date", type=str, default=None, help="Optional journal entry date YYYY-MM-DD used for conditioning features")
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
	parser.add_argument("--trainable-catalog", action="store_true", default=False, help="If set, registers catalog embeddings as a trainable parameter")
	parser.add_argument("--debug", action="store_true", default=False, help="Print debug shapes and logits")
	parser.add_argument("--min-lines", type=int, default=1, help="Force at least this many lines via fallback if beam returns empty")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

	
	if args.list_runs:
		_dbg(True, f"Listing runs under root: {args.runs_root}")
		runs = list_runs(args.runs_root, debug=True)
		if not runs:
			print("No runs found.")
			return
		for name, ts in runs:
			print(f"{name}\t{ts}")
		return

	
	_dbg(args.debug, f"Initial args.runs_root={args.runs_root} args.run_name={args.run_name} args.use_latest_run={args.use_latest_run} args.checkpoint={args.checkpoint}")
	ckpt_uri = args.checkpoint
	
	if args.run_name or args.use_latest_run:
		base_uri_sel = None
		if args.run_name:
			base_uri_sel = _uri_join(args.runs_root, args.run_name)
		else:
			runs = list_runs(args.runs_root, debug=args.debug)
			if runs:
				base_uri_sel = _uri_join(args.runs_root, runs[0][0])
			else:
				raise FileNotFoundError(f"No runs found under {args.runs_root}")
		ckpt_uri = _uri_join(base_uri_sel, "model_state.pt")
	_dbg(args.debug, f"Resolved checkpoint candidate: {ckpt_uri}")
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
	_dbg(args.debug, f"Final checkpoint URI: {ckpt_uri}")
	_dbg(args.debug, f"Base URI for side files: {base_uri}")

	
	cfg: Dict[str, Any] = {}
	config_uri = _uri_join(base_uri, "config.json")
	if _uri_exists(config_uri):
		try:
			cfg = load_json_from_uri(config_uri)
			
			args.encoder = cfg.get("encoder", args.encoder)
			args.hidden_dim = int(cfg.get("hidden_dim", args.hidden_dim))
			args.max_lines = int(cfg.get("max_lines", args.max_lines))
			args.max_length = int(cfg.get("max_length", args.max_length))
			if args.debug:
				print(f"[infer] Loaded config from {config_uri}")
		except Exception as e:
			if args.debug:
				print(f"[infer] Failed to load config.json from {config_uri}: {e}")
	else:
		_dbg(args.debug, f"No config.json found at {config_uri}")

	
	_dbg(args.debug, f"Tokenizing description: '{args.description.strip()}' with encoder '{args.encoder}' max_length={args.max_length}")
	tok = DescriptionTokenizer(model_name_or_path=args.encoder, max_length=args.max_length)
	desc = args.description.strip()
	batch = tok.tokenize_batch([desc])
	input_ids = torch.tensor([batch["input_ids"][0]], dtype=torch.long, device=device)
	attention_mask = torch.tensor([batch["attention_mask"][0]], dtype=torch.long, device=device)

	
	accounts_uri: Optional[str] = args.accounts_artifact
	if not accounts_uri:
		cand = _uri_join(base_uri, "accounts_artifact.json")
		if _uri_exists(cand):
			accounts_uri = cand
		else:
			raise FileNotFoundError("Accounts artifact not provided and accounts_artifact.json not found next to checkpoint.")
	_dbg(args.debug, f"Accounts artifact URI: {accounts_uri}")
	artifact = load_json_from_uri(accounts_uri)
	cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)
	if args.debug:
		print(f"[infer] catalog_embeddings shape={tuple(cat_emb.shape)} num_accounts={int(cat_emb.shape[0])} from={accounts_uri}")

	
	cond_numeric = torch.zeros((1, 8), dtype=torch.float32, device=device)
	if args.date:
		try:
			d = datetime.strptime(args.date, "%Y-%m-%d").date()
			year = float(d.year)
			month = float(d.month)
			day = float(d.day)
			
			dow = float(d.weekday())
			month_sin = math.sin(2.0 * math.pi * ((month - 1.0) / 12.0))
			month_cos = math.cos(2.0 * math.pi * ((month - 1.0) / 12.0))
			day_sin = math.sin(2.0 * math.pi * ((day - 1.0) / 31.0))
			day_cos = math.cos(2.0 * math.pi * ((day - 1.0) / 31.0))
			cond_numeric = torch.tensor([[year, month, day, dow, month_sin, month_cos, day_sin, day_cos]], dtype=torch.float32, device=device)
			if args.debug:
				print(f"[infer] using date conditioning: {args.date} -> {cond_numeric.tolist()}")
		except Exception as e:
			print(f"[infer] Warning: failed to parse --date='{args.date}': {e}. Falling back to zeros for cond_numeric.")

	
	_dbg(args.debug, f"Creating JEModel with hidden_dim={args.hidden_dim} max_lines={args.max_lines}")
	
	ptr_temp = float(cfg.get("temperature", 1.0))
	ptr_scale = float(cfg.get("pointer_scale_init", 1.0))
	ptr_learn = bool(cfg.get("learnable_pointer_scale", False))
	ptr_use_norm = bool(cfg.get("pointer_use_norm", True))
	
	trainable_cat_cfg = bool(cfg.get("trainable_catalog", False))
	if trainable_cat_cfg:
		args.trainable_catalog = True
	model = JEModel(
		encoder_loc=args.encoder,
		hidden_dim=args.hidden_dim,
		max_lines=args.max_lines,
		temperature=float(ptr_temp),
		pointer_scale_init=ptr_scale,
		pointer_learnable_scale=ptr_learn,
		use_pointer_norm=ptr_use_norm,
		learn_catalog=bool(args.trainable_catalog),
	).to(device)
 
	if args.trainable_catalog:
		model.set_catalog_embeddings(cat_emb)
		_dbg(args.debug, "Registered catalog embeddings as trainable parameter")
     
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
	_dbg(args.debug, "Model set to eval()")

	
	currency = [args.currency]
	je_type = [args.je_type]
	_dbg(args.debug, f"Conditioning: currency={currency} je_type={je_type}")

	
	retr_mem = torch.zeros((1, args.hidden_dim), dtype=torch.float32, device=device)
	_dbg(args.debug, f"Retrieval memory initialized with shape={tuple(retr_mem.shape)}")

	
	if args.debug:
		with torch.no_grad():
			prev_acc0 = torch.full((1, model.max_lines), -1, dtype=torch.long, device=device)
			prev_side0 = torch.full((1, model.max_lines), -1, dtype=torch.long, device=device)
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
			_dbg(True if args.debug else False, f"step0 stop_probs={stop0.tolist()} side_probs={side0.tolist()}")
			_dbg(True if args.debug else False, f"step0 pointer topk indices={topi.tolist()} probs={topv.tolist()}")

	
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
	_dbg(args.debug, f"beam candidates returned={len(cands) if cands else 0}")
	if args.debug:
		_dbg(True, f"{cands}")
	
	if (not cands or (cands and cands[0].get("length", 0) == 0)) and int(args.min_lines) > 0:
		print(f"[infer] fallback to min-lines={args.min_lines}")
		with torch.no_grad():
			prev_acc0 = torch.full((1, model.max_lines), -1, dtype=torch.long, device=device)
			prev_side0 = torch.full((1, model.max_lines), -1, dtype=torch.long, device=device)
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
			if acc_idx < 0:
				
				if cat_emb.shape[0] > 0:
					acc_idx = 0
			if acc_idx >= 0:
				cands = [{"accounts": [acc_idx], "sides": [side_idx], "length": 1, "score": float(torch.log(torch.clamp(ptr0[acc_idx], min=1e-9)).item()), "prob": float(ptr0[acc_idx].item()), "logprob": float(torch.log(torch.clamp(ptr0[acc_idx], min=1e-9)).item())}]
			else:
				cands = []
	if not cands:
		print(json.dumps({"candidates": []}, indent=2))
		return
	
	top = cands[:3]
	print(json.dumps({"candidates": top}, indent=2))


if __name__ == "__main__":
	main()


