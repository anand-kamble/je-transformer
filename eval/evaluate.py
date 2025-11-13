from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from data.text_normalization import normalize_description
from inference.beam_decode import beam_search_decode
from inference.postprocess import postprocess_candidates
from models.catalog_encoder import CatalogEncoder
from models.je_model_torch import JEModel
from models.losses import SetF1Metric
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
	return encoder({"number": number, "name": name, "nature": nature}).to(device)


def ece(probs: List[float], correct: List[bool], num_bins: int = 15) -> float:
	probs_np = np.asarray(probs, dtype=np.float64)
	corr_np = np.asarray(correct, dtype=np.float64)
	bins = np.linspace(0.0, 1.0, num_bins + 1)
	ece_val = 0.0
	n = len(probs_np)
	for i in range(num_bins):
		l, r = bins[i], bins[i + 1]
		mask = (probs_np >= l) & (probs_np < r) if i < num_bins - 1 else (probs_np >= l) & (probs_np <= r)
		if not np.any(mask):
			continue
		conf = probs_np[mask].mean()
		acc = corr_np[mask].mean()
		ece_val += (mask.sum() / n) * abs(acc - conf)
	return float(ece_val)


def tune_temperature(probs: List[float], correct: List[bool], grid: List[float]) -> Tuple[float, float]:
	best_T = 1.0
	best_ece = ece(probs, correct)
	for T in grid:
		if T <= 0:
			continue
		scaled = [float(p ** (1.0 / T)) for p in probs]
		cur_ece = ece(scaled, correct)
		if cur_ece < best_ece:
			best_ece = cur_ece
			best_T = T
	return best_T, best_ece


def main():
	parser = argparse.ArgumentParser(description="Evaluate JE model: token/set metrics and calibration (PyTorch)")
	parser.add_argument("--parquet-pattern", type=str, required=True)
	parser.add_argument("--accounts-artifact", type=str, required=True)
	parser.add_argument("--encoder", type=str, default="bert-base-multilingual-cased")
	parser.add_argument("--max-length", type=int, default=128)
	parser.add_argument("--max-lines", type=int, default=8)
	parser.add_argument("--hidden-dim", type=int, default=256)
	parser.add_argument("--limit", type=int, default=1000)
	parser.add_argument("--beam-size", type=int, default=20)
	parser.add_argument("--alpha", type=float, default=0.7)
	parser.add_argument("--tau", type=float, default=0.5)
	parser.add_argument("--output-report", type=str, default="eval_report.json")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
	tok = DescriptionTokenizer(model_name_or_path=args.encoder, max_length=args.max_length)

	artifact = load_json_from_uri(args.accounts_artifact)
	cat_emb = build_catalog_embeddings(artifact, emb_dim=args.hidden_dim, device=device)

	model = JEModel(
		encoder_loc=args.encoder,
		hidden_dim=args.hidden_dim,
		max_lines=args.max_lines,
		temperature=1.0,
	).to(device)
	model.eval()

	metric_set = SetF1Metric()
	token_acc_ptr = 0.0
	token_acc_side = 0.0
	token_acc_stop = 0.0
	cnt_tokens = 0

	seq_probs: List[float] = []
	seq_correct: List[bool] = []

	import glob
	paths = sorted(glob.glob(args.parquet_pattern))
	import pandas as pd
	import pyarrow.parquet as pq
	frames = [pq.read_table(p).to_pandas() for p in paths]
	df = pd.concat(frames, ignore_index=True)

	count = 0
	for _, ex in df.iterrows():
		if count >= args.limit:
			break
		desc = normalize_description(ex.get("description", ""))
		batch = tok.tokenize_batch([desc])
		input_ids = torch.tensor([batch["input_ids"][0]], dtype=torch.long, device=device)
		attention_mask = torch.tensor([batch["attention_mask"][0]], dtype=torch.long, device=device)

		cond_numeric = torch.tensor([[float(ex.get("date_year", 0)), float(ex.get("date_month", 0)), float(ex.get("date_day", 0)), float(ex.get("date_dow", 0)), float(ex.get("date_month_sin", 0.0)), float(ex.get("date_month_cos", 0.0)), float(ex.get("date_day_sin", 0.0)), float(ex.get("date_day_cos", 0.0))]], dtype=torch.float32, device=device)
		currency = [str(ex.get("currency", ""))]
		je_type = [str(ex.get("journal_entry_type", ""))]

		
		debits = [int(i) for i in (ex.get("debit_accounts", []) or []) if int(i) >= 0]
		credits = [int(i) for i in (ex.get("credit_accounts", []) or []) if int(i) >= 0]
		tgt_seq = debits + credits
		tgt_side = [0] * len(debits) + [1] * len(credits)
		T = min(len(tgt_seq), args.max_lines)
		prev_acc = [-1] + tgt_seq[: max(0, T - 1)]
		prev_side = [-1] + tgt_side[: max(0, T - 1)]
		prev_acc = prev_acc + [-1] * (args.max_lines - len(prev_acc))
		prev_side = prev_side + [-1] * (args.max_lines - len(prev_side))
		target_acc = tgt_seq[:T] + [-1] * (args.max_lines - T)
		target_side = tgt_side[:T] + [-1] * (args.max_lines - T)
		target_stop = [0] * (max(0, T - 1)) + ([1] if T > 0 else [1]) + [0] * (args.max_lines - T)

		outs = model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			prev_account_idx=torch.tensor([prev_acc], dtype=torch.long, device=device),
			prev_side_id=torch.tensor([prev_side], dtype=torch.long, device=device),
			catalog_embeddings=cat_emb,
			retrieval_memory=torch.zeros((1, args.hidden_dim), dtype=torch.float32, device=device),
			cond_numeric=cond_numeric,
			currency=currency,
			journal_entry_type=je_type,
		)
		ptr_pred = outs["pointer_logits"].argmax(dim=-1).detach().cpu().numpy()[0][:T] if T > 0 else []
		side_pred = outs["side_logits"].argmax(dim=-1).detach().cpu().numpy()[0][:T] if T > 0 else []
		stop_pred = outs["stop_logits"].argmax(dim=-1).detach().cpu().numpy()[0][:T] if T > 0 else []
		if T > 0:
			token_acc_ptr += float(np.mean((np.array(ptr_pred) == np.array(tgt_seq[:T])).astype(np.float32)))
			token_acc_side += float(np.mean((np.array(side_pred) == np.array(tgt_side[:T])).astype(np.float32)))
			token_acc_stop += float(stop_pred[-1] == 1)
			cnt_tokens += 1

		metric_set.update_state(
			outs["pointer_logits"].detach().cpu(),
			outs["side_logits"].detach().cpu(),
			torch.tensor([target_acc]),
			torch.tensor([target_side]),
			torch.tensor([target_stop]),
		)

		cands = beam_search_decode(
			model=model,
			input_ids=input_ids,
			attention_mask=attention_mask,
			catalog_embeddings=cat_emb,
			retrieval_memory=None,
			cond_numeric=cond_numeric,
			currency=currency,
			journal_entry_type=je_type,
			beam_size=args.beam_size,
			alpha=args.alpha,
			max_lines=args.max_lines,
			tau=args.tau,
		)
		if not cands:
			continue
		best = cands[0]
		pairs_pred = list(zip(best["accounts"], best["sides"]))
		pairs_true = list(zip(tgt_seq[:T], tgt_side[:T]))
		seq_probs.append(best["prob"])
		seq_correct.append(pairs_pred == pairs_true)

		count += 1

	
	ece_raw = ece(seq_probs, seq_correct, num_bins=15)
	best_T, best_ece = tune_temperature(seq_probs, seq_correct, grid=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0])

	report = {
		"examples": count,
		"token_ptr_acc": float(token_acc_ptr / max(1, cnt_tokens)),
		"token_side_acc": float(token_acc_side / max(1, cnt_tokens)),
		"token_stop_acc": float(token_acc_stop / max(1, cnt_tokens)),
		"set_f1": float(metric_set.result()),
		"calibration": {"ece_raw": ece_raw, "best_temperature": best_T, "ece_best": best_ece},
	}
	with open(args.output_report, "w") as f:
		json.dump(report, f, indent=2)
	print(json.dumps(report, indent=2))


if __name__ == "__main__":
	main()


