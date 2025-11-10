from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from inference.retrieval_memory import build_retrieval_memory_for_text


@dataclass
class BeamState:
	accounts: List[int]
	sides: List[int]
	logprob: float
	finished: bool

	def length(self) -> int:
		return len(self.accounts)


def length_penalized_score(logprob: float, length: int, alpha: float) -> float:
	L = max(1, length)
	if alpha <= 0.0:
		return logprob
	return logprob / (L ** alpha)


def beam_search_decode(
	model,
	input_ids: tf.Tensor,
	attention_mask: tf.Tensor,
	catalog_embeddings: tf.Tensor,
	retrieval_memory: Optional[tf.Tensor] = None,
	cond_numeric: Optional[tf.Tensor] = None,
	currency: Optional[tf.Tensor] = None,
	journal_entry_type: Optional[tf.Tensor] = None,
	beam_size: int = 10,
	alpha: float = 0.7,
	max_lines: int = 8,
	topk_accounts: Optional[int] = None,
	tau: Optional[float] = None,
	# Optional retrieval artifacts to auto-build memory when not provided
	query_text: Optional[str] = None,
	index_dir: Optional[str] = None,
	ids_uri: Optional[str] = None,
	embeddings_uri: Optional[str] = None,
	tokenizer_loc: str = "bert-base-multilingual-cased",
	encoder_loc: str = "bert-base-multilingual-cased",
	use_cls: bool = False,
	top_k_retrieval: int = 5,
) -> List[Dict[str, Any]]:
	"""
	Autoregressive beam search over (account, side) with STOP head.
	- input_ids: [1, L]
	- attention_mask: [1, L]
	- catalog_embeddings: [C, H]
	- retrieval_memory: [K, H] or None (defaults to zeros [1, H])
	Returns candidates sorted by length-penalized score (desc). If tau is provided (0<tau<=1),
	filters by probability >= tau (using exp of length-penalized logprob). If none qualify, returns top-1.
	"""
	if retrieval_memory is None:
		# Try to build from artifacts if provided
		if query_text and index_dir and ids_uri and embeddings_uri:
			retrieval_memory = build_retrieval_memory_for_text(
				query_text, index_dir, ids_uri, embeddings_uri,
				tokenizer_loc=tokenizer_loc, encoder_loc=encoder_loc, use_cls=use_cls, top_k=top_k_retrieval
			)  # [K, Henc]
		else:
			retrieval_memory = tf.zeros((1, int(catalog_embeddings.shape[-1])), dtype=tf.float32)
	if cond_numeric is None:
		cond_numeric = tf.zeros((1, 8), dtype=tf.float32)
	if currency is None:
		currency = tf.constant([""], dtype=tf.string)
	if journal_entry_type is None:
		journal_entry_type = tf.constant([""], dtype=tf.string)

	C = int(catalog_embeddings.shape[0])
	if topk_accounts is None or topk_accounts > C:
		topk_accounts = min(C, beam_size)

	# Initialize beams
	beams: List[BeamState] = [BeamState(accounts=[], sides=[], logprob=0.0, finished=False)]
	finished_beams: List[BeamState] = []

	for t in range(max_lines):
		# Collect candidates from all beams
		new_beams: List[BeamState] = []
		for b in beams:
			if b.finished:
				new_beams.append(b)
				continue

			# Build prev arrays (shape [1, T])
			prev_acc = np.full((1, max_lines), -1, dtype=np.int32)
			prev_side = np.full((1, max_lines), -1, dtype=np.int32)
			for i in range(len(b.accounts)):
				prev_acc[0, i] = b.accounts[i]
				prev_side[0, i] = b.sides[i]

			outputs = model(
				{
					"input_ids": input_ids,
					"attention_mask": attention_mask,
					"prev_account_idx": tf.convert_to_tensor(prev_acc),
					"prev_side_id": tf.convert_to_tensor(prev_side),
					"catalog_embeddings": catalog_embeddings,
					"retrieval_memory": retrieval_memory,
					"cond_numeric": cond_numeric,
					"currency": currency,
					"journal_entry_type": journal_entry_type,
				},
				training=False,
			)
			ptr_logits = outputs["pointer_logits"].numpy()[0, t, :]  # [C]
			side_logits = outputs["side_logits"].numpy()[0, t, :]    # [2]
			stop_logits = outputs["stop_logits"].numpy()[0, t, :]    # [2] [continue, stop]

			# Log-softmax
			ptr_logp = (ptr_logits - tf.reduce_logsumexp(ptr_logits)).numpy()
			side_logp = (side_logits - tf.reduce_logsumexp(side_logits)).numpy()
			stop_logp = (stop_logits - tf.reduce_logsumexp(stop_logits)).numpy()

			# Option 1: stop here
			stop_lp = float(stop_logp[1])  # index 1 = stop
			new_beams.append(BeamState(
				accounts=b.accounts.copy(),
				sides=b.sides.copy(),
				logprob=b.logprob + stop_lp,
				finished=True,
			))

			# Option 2: continue with (account, side)
			cont_lp = float(stop_logp[0])  # index 0 = continue
			# Take top-k accounts
			top_idx = np.argpartition(-ptr_logp, range(topk_accounts))[:topk_accounts]
			# For stability, sort by logp within the slice
			top_idx = top_idx[np.argsort(-ptr_logp[top_idx])]

			for acc_id in top_idx:
				for side_id in (0, 1):
					lp = cont_lp + float(ptr_logp[acc_id]) + float(side_logp[side_id])
					new_beams.append(BeamState(
						accounts=b.accounts + [int(acc_id)],
						sides=b.sides + [int(side_id)],
						logprob=b.logprob + lp,
						finished=False,
					))

		# Prune to beam_size by length-penalized scores
		new_beams.sort(key=lambda bs: length_penalized_score(bs.logprob, max(1, bs.length()), alpha), reverse=True)
		beams = new_beams[:beam_size]

		# Separate out finished beams
		finished_beams.extend([b for b in beams if b.finished])
		# If all beams finished, stop early
		if all(b.finished for b in beams):
			break

	# If none finished, use best ongoing beams as candidates
	if not finished_beams:
		finished_beams = beams

	# Prepare results sorted by penalized score
	results = []
	for b in finished_beams:
		score = length_penalized_score(b.logprob, max(1, b.length()), alpha)
		results.append({
			"accounts": b.accounts,
			"sides": b.sides,
			"score": float(score),
			"prob": float(math.exp(score)),
			"logprob": float(b.logprob),
			"length": int(b.length()),
		})
	results.sort(key=lambda r: r["score"], reverse=True)

	# τ filtering (probability after length penalty)
	if tau is not None:
		kept = [r for r in results if r["prob"] >= float(tau)]
		if kept:
			return kept
		# fallback: return top-1
		return results[:1]
	return results


