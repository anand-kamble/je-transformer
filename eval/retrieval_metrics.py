from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Set, Tuple


def compute_jaccard(a: Set[int], b: Set[int]) -> float:
	ia = len(a & b)
	ua = len(a | b)
	return float(ia) / float(max(ua, 1))


def context_precision_recall(query_set: Set[int], neighbor_sets: Sequence[Set[int]]) -> Tuple[float, float]:
	if len(query_set) == 0:
		return 0.0, 0.0
	union_neighbors: Set[int] = set().union(*neighbor_sets) if neighbor_sets else set()
	intersection = len(query_set & union_neighbors)
	precision = float(intersection) / float(max(len(union_neighbors), 1))
	recall = float(intersection) / float(max(len(query_set), 1))
	return precision, recall


def mrr_from_binary_hits(hits: Sequence[bool]) -> float:
	for i, h in enumerate(hits, start=1):
		if h:
			return 1.0 / float(i)
	return 0.0


def aggregate_retrieval_metrics(
	query_ids: Sequence[str],
	topk_indices: Sequence[Sequence[int]],
	index_ids: Sequence[str],
	jeid_to_accountset: Dict[str, Set[int]],
) -> Dict[str, float]:
	"""
	Computes retrieval metrics based on Jaccard overlap of account sets.
	- avg_jaccard@K: mean Jaccard across all queries and their top-K neighbors
	- top1_jaccard: mean Jaccard of rank-1 neighbors
	- hit_rate@K: fraction of queries with any Jaccard > 0 among top-K
	- mrr: mean reciprocal rank where first Jaccard > 0
	- context_precision@K / context_recall@K using union of neighbor sets
	"""
	total_top1 = 0.0
	total_avg = 0.0
	total_queries = 0
	hit_count = 0
	mrr_sum = 0.0
	prec_sum = 0.0
	rec_sum = 0.0
	for qi, qid in enumerate(query_ids):
		q_set = jeid_to_accountset.get(str(qid), set())
		if q_set is None:
			q_set = set()
		ranks = topk_indices[qi] if qi < len(topk_indices) else []
		if not ranks:
			continue
		jacc_per_rank: List[float] = []
		neighbor_sets: List[Set[int]] = []
		for ridx in ranks:
			if 0 <= ridx < len(index_ids):
				nid = index_ids[ridx]
				n_set = jeid_to_accountset.get(str(nid), set())
			else:
				n_set = set()
			neighbor_sets.append(n_set)
			jacc_per_rank.append(compute_jaccard(q_set, n_set))
		if not jacc_per_rank:
			continue
		total_queries += 1
		total_top1 += jacc_per_rank[0]
		total_avg += sum(jacc_per_rank) / float(len(jacc_per_rank))
		any_hit = any(j > 0.0 for j in jacc_per_rank)
		if any_hit:
			hit_count += 1
		mrr_sum += mrr_from_binary_hits([j > 0.0 for j in jacc_per_rank])
		prec, rec = context_precision_recall(q_set, neighbor_sets)
		prec_sum += prec
		rec_sum += rec
	if total_queries == 0:
		return {
			"avg_jaccard@K": 0.0,
			"top1_jaccard": 0.0,
			"hit_rate@K": 0.0,
			"mrr": 0.0,
			"context_precision@K": 0.0,
			"context_recall@K": 0.0,
		}
	return {
		"avg_jaccard@K": total_avg / float(total_queries),
		"top1_jaccard": total_top1 / float(total_queries),
		"hit_rate@K": float(hit_count) / float(total_queries),
		"mrr": mrr_sum / float(total_queries),
		"context_precision@K": prec_sum / float(total_queries),
		"context_recall@K": rec_sum / float(total_queries),
	}


