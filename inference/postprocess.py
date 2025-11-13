from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _has_both_sides(sides: List[int]) -> bool:
	has_debit = any(s == 0 for s in sides)
	has_credit = any(s == 1 for s in sides)
	return bool(has_debit and has_credit)


def _collapse_unique_pairs(accounts: List[int], sides: List[int]) -> Tuple[List[int], List[int]]:
	seen = set()
	new_acc: List[int] = []
	new_sides: List[int] = []
	for a, s in zip(accounts, sides):
		key = (int(a), int(s))
		if key in seen:
			continue
		seen.add(key)
		new_acc.append(int(a))
		new_sides.append(int(s))
	return new_acc, new_sides


def _limit_per_account(
	accounts: List[int],
	sides: List[int],
	max_dup_per_account: int,
) -> Tuple[List[int], List[int]]:
	counts: Dict[int, int] = {}
	new_acc: List[int] = []
	new_sides: List[int] = []
	for a, s in zip(accounts, sides):
		a = int(a)
		c = counts.get(a, 0)
		if c >= max_dup_per_account:
			continue
		counts[a] = c + 1
		new_acc.append(a)
		new_sides.append(int(s))
	return new_acc, new_sides


def apply_duplicate_policy(
	cand: Dict[str, Any],
	policy: str = "allow",
	max_dup_per_account: Optional[int] = None,
) -> Dict[str, Any]:

	accounts: List[int] = list(cand.get("accounts", []))
	sides: List[int] = list(cand.get("sides", []))

	if policy == "allow":
		return cand
	if policy == "collapse_unique_pairs":
		accounts, sides = _collapse_unique_pairs(accounts, sides)
	elif policy == "limit_per_account":
		if not isinstance(max_dup_per_account, int) or max_dup_per_account <= 0:
			raise ValueError("max_dup_per_account must be a positive integer for 'limit_per_account' policy")
		accounts, sides = _limit_per_account(accounts, sides, max_dup_per_account)
	else:
		raise ValueError(f"Unknown duplicate policy: {policy}")

	out = dict(cand)
	out["accounts"] = accounts
	out["sides"] = sides
	out["length"] = len(accounts)
	out["postprocessed"] = True
	out.setdefault("notes", []).append(f"duplicate_policy={policy}")
	return out


def structural_filter(
	cand: Dict[str, Any],
	require_both_sides: bool = True,
	min_lines: int = 2,
) -> bool:
	accounts: List[int] = list(cand.get("accounts", []))
	sides: List[int] = list(cand.get("sides", []))
	if len(accounts) != len(sides):
		return False
	if len(accounts) < int(min_lines):
		return False
	if require_both_sides and not _has_both_sides(sides):
		return False
	return True


def postprocess_candidates(
	candidates: List[Dict[str, Any]],
	duplicate_policy: str = "allow",
	max_dup_per_account: Optional[int] = None,
	require_both_sides: bool = True,
	min_lines: int = 2,
	keep_top_k_if_empty: int = 1,
) -> List[Dict[str, Any]]:
	processed: List[Dict[str, Any]] = []
	for cand in candidates:
		c = apply_duplicate_policy(cand, policy=duplicate_policy, max_dup_per_account=max_dup_per_account)
		if structural_filter(c, require_both_sides=require_both_sides, min_lines=min_lines):
			processed.append(c)
	
	if not processed:
		return candidates[:keep_top_k_if_empty]
	return processed


