from __future__ import annotations

import keras
import tensorflow as tf


def masked_sparse_ce(logits: tf.Tensor, targets: tf.Tensor, ignore_index: int = -1) -> tf.Tensor:
	"""
	Compute sparse CE over last dimension of logits with mask where targets == ignore_index.
	Shapes:
	  - logits: [B, T, V]
	  - targets: [B, T]
	Returns mean loss over non-ignored positions.
	"""
	logits = tf.convert_to_tensor(logits)
	targets = tf.convert_to_tensor(targets)
	mask = tf.not_equal(targets, tf.constant(ignore_index, dtype=targets.dtype))  # [B, T]
	valid = tf.cast(mask, tf.float32)
	# Keras expects int32 labels
	targets_safe = tf.where(mask, targets, tf.zeros_like(targets))
	loss = keras.losses.sparse_categorical_crossentropy(targets_safe, logits, from_logits=True)  # [B, T]
	loss = tf.reduce_sum(loss * valid) / (tf.reduce_sum(valid) + 1e-6)
	return loss


def pointer_loss(pointer_logits: tf.Tensor, target_account_idx: tf.Tensor, ignore_index: int = -1) -> tf.Tensor:
	return masked_sparse_ce(pointer_logits, target_account_idx, ignore_index=ignore_index)


def side_loss(side_logits: tf.Tensor, target_side_id: tf.Tensor, ignore_index: int = -1) -> tf.Tensor:
	return masked_sparse_ce(side_logits, target_side_id, ignore_index=ignore_index)


def stop_loss(stop_logits: tf.Tensor, target_stop_id: tf.Tensor, ignore_index: int = -1) -> tf.Tensor:
	return masked_sparse_ce(stop_logits, target_stop_id, ignore_index=ignore_index)


def coverage_penalty(pointer_logits: tf.Tensor, max_total: float = 1.0) -> tf.Tensor:
	"""
	Encourage the model not to overspread mass repeatedly on the same accounts across time.
	Compute softmax over accounts per step, sum over time per account, and penalize surplus over max_total.
	  - pointer_logits: [B, T, C]
	Returns mean surplus across batch.
	"""
	p = tf.nn.softmax(pointer_logits, axis=-1)  # [B, T, C]
	sum_over_t = tf.reduce_sum(p, axis=1)       # [B, C]
	surplus = tf.nn.relu(sum_over_t - max_total)
	return tf.reduce_mean(tf.reduce_sum(surplus, axis=-1))


def flow_aux_loss(
	pointer_logits: tf.Tensor,
	side_logits: tf.Tensor,
	debit_indices: tf.Tensor,
	debit_weights: tf.Tensor,
	credit_indices: tf.Tensor,
	credit_weights: tf.Tensor,
) -> tf.Tensor:
	"""
	Align predicted per-side mass over accounts with normalized debit/credit amounts.
	- pointer_logits: [B, T, C]
	- side_logits: [B, T, 2]
	- debit_indices: [B, D] int32 indices into catalog
	- debit_weights: [B, D] float normalized to sum 1 per example (or zeros)
	- credit_indices: [B, K] int32
	- credit_weights: [B, K] float normalized to sum 1 per example (or zeros)
	Loss: MSE between gathered predicted mass (renormalized on the target support) and target weights, averaged over sides and batch.
	"""
	p = tf.nn.softmax(pointer_logits, axis=-1)  # [B, T, C]
	s = tf.nn.softmax(side_logits, axis=-1)     # [B, T, 2]
	# Aggregate predicted mass per catalog over time for each side
	pred_debit_mass = tf.reduce_sum(tf.expand_dims(s[:, :, 0], -1) * p, axis=1)   # [B, C]
	pred_credit_mass = tf.reduce_sum(tf.expand_dims(s[:, :, 1], -1) * p, axis=1)  # [B, C]

	def side_mse(pred_mass, idxs, wts):
		# Gather predicted mass at target indices
		gath = tf.gather(pred_mass, idxs, batch_dims=1)  # [B, L]
		# Renormalize gathered mass to sum 1 over L to compare with normalized targets
		sum_g = tf.reduce_sum(gath, axis=-1, keepdims=True) + 1e-8
		g_norm = gath / sum_g
		# Some rows may have zero target length (all idxs == -1). Mask them out.
		valid = tf.cast(tf.reduce_any(tf.not_equal(idxs, -1), axis=-1), tf.float32)  # [B]
		mse = tf.reduce_mean(tf.square(g_norm - wts), axis=-1)  # [B]
		# Replace NaNs (when L=0) with 0
		mse = tf.where(tf.math.is_finite(mse), mse, tf.zeros_like(mse))
		# Average over valid examples
		return tf.reduce_sum(mse * valid) / (tf.reduce_sum(valid) + 1e-6)

	l_d = side_mse(pred_debit_mass, debit_indices, debit_weights)
	l_c = side_mse(pred_credit_mass, credit_indices, credit_weights)
	return 0.5 * (l_d + l_c)


class SetF1Metric(keras.metrics.Metric):
	"""
	Approximate set-level F1 over (account, side) pairs ignoring order.
	Uses greedy argmax decoding per step until STOP=1 in targets (or uses T steps if no explicit stop labels).
	Note: This is a simple approximation; Hungarian matching can be added later for stricter evaluation.
	"""
	def __init__(self, name: str = "set_f1", stop_id: int = 1):
		super().__init__(name=name)
		self.stop_id = stop_id
		self.tp = self.add_weight(name="tp", initializer="zeros")
		self.fp = self.add_weight(name="fp", initializer="zeros")
		self.fn = self.add_weight(name="fn", initializer="zeros")

	def update_state(
		self,
		pointer_logits: tf.Tensor,
		side_logits: tf.Tensor,
		target_accounts: tf.Tensor,
		target_sides: tf.Tensor,
		target_stop: tf.Tensor | None = None,
	):
		# Predicted pairs
		pred_accounts = tf.argmax(pointer_logits, axis=-1)  # [B, T]
		pred_sides = tf.argmax(side_logits, axis=-1)        # [B, T]

		if target_stop is not None:
			# Use the first position where target_stop==1 as length per example
			stop_idx = tf.argmax(
				tf.cast(tf.equal(target_stop, self.stop_id), tf.int32),
				axis=-1,
				output_type=tf.int32,
			)  # [B] int32
			# If no STOP present, use full length (fallback)
			has_stop = tf.reduce_any(tf.equal(target_stop, self.stop_id), axis=-1)  # [B]
			T = tf.shape(pred_accounts)[1]
			lengths = tf.where(has_stop, stop_idx, tf.fill(tf.shape(stop_idx), T))
		else:
			T = tf.shape(pred_accounts)[1]
			lengths = tf.fill([tf.shape(pred_accounts)[0]], T)

		def gather_pairs(acc, side, L):
			acc = acc[:L]
			side = side[:L]
			return tf.stack([acc, side], axis=-1)  # [L, 2]

		batch = tf.shape(pred_accounts)[0]
		for b in tf.range(batch):
			Lb = lengths[b]
			# Build multisets as tensors
			pairs_pred = gather_pairs(pred_accounts[b], pred_sides[b], Lb)
			pairs_true = gather_pairs(target_accounts[b], target_sides[b], Lb)

			# Convert to sets of tuples by hashing pairs into strings
			pred_str = tf.strings.join([tf.as_string(pairs_pred[:, 0]), tf.as_string(pairs_pred[:, 1])], separator=":")
			true_str = tf.strings.join([tf.as_string(pairs_true[:, 0]), tf.as_string(pairs_true[:, 1])], separator=":")

			# Unique counts approximate multiset behavior weakly
			pred_set = tf.unique(pred_str).y
			true_set = tf.unique(true_str).y

			# Compute overlaps
			# intersection = |pred ∩ true|
			# fp = |pred \ true|, fn = |true \ pred|
			def set_size(x):
				return tf.cast(tf.shape(x)[0], tf.float32)

			# Membership mask
			def is_in(a, b):
				# a in b?
				matches = tf.equal(tf.expand_dims(a, 1), tf.expand_dims(b, 0))
				return tf.reduce_any(matches, axis=1)

			in_mask = is_in(pred_set, true_set)
			intersection = tf.reduce_sum(tf.cast(in_mask, tf.float32))
			fp = set_size(pred_set) - intersection
			fn = set_size(true_set) - intersection

			self.tp.assign_add(intersection)
			self.fp.assign_add(fp)
			self.fn.assign_add(fn)

	def result(self):
		precision = self.tp / (self.tp + self.fp + 1e-6)
		recall = self.tp / (self.tp + self.fn + 1e-6)
		f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
		return f1

	def reset_states(self):
		for v in (self.tp, self.fp, self.fn):
			v.assign(0.0)


class SetF1Hungarian(keras.metrics.Metric):
	"""
	Set-level F1 using Hungarian matching between predicted and true (account, side) pairs.
	- Builds pairs up to STOP (or full T if no STOP).
	- Uses scipy.optimize.linear_sum_assignment on a 0/1 cost matrix (0 if pairs equal, else 1).
	- TP = number of zero-cost assignments; FP = |pred| - TP; FN = |true| - TP.
	"""
	def __init__(self, name: str = "set_f1_hungarian", stop_id: int = 1):
		super().__init__(name=name)
		self.stop_id = stop_id
		self.tp = self.add_weight(name="tp", initializer="zeros")
		self.fp = self.add_weight(name="fp", initializer="zeros")
		self.fn = self.add_weight(name="fn", initializer="zeros")

	def _extract_pairs(self, accounts: tf.Tensor, sides: tf.Tensor, stop: tf.Tensor | None) -> list[tuple[int, int]]:
		# Determine effective length
		T = int(accounts.shape[0])
		L = T
		if stop is not None:
			stop_np = stop.numpy().tolist()
			if self.stop_id in stop_np:
				L = stop_np.index(self.stop_id) + 1
		acc_np = accounts.numpy().tolist()[:L]
		side_np = sides.numpy().tolist()[:L]
		return [(int(a), int(s)) for a, s in zip(acc_np, side_np)]

	def update_state(
		self,
		pointer_logits: tf.Tensor,
		side_logits: tf.Tensor,
		target_accounts: tf.Tensor,
		target_sides: tf.Tensor,
		target_stop: tf.Tensor | None = None,
	):
		# Eager numpy/scipy path via py_function to avoid graph complications
		def _update_np(ptr, side, t_acc, t_side, t_stop):
			import numpy as np
			try:
				from scipy.optimize import linear_sum_assignment
			except Exception:
				# If SciPy unavailable, fall back to approximate metric behavior (no update)
				return np.array([0.0, 0.0, 0.0], dtype=np.float32)

			# Argmax predictions
			pred_acc = np.argmax(ptr, axis=-1)   # [T]
			pred_side = np.argmax(side, axis=-1) # [T]

			# Build pair lists
			def pairs(acc, sd, stop_vec):
				T = acc.shape[0]
				L = T
				if stop_vec is not None:
					stop_list = stop_vec.tolist()
					if 1 in stop_list:
						L = stop_list.index(1) + 1
				return [(int(a), int(s)) for a, s in zip(acc[:L].tolist(), sd[:L].tolist())]

			p_pairs = pairs(pred_acc, pred_side, t_stop if t_stop is not None else None)
			t_pairs = pairs(t_acc, t_side, t_stop if t_stop is not None else None)

			if len(p_pairs) == 0 and len(t_pairs) == 0:
				return np.array([0.0, 0.0, 0.0], dtype=np.float32)
			if len(p_pairs) == 0:
				return np.array([0.0, 0.0, float(len(t_pairs))], dtype=np.float32)
			if len(t_pairs) == 0:
				return np.array([0.0, float(len(p_pairs)), 0.0], dtype=np.float32)

			# Cost matrix: 0 if equal pair else 1
			P = len(p_pairs)
			Tn = len(t_pairs)
			cost = np.ones((P, Tn), dtype=np.float32)
			t_map = {pair: [] for pair in t_pairs}
			for j, tp in enumerate(t_pairs):
				t_map.setdefault(tp, []).append(j)
			for i, pp in enumerate(p_pairs):
				if pp in t_map:
					for j in t_map[pp]:
						cost[i, j] = 0.0
			row_ind, col_ind = linear_sum_assignment(cost)
			# TP = number of 0-cost assignments
			matched = int(np.sum(cost[row_ind, col_ind] < 0.5))
			fp = float(P - matched)
			fn = float(Tn - matched)
			return np.array([float(matched), fp, fn], dtype=np.float32)

		# Iterate batch with tf.py_function to aggregate counts
		B = tf.shape(pointer_logits)[0]
		for b in tf.range(B):
			ptr_b = pointer_logits[b]      # [T, C]
			side_b = side_logits[b]        # [T, 2]
			t_acc_b = target_accounts[b]   # [T]
			t_side_b = target_sides[b]     # [T]
			t_stop_b = target_stop[b] if target_stop is not None else None

			res = tf.py_function(
				func=_update_np,
				inp=[ptr_b, side_b, t_acc_b, t_side_b, t_stop_b] if t_stop_b is not None else [ptr_b, side_b, t_acc_b, t_side_b, None],
				Tout=tf.float32,
			)
			tp_v, fp_v, fn_v = tf.unstack(res)
			self.tp.assign_add(tp_v)
			self.fp.assign_add(fp_v)
			self.fn.assign_add(fn_v)

	def result(self):
		precision = self.tp / (self.tp + self.fp + 1e-6)
		recall = self.tp / (self.tp + self.fn + 1e-6)
		f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
		return f1

	def reset_states(self):
		for v in (self.tp, self.fp, self.fn):
			v.assign(0.0)


