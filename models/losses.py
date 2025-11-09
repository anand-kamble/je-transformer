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
	loss = tf.keras.losses.sparse_categorical_crossentropy(targets_safe, logits, from_logits=True)  # [B, T]
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
			stop_idx = tf.argmax(tf.cast(tf.equal(target_stop, self.stop_id), tf.int32), axis=-1)  # [B]
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


