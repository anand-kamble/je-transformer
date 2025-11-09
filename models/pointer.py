from __future__ import annotations

import tensorflow as tf


class PointerLayer(tf.keras.layers.Layer):
    """
    Dot-product pointer over a catalog.

    call(decoder_state, catalog_embeddings, mask=None) -> logits
      - decoder_state: [batch, hidden]
      - catalog_embeddings: [catalog_size, hidden] (or [batch, catalog_size, hidden] if per-batch)
      - mask: [catalog_size] or [batch, catalog_size], 1 for valid, 0 to mask

    Returns:
      - logits over catalog indices: [batch, catalog_size]
    """

    def __init__(self, temperature: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = float(max(1e-6, temperature))

    def call(self, decoder_state: tf.Tensor, catalog_embeddings: tf.Tensor, mask: tf.Tensor | None = None) -> tf.Tensor:
        # Ensure shapes
        dec = tf.convert_to_tensor(decoder_state, dtype=tf.float32)  # [B, H]
        cat = tf.convert_to_tensor(catalog_embeddings, dtype=tf.float32)  # [C, H] or [B, C, H]

        if tf.rank(cat) == 2:
            # [C, H] -> broadcast to [B, C, H]
            cat = tf.expand_dims(cat, axis=0)  # [1, C, H]
            cat = tf.repeat(cat, repeats=tf.shape(dec)[0], axis=0)  # [B, C, H]

        # Normalize (optional; improves stability)
        dec_n = tf.math.l2_normalize(dec, axis=-1)  # [B, H]
        cat_n = tf.math.l2_normalize(cat, axis=-1)  # [B, C, H]

        # Dot product -> [B, C]
        logits = tf.einsum("bh,bch->bc", dec_n, cat_n) / self.temperature

        if mask is not None:
            m = tf.cast(mask, tf.float32)
            if tf.rank(m) == 1:
                m = tf.expand_dims(m, 0)
                m = tf.repeat(m, repeats=tf.shape(dec)[0], axis=0)
            very_neg = tf.constant(-1e9, dtype=tf.float32)
            logits = tf.where(m > 0.5, logits, very_neg)

        return logits

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"temperature": self.temperature})
        return cfg

