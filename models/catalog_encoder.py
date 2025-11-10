from __future__ import annotations

import keras
import tensorflow as tf


class CatalogEncoder(keras.layers.Layer):
    """
    Money-flow-aware account encoder (lightweight, hash-based).

    Inputs: dict with string tensors (shape [num_accounts]):
      - number: account code like "502-0116"
      - name: account name
      - nature: e.g., "A","L","E","R","X" (asset/liability/...)

    Produces: float32 tensor of shape [num_accounts, emb_dim]
    """

    def __init__(
        self,
        emb_dim: int = 256,
        code_hash_bins: int = 4096,
        name_hash_bins: int = 16384,
        nature_hash_bins: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.emb_dim = int(emb_dim)

        proj_dim = max(64, emb_dim // 2)

        # Hashing + embedding for each string field
        self.code_hash = keras.layers.Hashing(num_bins=code_hash_bins)
        self.code_emb = keras.layers.Embedding(input_dim=code_hash_bins, output_dim=proj_dim)

        self.name_hash = keras.layers.Hashing(num_bins=name_hash_bins)
        self.name_emb = keras.layers.Embedding(input_dim=name_hash_bins, output_dim=proj_dim)

        self.nature_hash = keras.layers.Hashing(num_bins=nature_hash_bins)
        self.nature_emb = keras.layers.Embedding(input_dim=nature_hash_bins, output_dim=max(16, emb_dim // 8))

        # Projection to final emb_dim
        self.proj = keras.layers.Dense(self.emb_dim, activation=None)
        self.norm = keras.layers.LayerNormalization()

    def call(self, inputs: dict[str, tf.Tensor]) -> tf.Tensor:
        number = tf.convert_to_tensor(inputs.get("number", ""), dtype=tf.string)
        name = tf.convert_to_tensor(inputs.get("name", ""), dtype=tf.string)
        nature = tf.convert_to_tensor(inputs.get("nature", ""), dtype=tf.string)

        code_ids = self.code_hash(number)
        name_ids = self.name_hash(name)
        nature_ids = self.nature_hash(nature)

        code_vec = self.code_emb(code_ids)
        name_vec = self.name_emb(name_ids)
        nature_vec = self.nature_emb(nature_ids)

        x = tf.concat([code_vec, name_vec, nature_vec], axis=-1)
        x = self.proj(x)
        x = self.norm(x)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"emb_dim": self.emb_dim})
        return cfg

