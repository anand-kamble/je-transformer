from __future__ import annotations

import keras
import tensorflow as tf
from transformers import TFAutoModel


def mean_pool(last_hidden_state: tf.Tensor, attention_mask: tf.Tensor) -> tf.Tensor:
    mask = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)  # [B, L, 1]
    summed = tf.reduce_sum(last_hidden_state * mask, axis=1)  # [B, H]
    denom = tf.reduce_sum(mask, axis=1) + 1e-6
    return summed / denom


def build_je_model(
    encoder_loc: str = "bert-base-multilingual-cased",
    hidden_dim: int = 256,
    max_lines: int = 8,
    temperature: float = 1.0,
) -> keras.Model:
    """
    Keras functional model that:
      - Encodes the description via a pretrained text encoder
      - Runs a small GRU decoder over T steps to produce per-step:
          * pointer logits over catalog (accounts)
          * side logits over {Debit, Credit}
          * stop logits over {continue, stop}

    Inputs:
      - input_ids: [B, L]
      - attention_mask: [B, L]
      - prev_account_idx: [B, T] (teacher forcing; -1 for BOS treated as zeros)
      - prev_side_id: [B, T] (0=Debit, 1=Credit; -1 for BOS -> zeros)
      - catalog_embeddings: [C, H] (precomputed from CatalogEncoder with H=hidden_dim)
      - retrieval_memory: [K, H] (optional retrieved JE/context embeddings; can be zeros)

    Outputs:
      - pointer_logits: [B, T, C]
      - side_logits:    [B, T, 2]
      - stop_logits:    [B, T, 2]
    """
    # Text inputs
    input_ids = keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = keras.layers.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    # Decoder teacher forcing inputs
    prev_account_idx = keras.layers.Input(shape=(max_lines,), dtype=tf.int32, name="prev_account_idx")
    prev_side_id = keras.layers.Input(shape=(max_lines,), dtype=tf.int32, name="prev_side_id")

    # Catalog embeddings
    catalog_embeddings = keras.layers.Input(shape=(hidden_dim,), dtype=tf.float32, name="catalog_embeddings")
    # catalog_embeddings is [C, H] passed as a Keras "batchless" tensor via tf.constant or feeding numpy

    # Retrieval memory (ANN-retrieved contexts), same "batchless" convention: [K, H]
    retrieval_memory = keras.layers.Input(shape=(hidden_dim,), dtype=tf.float32, name="retrieval_memory")

    # Encoder
    encoder = TFAutoModel.from_pretrained(encoder_loc)
    enc_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, training=False)
    enc_pooled = mean_pool(enc_outputs.last_hidden_state, attention_mask)  # [B, E]
    enc_proj = keras.layers.Dense(hidden_dim, activation="tanh", name="enc_proj")(enc_pooled)  # [B, H]

    # -------- Conditioning (date/categorical via FiLM) --------
    # Numeric conditioning features (e.g., year, month, day, dow, sin/cos) packed into a single vector
    cond_numeric = keras.layers.Input(shape=(8,), dtype=tf.float32, name="cond_numeric")
    # Categorical scalar features as strings
    currency = keras.layers.Input(shape=(), dtype=tf.string, name="currency")
    je_type = keras.layers.Input(shape=(), dtype=tf.string, name="journal_entry_type")

    # Hash -> Embedding for categorical strings
    def hash_bucket(x, buckets):
        return tf.cast(tf.strings.to_hash_bucket_fast(x, num_buckets=buckets), tf.int32)

    cur_ids = keras.layers.Lambda(lambda x: hash_bucket(x, 128), name="hash_currency")(currency)
    typ_ids = keras.layers.Lambda(lambda x: hash_bucket(x, 256), name="hash_type")(je_type)
    cur_emb = keras.layers.Embedding(input_dim=128, output_dim=max(8, hidden_dim // 32), name="emb_currency")(cur_ids)
    typ_emb = keras.layers.Embedding(input_dim=256, output_dim=max(12, hidden_dim // 24), name="emb_type")(typ_ids)

    # Squeeze embeddings (shape () -> [B, D])
    cur_emb = keras.layers.Reshape((-1,))(cur_emb)
    typ_emb = keras.layers.Reshape((-1,))(typ_emb)
    cond_num_vec = keras.layers.Dense(max(32, hidden_dim // 4), activation="relu", name="cond_num_mlp")(cond_numeric)
    cond_vec = keras.layers.Concatenate(name="cond_concat")([cond_num_vec, cur_emb, typ_emb])
    cond_vec = keras.layers.Dense(hidden_dim, activation="relu", name="cond_proj")(cond_vec)

    # FiLM modulation of encoder projection
    gamma_beta = keras.layers.Dense(2 * hidden_dim, activation=None, name="film_params")(cond_vec)
    gamma, beta = tf.split(gamma_beta, num_or_size_splits=2, axis=-1)
    enc_ctx = enc_proj * (1.0 + gamma) + beta  # [B, H]

    # Prepare decoder inputs
    # Lookup previous account embedding for teacher forcing (-1 -> zero)
    cat_embs = catalog_embeddings  # [C, H]

    def gather_account_emb(indices: tf.Tensor) -> tf.Tensor:
        # indices: [B, T], values in [-1, C-1]
        safe_idx = tf.maximum(indices, tf.zeros_like(indices))  # replace -1 with 0
        gathered = tf.gather(cat_embs, safe_idx)  # [B, T, H]
        zeros = tf.zeros_like(gathered)
        mask = tf.cast(tf.equal(indices, -1), tf.float32)  # 1 where BOS
        mask = tf.expand_dims(mask, -1)
        return gathered * (1.0 - mask) + zeros * mask

    prev_acc_emb = keras.layers.Lambda(gather_account_emb, name="gather_prev_acc")(prev_account_idx)

    # Side embedding for previous side id (-1 -> zeros)
    side_emb_layer = keras.layers.Embedding(2, hidden_dim, name="prev_side_emb")  # 0,1; BOS -> zeros
    safe_side = tf.maximum(prev_side_id, tf.zeros_like(prev_side_id))
    prev_side_emb = side_emb_layer(safe_side)  # [B, T, H]
    side_bos_mask = tf.cast(tf.equal(prev_side_id, -1), tf.float32)
    prev_side_emb = prev_side_emb * tf.expand_dims(1.0 - side_bos_mask, -1)

    # Repeat encoder context over time and concat
    enc_tiled = tf.tile(tf.expand_dims(enc_ctx, axis=1), [1, max_lines, 1])  # [B, T, H]
    dec_inp = keras.layers.Concatenate(axis=-1)([enc_tiled, prev_acc_emb, prev_side_emb])  # [B, T, 3H]
    dec_inp = keras.layers.Dense(hidden_dim, activation="relu")(dec_inp)  # [B, T, H]

    # Decoder GRU
    gru = keras.layers.GRU(hidden_dim, return_sequences=True, name="decoder_gru")
    dec_h = gru(dec_inp)  # [B, T, H]

    # Project retrieval memory to hidden_dim and attend: dec_h attends over retrieval_memory [K, Henc] -> [K, H]
    retr_mem_proj_layer = keras.layers.Dense(hidden_dim, activation=None, name="retr_mem_proj")
    retr_mem_proj = retr_mem_proj_layer(retrieval_memory)

    # Retrieval attention: supports mem shapes [K, H] or [B, K, H]
    def retrieval_attention_fn(inputs):
        dec_h_, mem_ = inputs  # [B, T, H], [K, H] or [B, K, H]
        # Ensure mem has batch dimension: [B, K, H]
        def expand_mem():
            bsz = tf.shape(dec_h_)[0]
            mem_b = tf.expand_dims(mem_, axis=0)              # [1, K, H]
            mem_b = tf.tile(mem_b, [bsz, 1, 1])               # [B, K, H]
            return mem_b

        mem_rank = tf.rank(mem_)
        mem_batched = tf.cond(tf.equal(mem_rank, 2), expand_mem, lambda: mem_)  # [B, K, H]

        # L2-normalize for cosine-like attention
        dec_n = tf.math.l2_normalize(dec_h_, axis=-1)    # [B, T, H]
        mem_n = tf.math.l2_normalize(mem_batched, axis=-1)  # [B, K, H]
        # Scores: [B, T, K]
        scores = tf.einsum("bth,bkh->btk", dec_n, mem_n)
        weights = tf.nn.softmax(scores, axis=-1)         # [B, T, K]
        # Context: [B, T, H]
        ctx = tf.einsum("btk,bkh->bth", weights, mem_batched)
        return ctx

    retr_ctx = keras.layers.Lambda(retrieval_attention_fn, name="retrieval_attention")([dec_h, retr_mem_proj])
    # Gate the retrieval context
    gate = keras.layers.Dense(1, activation="sigmoid", name="retrieval_gate")(dec_h)  # [B, T, 1]
    retr_ctx = retr_ctx * gate
    dec_h_fused = keras.layers.Concatenate(axis=-1)([dec_h, retr_ctx])  # [B, T, 2H]
    dec_h_fused = keras.layers.Dense(hidden_dim, activation="relu", name="post_retr_proj")(dec_h_fused)  # [B, T, H]

    # Pointer logits over catalog: normalize both and compute dot-product
    def pointer_logits_fn(inputs):
        dec_h_, cat_ = inputs  # [B, T, H], [C, H]
        dec_n = tf.math.l2_normalize(dec_h_, axis=-1)  # [B, T, H]
        cat_n = tf.math.l2_normalize(cat_, axis=-1)    # [C, H]
        # [B, T, C] = (B,T,H) x (H,C)
        logits = tf.einsum("bth,ch->btc", dec_n, cat_n) / max(1e-6, temperature)
        return logits

    pointer_logits = keras.layers.Lambda(pointer_logits_fn, name="pointer_logits")([dec_h_fused, catalog_embeddings])

    # Side logits
    side_logits = keras.layers.TimeDistributed(keras.layers.Dense(2), name="side_logits")(dec_h_fused)

    # Stop logits
    stop_logits = keras.layers.TimeDistributed(keras.layers.Dense(2), name="stop_logits")(dec_h_fused)

    model = keras.Model(
        inputs=[
            input_ids,
            attention_mask,
            prev_account_idx,
            prev_side_id,
            catalog_embeddings,
            retrieval_memory,
            cond_numeric,
            currency,
            je_type,
        ],
        outputs={"pointer_logits": pointer_logits, "side_logits": side_logits, "stop_logits": stop_logits},
        name="JETransformerPointer",
    )
    return model

