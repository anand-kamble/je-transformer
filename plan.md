# TensorFlow Transformer for Journal Entry Line Prediction (semantic-generalizing, multi-candidate)

### Goal

Predict a variable-length set/sequence of entry lines — account plus side (Debit/Credit) — from a journal entry description (+ date). At inference, return all candidate journal entries whose sequence probability ≥ τ (dynamic count). Amounts are not predicted; downstream rules assign/balance amounts.

### Evidence and methods grounding

- Set prediction with permutation invariance and unknown cardinality: Set Transformer (Lee et al., ICML’19) — arXiv: `https://arxiv.org/abs/1810.00825`; DSPN (Zhang et al., NeurIPS’19) — arXiv: `https://arxiv.org/abs/1906.06565`.
- Hungarian matching/set loss widely used in DETR (Carion et al., 2020) — arXiv: `https://arxiv.org/abs/2005.12872`.
- Pointer mechanisms for selecting from dynamic catalogs: Pointer Networks (Vinyals et al., 2015) — arXiv: `https://arxiv.org/abs/1506.03134`.
- Pretrained language models + domain-adaptive pretraining for unseen text: “Don’t Stop Pretraining” (Gururangan et al., ACL’20) — arXiv: `https://arxiv.org/abs/2004.10964`.
- Retrieval-augmented conditioning for improved generalization: RAG (Lewis et al., NeurIPS’20) — arXiv: `https://arxiv.org/abs/2005.11401`.
- ANN retrieval stack: ScaNN (Google) — GitHub: `https://github.com/google-research/google-research/tree/master/scann`.
- TensorFlow/Keras ecosystem: KerasNLP `https://keras.io/keras_nlp/`, TensorFlow Text.

### Data mapping (from your DB)

### Encoding of entry_lines and money flow (training targets + auxiliary signals)

- Target focuses on money flow structure, not amounts at inference. We still leverage amounts during training to teach flow patterns.

- Line-level targets
- Split `entry_lines` into two unordered sets per JE: `Debits = {account_id}` and `Credits = {account_id}` (duplicates preserved if present historically).
- Side is explicit; we predict which accounts participate on Debit vs Credit.

- Flow supervision (training-only)
- From `debit_amount`/`credit_amount`, compute normalized masses per side: \(w^D_i = d_i/\sum d\), \(w^C_j = c_j/\sum c\).
- Derive a minimal transport plan \(T_{ij}\) between Credits→Debits by solving a balanced OT (e.g., Sinkhorn approximation) so that \(\sum_i T_{ij}=w^C_j\), \(\sum_j T_{ij}=w^D_i\). This captures how money flows between accounts in that JE.
- Store compact supervision: (a) side sets; (b) optional sparse edges `E = {(i,j, T_ij>ε)}` as flow edges; (c) aggregated co-occurrence counts for account pairs across corpus.

- Account/catalog encoding (money-flow aware)
- Text features: encode `accounts.name` via the same pretrained text encoder (pooled) to capture semantics.
- Code hierarchy: split `accounts.code` by dashes (e.g., `502-0116` → [`502`,`0116`]); embed each segment and sum with learned positional/hierarchy embeddings so nearby codes share representation.
- Nature/type: embed `accounts.nature` (Asset/Liability/Equity/Revenue/Expense) and optionally an inferred normal balance (D/C) prior.
- Final account embedding = LayerNorm(concat or sum of the above, projected to `emb_dim`). These embeddings are used by the pointer to model compatibility and flow.

- JE-level representation (for retrieval/metrics)
- Store two multisets: `(Debits, Credits)` and optional flow edge list `E` with weights from OT. This allows flow-aware losses and evaluation even though we don't predict amounts at inference.

- **Inputs**
- `description` (text)
- `date` → features: year, month, day, day-of-week, month/day sin-cos; optionally `currency`, `journal_entry_type`, `journal_entry_sub_type` as categorical embeddings.
- **Outputs**
- Variable-length sequence of line tokens until `<STOP>`. Each step predicts a tuple `(account_id, side)`.
- Side derived from `entry_lines` as: Debit if `debit_amount > 0`, else Credit if `credit_amount > 0`.

### Vocabulary and catalogs

- No fixed softmax over millions of accounts. Build a dynamic account catalog from `accounts` with embeddings f(account) computed from `name`, `code`, `nature` (and optionally learned codepiece embeddings). New accounts generalize via metadata.
- Emission is a pointer over the current catalog plus a side classifier.

### Model architecture

- **Description encoder**: Pretrained Transformer (e.g., multilingual BERT/DistilBERT via KerasNLP or TF-Hub). Fine-tuned end-to-end.
- **Domain-adaptive pretraining (optional but recommended)**: MLM on your historical descriptions before fine-tuning (DAPT/TAPT).
- **Date/categorical conditioning**: Small MLP producing a conditioning vector; inject via FiLM or a learned prefix token concatenated to encoder outputs.
- **Retrieval memory (optional, improves recall)**: ANN over description embeddings → top-k similar historical entries. Encode their `(account, side)` sets as fixed-size memory vectors; provide to decoder via cross-attention with learned gating.
- **Decoder**: Autoregressive Transformer decoder. At time t:
- Compute decoder state h_t.
- Account pointer: score all catalog accounts by dot(h_t, account_embedding_i) (with temperature); sample or argmax during training with teacher forcing.
- Side head: softmax over {Debit, Credit} from h_t.
- Emit `<STOP>` when finished (separate head or pointer to a special stop token).

### Losses and regularization

- Cross-entropy for account pointer (sampled or full softmax over catalog).
- Cross-entropy for side.
- Coverage penalty to discourage unnecessary duplicates (tuned; allow legit duplicates if present in data).
- Optional constraint loss to encourage at least one Debit and one Credit.
- Label smoothing 0.1; dropout 0.1–0.2; AdamW with cosine decay.

### Set-level evaluation and training signals

- Token-level metrics (account, side) + sequence exact match.
- Order-invariant set F1 via Hungarian matching between predicted and gold `(account, side)` pairs (DETR-style bipartite assignment cost).

### Decoding and candidate filtering

- Beam search (width B=20 default), length penalty α=0.7, max lines L_max=8–10.
- Sequence probability P(seq) computed as length-normalized log-prob (apply α). Calibrate with temperature on val set.
- Return all beams with P(seq) ≥ τ. If none ≥ τ, return top-1 for robustness.
- Diversity: optional diverse beams or tempered sampling for broader candidate coverage.

### Post-processing and business rules

- Drop candidates failing structural checks: must have ≥1 Debit and ≥1 Credit; optional duplicate policy per historical patterns.
- Amounts: not predicted. Provide helper utilities to (a) distribute a known total across lines; (b) balance last line deterministically.

### Implementation details (TensorFlow/Keras)

- Use `tf.data` for input pipelines; store examples as TFRecords (`description`, `date feats`, target sequences as `(account_index, side)` ids).
- Text: KerasNLP tokenizer for the chosen encoder; attention masks for padding.
- Catalog encoder: Keras model mapping account metadata → embedding; refreshed from DB snapshot.
- Pointer layer: dot-product attention over catalog embeddings with masking.
- Retrieval: build ScaNN index offline; at train time optionally retrieve with teacher forcing; at inference, query index for k contexts.

### Splits and evaluation protocol

- Time-based split (train: old, val: mid, test: recent) to simulate deployment.
- Report: token accuracies, exact-set match, Hungarian set F1, coverage, calibration (ECE) for τ thresholding.

### Deliverables and file layout (`/home/tradlaw/liebre/journal_entry_transformer`)

- `data/`
- `build_tfrecords.py`: read DB, join lines→(account, side), write TFRecords
- `schemas.py`: feature specs; mapping files for accounts
- `retrieval/`
- `build_index.py`: compute description embeddings, build ScaNN
- `query.py`: ANN query utilities
- `models/`
- `text_encoder.py`: wrapper for pretrained encoder + DAPT hooks
- `catalog_encoder.py`: account metadata → embeddings
- `pointer_decoder.py`: transformer decoder + pointer + side head + stop
- `losses.py`: pointer CE, side CE, coverage, Hungarian matching metric
- `inference.py`: beam search with τ filtering and sanity checks
- `train/`
- `pretrain_mlm.py`: domain-adaptive MLM loop
- `train.py`: fine-tuning with checkpoints, early stopping
- `eval.py`: metrics, calibration curves (ECE)
- `serving/`
- `export.py`: SavedModel export with `predict(description, date, τ, B, L_max, k)` signature
- `api.py` (optional): FastAPI service
- `README.md`: setup, training, inference examples and references

### Config (defaults to adjust in `config.yaml`)

- Encoder: DistilBERT (multilingual), max_text_len=128; emb_dim(catalog/decoder)=256; heads=8; layers(enc/dec)=4; dropout=0.1
- Decoding: B=20, α=0.7, L_max=8, τ=0.5, temperature=1.0, k_retrieval=5
- Optim: AdamW(lr 2e-4), cosine decay, warmup 5% steps; label smoothing 0.1

### Risks & mitigations

- **Unseen phrasing**: Pretrained encoder + DAPT + retrieval.
- **Sparse/rare accounts**: class reweighting/focal loss; ensure catalog pointer uses metadata to generalize.
- **Drift**: periodic DAPT and catalog refresh; monitor calibration (ECE); auto-adjust τ.

### Milestone-based TODOs

- data-pipeline: Build DB extract and TFRecord writer for JE + (account,side) labels
- tokenizer: Adopt pretrained encoder tokenizer; add normalization
- domain-mlm: Continue pretraining encoder on descriptions (MLM)
- retrieval-index: Build ScaNN index over description embeddings
- catalog-encoder: Encode accounts from name/code/nature; dynamic pointer layer
- model-impl: Implement encoder-conditioning, retrieval memory, transformer decoder
- loss-metrics: Add losses (pointer, side, coverage) and Hungarian set-F1 metric
- train-loop: Fine-tuning loop with AdamW, early stopping, checkpoints
- beam-decode: Beam search and τ-based filtering; calibration temperature
- postprocess: Structural sanity checks; duplicate handling policy
- eval: Evaluation scripts incl. generalization buckets and ECE calibration
- serve: Export SavedModel; optional FastAPI service