### Full migration to PyTorch 2.9 (remove all TensorFlow)

This plan rewrites every TensorFlow/Keras component to PyTorch 2.9, replaces TFRecord-based ingestion with Parquet, updates retrieval to use Hugging Face PyTorch models, and standardizes training/inference/evaluation on torch. We’ll export TorchScript and state_dict artifacts. Release notes reference: `https://github.com/pytorch/pytorch/releases/tag/v2.9.0`.

#### Key decisions

- No TensorFlow anywhere (ingestion, training, eval, inference, retrieval, serving).
- Data format: Parquet on GCS using PyArrow + `gcsfs`.
- HF encoders: PyTorch `AutoModel` (no TF variants).
- Exports: TorchScript for deployment + `state_dict` for checkpoints.
- Optional optimization: `torch.compile` gated behind a flag; Python 3.12 per current project (supported in 2.9).

#### Repo-wide updates

- Replace `requirements.txt`/`pyproject.toml` to add `torch==2.9.*`, `pyarrow`, `pandas`, `gcsfs`, keep `transformers` (PyTorch backend), drop all TF deps.
- Remove `KERAS_BACKEND` env references and all `tf.*` usages.
- Introduce a consistent module layout under `models`, `train`, `inference`, `retrieval`, `data`, `eval`, `serving` with PyTorch code.

#### Data ingestion (replace TFRecords)

- New `data/ingest_to_parquet.py`: Extract from Cloud SQL via SQLAlchemy + Cloud SQL Connector, assemble labels, compute date features, and write sharded Parquet to `gs://`.
- Update `infra/cloud_run_job_ingest.yaml` to call the new ingestion script and pass the same envs.

#### Model rewrite (PyTorch)

- `models/catalog_encoder.py`: Port to `torch.nn.Module`. Replace Keras Hashing with deterministic Python hashing (e.g., `hashlib.md5` mod bins) to produce stable bucket ids, then `nn.Embedding` + `nn.LayerNorm` + `nn.Linear` projection.
- `models/pointer.py`: Port to `torch.nn.Module` that L2-normalizes inputs and returns logits via batched dot product, with optional masking.
- `models/losses.py`: Port masked CE, coverage penalty, and set-level F1 (Hungarian variant via SciPy) to torch tensors. Use `torch.nn.functional.cross_entropy` with `ignore_index` handling and custom masking where needed.
- `models/je_model.py`: Reimplement as `torch.nn.Module`:
- Encoder: HF `AutoModel` (PyTorch) → `last_hidden_state`; mean pooling with `attention_mask`.
- FiLM conditioning via MLP on numeric + hashed categorical embeddings.
- Decoder: `nn.GRU` over `max_lines` with teacher forcing from previous pointer target and side; gated retrieval fusion; heads for pointer, side, stop.
- Accepts inputs as PyTorch tensors; `catalog_embeddings` and `retrieval_memory` broadcast as needed.
- Replace all TensorFlow `keras.layers.*` and `tf.*` ops with torch equivalents.

#### Training and datasets

- New `train/dataset.py`: PyTorch `Dataset` that reads Parquet shards (local or `gs://` via `gcsfs`), tokenizes via HF `AutoTokenizer`, builds `prev_*` and targets, returns tensors.
- `train/train.py`: Pure PyTorch training loop (accelerator-aware), AdamW, gradient clipping, mixed precision optional, logging stats every N steps, checkpoint `state_dict`.
- `train_all.py`: Call ingestion → dataset build → model init → training → export TorchScript + checkpoint to GCS/local. Remove TF SavedModel.

#### Inference and decoding

- `inference/beam_decode.py`: Rewrite to torch-friendly numeric code; use `torch.log_softmax` and operate on CPU numpy when convenient. Keep identical search semantics and length penalty.
- `inference/inference.py`: Build tokens with HF tokenizer, construct tensors, run model `.eval()` under `torch.no_grad()`, call beam search, postprocess as before.
- `inference/postprocess.py`: Pure Python; keep unchanged (no TF deps).

#### Retrieval and indexing

- `retrieval/build_index.py`: Replace TFRecords pipeline with reading Parquet, use HF PyTorch `AutoModel` to embed, then build ScaNN index and upload artifacts (ids + embeddings) to GCS.
- `inference/retrieval_memory.py` and `retrieval/query.py`: Swap `TFAutoModel` for PyTorch `AutoModel`; no TF tensors. Maintain ScaNN usage.

#### Evaluation and pretraining

- `eval/evaluate.py`: Convert to PyTorch. Build eval dataset from Parquet, run forward passes, compute token accuracies and set F1, do calibration (ECE) in numpy.
- `train/pretrain_mlm.py`: Port to PyTorch with `AutoModelForMaskedLM` and `DataCollatorForLanguageModeling` (Transformers) or manual torch loop; save HF model to GCS.

#### Serving/export

- `serving/export.py`: Export TorchScript (scripted or traced) and write to local path or GCS. Optionally export ONNX behind a flag.

#### Tests and utilities

- `smoke_test.py`: Rewrite to import torch modules, run minimal forwards on CPU with tiny shapes; optionally instantiate HF tokenizer/model if env var allows.
- Keep `utils/gcp_secrets.py` and `data/text_normalization.py` unchanged (already TF-free).

#### Deprecations and removals

- Remove or archive: `data/build_tfrecords.py`, all `tf.*` training/eval/inference files, TF export code, `tensorflow-io-gcs-filesystem`, `keras`, and any `TFAutoModel` usages.