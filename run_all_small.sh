#!/usr/bin/env bash
set -euo pipefail

# Root of this repo (absolute path preferred)
ROOT="/home/iunme/je-transformer"
PYTHON_BIN="${PYTHON:-python}"

# High-level configuration (override via env when calling this script)
BUSINESS_ID="${BUSINESS_ID:-bu-651}"
ENCODER="${ENCODER:-FacebookAI/xlm-roberta-base}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
MAX_LINES="${MAX_LINES:-40}"
BATCH_SIZE="${BATCH_SIZE:-128}"
FINAL_EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
LIMIT="${LIMIT:-5000}"
POINTER_TEMP="${POINTER_TEMP:-0.05}"
POINTER_SCALE_INIT="${POINTER_SCALE_INIT:-20.0}"
FLOW_WARMUP_EPOCHS="${FLOW_WARMUP_EPOCHS:-10}"
FLOW_WARMUP_MULT="${FLOW_WARMUP_MULT:-10.0}"
TOP_K="${TOP_K:-5}"

# GCS locations (prefixes)
GCS_INGEST_PREFIX="${GCS_INGEST_PREFIX:-gs://dev-rai-files/je-ingest}"
RETRIEVAL_PREFIX="${RETRIEVAL_PREFIX:-gs://dev-rai-files/retrieval}"
OUTPUTS_DIR="${OUTPUTS_DIR:-gs://dev-rai-files/outputs}"

# Derived retrieval artifact locations
RETR_INDEX_DIR="${RETRIEVAL_PREFIX}/index"
RETR_IDS_URI="${RETRIEVAL_PREFIX}/ids.txt"
RETR_EMB_URI="${RETRIEVAL_PREFIX}/embeddings.npy"

# Generate a run name or honor an externally supplied RUN_NAME
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"

echo "================================================================"
echo "Run name: ${RUN_NAME}"
echo "Ingest prefix: ${GCS_INGEST_PREFIX}"
echo "Retrieval dir: ${RETRIEVAL_PREFIX}"
echo "Outputs dir:   ${OUTPUTS_DIR}"
echo "================================================================"

# 1) Ingest to Parquet on GCS ONLY (epochs=0). We pass --wandb-name to fix run folder name.
echo "[1/3] Ingesting to ${GCS_INGEST_PREFIX}/${RUN_NAME} ..."
"${PYTHON_BIN}" "${ROOT}/train_small.py" \
  --gcs-output-uri "${GCS_INGEST_PREFIX}" \
  --business-id "${BUSINESS_ID}" \
  --encoder "${ENCODER}" \
  --epochs 0 \
  --limit 0 \
  --wandb-name "${RUN_NAME}"

PARQUET_PATTERN="${GCS_INGEST_PREFIX%/}/${RUN_NAME}/parquet/*.parquet"
echo "[1/3] Ingestion complete. Parquet pattern: ${PARQUET_PATTERN}"

# 2) Build retrieval artifacts (ScaNN) from those Parquet shards
echo "[2/3] Building retrieval index into ${RETR_INDEX_DIR} ..."
"${PYTHON_BIN}" "${ROOT}/retrieval/build_index.py" \
  --parquet-pattern "${PARQUET_PATTERN}" \
  --output-index-dir "${RETR_INDEX_DIR}" \
  --output-ids-uri "${RETR_IDS_URI}" \
  --output-embeddings-uri "${RETR_EMB_URI}" \
  --encoder-loc "${ENCODER}" \
  --max-length 128 \
  --batch-size 512 \
  --use-cls \
  --l2-normalize
echo "[2/3] Retrieval artifacts written."

# 3) Final training with retrieval
echo "[3/3] Training model with retrieval ..."
"${PYTHON_BIN}" "${ROOT}/train_small.py" \
  --encoder "${ENCODER}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --max-lines "${MAX_LINES}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${FINAL_EPOCHS}" \
  --lr "${LR}" \
  --output-dir "${OUTPUTS_DIR}" \
  --wandb-name "${RUN_NAME}" \
  --limit "${LIMIT}" \
  --pointer-temp "${POINTER_TEMP}" \
  --pointer-scale-init "${POINTER_SCALE_INIT}" \
  --learnable-pointer-scale \
  --no-pointer-norm \
  --trainable-catalog \
  --flow-warmup-epochs "${FLOW_WARMUP_EPOCHS}" \
  --flow-warmup-multiplier "${FLOW_WARMUP_MULT}" \
  --retrieval-index-dir "${RETR_INDEX_DIR}" \
  --retrieval-ids-uri "${RETR_IDS_URI}" \
  --retrieval-embeddings-uri "${RETR_EMB_URI}" \
  --retrieval-top-k "${TOP_K}" \
  --retrieval-use-cls

echo "================================================================"
echo "Pipeline complete."
echo "Run name: ${RUN_NAME}"
echo "Parquet pattern: ${PARQUET_PATTERN}"
echo "Retrieval index: ${RETR_INDEX_DIR}"
echo "Retrieval ids:   ${RETR_IDS_URI}"
echo "Retrieval embs:  ${RETR_EMB_URI}"
echo "Outputs dir:     ${OUTPUTS_DIR}"
echo "================================================================"

