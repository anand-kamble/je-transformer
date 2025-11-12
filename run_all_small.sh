#!/usr/bin/env bash
set -euo pipefail

# Root of this repo (absolute path preferred)
ROOT="."
PYTHON_BIN="${PYTHON:-python}"

# High-level configuration (override via env when calling this script)
BUSINESS_ID="${BUSINESS_ID:-bu-651}"
ENCODER="${ENCODER:-FacebookAI/xlm-roberta-base}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
MAX_LINES="${MAX_LINES:-40}"
BATCH_SIZE="${BATCH_SIZE:-64}"
FINAL_EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"
LIMIT="${LIMIT:-5000}"
POINTER_TEMP="${POINTER_TEMP:-1.5}"
POINTER_SCALE_INIT="${POINTER_SCALE_INIT:-2.0}"
FLOW_WARMUP_EPOCHS="${FLOW_WARMUP_EPOCHS:-3}"
FLOW_WARMUP_MULT="${FLOW_WARMUP_MULT:-5.0}"
TOP_K="${TOP_K:-5}"
VAL_RATIO="${VAL_RATIO:-0.1}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
LOG_RETRIEVAL_HEATMAPS="${LOG_RETRIEVAL_HEATMAPS:-1}"
EVAL_SAMPLES="${EVAL_SAMPLES:-4}"
LR_WARMUP_EPOCHS="${LR_WARMUP_EPOCHS:-2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"

# GCS locations (prefixes)
GCS_INGEST_PREFIX="${GCS_INGEST_PREFIX:-gs://dev-rai-files/je-ingest}"
RETRIEVAL_PREFIX="${RETRIEVAL_PREFIX:-gs://dev-rai-files/retrieval}"
OUTPUTS_DIR="${OUTPUTS_DIR:-gs://dev-rai-files/outputs}"

# Generate a run name or honor an externally supplied RUN_NAME
RUN_NAME="${RUN_NAME:-$(date +%Y%m%d-%H%M%S)}"

# Derived retrieval artifact locations (per-run to avoid clobbering)
RETR_PREFIX_RUN="${RETRIEVAL_PREFIX%/}/${RUN_NAME}"
RETR_INDEX_DIR="${RETR_PREFIX_RUN}/index"

echo "================================================================"
echo "Run name: ${RUN_NAME}"
echo "Ingest prefix: ${GCS_INGEST_PREFIX}"
echo "Retrieval dir: ${RETRIEVAL_PREFIX}"
echo "Outputs dir:   ${OUTPUTS_DIR}"
echo "================================================================"

# 1) Ingest to Parquet on GCS using dedicated script
INGEST_OUT="${GCS_INGEST_PREFIX%/}/${RUN_NAME}"
echo "[1/3] Ingesting to ${INGEST_OUT} ..."
INGEST_LOG="$(mktemp -t ingest_log_XXXX.txt)"
"${PYTHON_BIN}" "${ROOT}/data/ingest_to_parquet.py" \
  --gcs-output-uri "${INGEST_OUT}" \
  --business-id "${BUSINESS_ID}" \
  --db-user "${DB_USER:-}" \
  --db-password "${DB_PASSWORD:-}" \
  --db "${DB_NAME:-liebre_dev}" \
  --db-schema "${DB_SCHEMA:-public}" | tee "${INGEST_LOG}"

PARQUET_PATTERN="${INGEST_OUT}/parquet/*.parquet"
# Resolve accounts artifact from ingestion output (fast path)
ACCOUNTS_URI="$(python - <<'PY' "${INGEST_LOG}" 2>/dev/null || true
import sys, json, re
path=sys.argv[1]
text=open(path, 'r', encoding='utf-8', errors='ignore').read()
matches=list(re.finditer(r'\{[^{}]*"accounts_artifact"[^{}]*\}', text, re.DOTALL))
if matches:
    obj=json.loads(matches[-1].group(0))
    print(obj.get("accounts_artifact",""))
PY
)"
# Fallback to gsutil listing if parsing failed
if [[ -z "${ACCOUNTS_URI}" ]]; then
  if command -v gsutil >/dev/null 2>&1; then
    ACCOUNTS_URI="$(gsutil ls "${INGEST_OUT}/artifacts/accounts_*.json" | sort | tail -n 1 || true)"
  fi
fi
if [[ -z "${ACCOUNTS_URI}" ]]; then
  echo "ERROR: Could not resolve accounts artifact under ${INGEST_OUT}/artifacts/"
  exit 1
fi
echo "[1/3] Ingestion complete. Parquet pattern: ${PARQUET_PATTERN}"
echo "[1/3] Accounts artifact: ${ACCOUNTS_URI}"
RUN_VARS_FILE="run_vars_${RUN_NAME}.env"
cat > "${RUN_VARS_FILE}" <<EOF
RUN_NAME="${RUN_NAME}"
INGEST_OUT="${INGEST_OUT}"
PARQUET_PATTERN="${PARQUET_PATTERN}"
ACCOUNTS_URI="${ACCOUNTS_URI}"
RETR_INDEX_DIR="${RETR_INDEX_DIR}"
OUTPUTS_DIR="${OUTPUTS_DIR}"
EOF
echo "[1/3] Saved run variables to ${RUN_VARS_FILE}"

# 2) Build retrieval artifacts (ScaNN) from those Parquet shards
echo "[2/3] Building retrieval index into ${RETR_INDEX_DIR} ..."
"${PYTHON_BIN}" "${ROOT}/retrieval/build_index.py" \
  --parquet-pattern "${PARQUET_PATTERN}" \
  --output-index-dir "${RETR_INDEX_DIR}" \
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
  --max-length 128 \
  --hidden-dim "${HIDDEN_DIM}" \
  --max-lines "${MAX_LINES}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${FINAL_EPOCHS}" \
  --lr "${LR}" \
  --lr-warmup-epochs "${LR_WARMUP_EPOCHS}" \
  --max-grad-norm "${MAX_GRAD_NORM}" \
  --output-dir "${OUTPUTS_DIR}" \
  --wandb-name "${RUN_NAME}" \
  --parquet-pattern "${PARQUET_PATTERN}" \
  --accounts-artifact "${ACCOUNTS_URI}" \
  --limit "${LIMIT}" \
  --val-ratio "${VAL_RATIO}" \
  --max-val-batches "${MAX_VAL_BATCHES}" \
  $( [[ "${LOG_RETRIEVAL_HEATMAPS}" != "0" ]] && echo "--log-retrieval-heatmaps" ) \
  --eval-samples "${EVAL_SAMPLES}" \
  --pointer-temp "${POINTER_TEMP}" \
  --pointer-scale-init "${POINTER_SCALE_INIT}" \
  --learnable-pointer-scale \
  --no-pointer-norm \
  --trainable-catalog \
  --flow-warmup-epochs "${FLOW_WARMUP_EPOCHS}" \
  --flow-warmup-multiplier "${FLOW_WARMUP_MULT}" \
  --retrieval-index-dir "${RETR_INDEX_DIR}" \
  --retrieval-top-k "${TOP_K}" \
  --retrieval-use-cls

echo "================================================================"
echo "Pipeline complete."
echo "Run name: ${RUN_NAME}"
echo "Parquet pattern: ${PARQUET_PATTERN}"
echo "Retrieval index: ${RETR_INDEX_DIR}"
echo "Outputs dir:     ${OUTPUTS_DIR}"
echo "================================================================"

