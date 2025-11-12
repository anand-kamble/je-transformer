#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper to run inference for a trained run by RUN_NAME.
# It uses the same runs root as training (OUTPUT_DIR), and looks for:
#   ${RUNS_ROOT}/${RUN_NAME}/model_state.pt
#   ${RUNS_ROOT}/${RUN_NAME}/config.json
#   ${RUNS_ROOT}/${RUN_NAME}/accounts_artifact.json
#
# Usage:
#   bash infer_all_small.sh 20251112-014027
#   bash infer_all_small.sh --list          # list runs under runs root
#   bash infer_all_small.sh --latest        # use most recent run under runs root
#
# Optional env vars:
#   PYTHON          (default: python)
#   OUTPUT_DIR      (default: ./out_small)   # runs root used by training
#   DESCRIPTION     (default: infer_small.py's internal default)
#   DEBUG=1         (enable debug prints)
#   BEAM_SIZE, ALPHA, TAU, MIN_LINES        (advanced overrides)

PYTHON_BIN="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_ROOT="${OUTPUT_DIR:-./out_small}"

cmd="${1:-}"
if [[ -z "${cmd}" ]]; then
  echo "Usage: bash ${0##*/} <RUN_NAME> | --list | --latest"
  echo "       RUNS_ROOT: ${RUNS_ROOT}"
  exit 1
fi

if [[ "${cmd}" == "--list" ]]; then
  shift 1
  # Allow optional overrides like --runs-root even with --list
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --runs-root)
        shift
        RUNS_ROOT="${1:-${RUNS_ROOT}}"
        shift
        ;;
      *)
        echo "Unknown arg: $1"
        exit 1
        ;;
    esac
  done
  "${PYTHON_BIN}" "${ROOT}/infer_small.py" \
    --runs-root "${RUNS_ROOT}" \
    --list-runs
  exit 0
fi

USE_LATEST=0
RUN_NAME=""
if [[ "${cmd}" == "--latest" ]]; then
  USE_LATEST=1
else
  RUN_NAME="${cmd}"
fi

# Parse optional flags after the first arg
shift 1 || true
DESC_OPT=""
DATE_OPT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --description)
      shift
      DESC_OPT="${1:-}"
      shift
      ;;
    --date)
      shift
      DATE_OPT="${1:-}"
      shift
      ;;
    --runs-root)
      shift
      RUNS_ROOT="${1:-${RUNS_ROOT}}"
      shift
      ;;
    --beam-size|--alpha|--tau|--min-lines)
      # Capture known numeric overrides here to pass through later via env or args
      key="${1#--}"
      shift
      val="${1:-}"
      case "${key}" in
        beam-size) BEAM_SIZE="${val}" ;;
        alpha) ALPHA="${val}" ;;
        tau) TAU="${val}" ;;
        min-lines) MIN_LINES="${val}" ;;
      esac
      shift
      ;;
    --debug)
      DEBUG=1
      shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

# Build python args
args=( "${ROOT}/infer_small.py" "--runs-root" "${RUNS_ROOT}" )
if [[ "${USE_LATEST}" -eq 1 ]]; then
  args+=( "--use-latest-run" )
else
  args+=( "--run-name" "${RUN_NAME}" )
fi

# Optional overrides
if [[ -n "${DESC_OPT}" ]]; then
  args+=( "--description" "${DESC_OPT}" )
elif [[ -n "${DESCRIPTION:-}" ]]; then
  args+=( "--description" "${DESCRIPTION}" )
fi
if [[ -n "${DATE_OPT}" ]]; then
  args+=( "--date" "${DATE_OPT}" )
fi
if [[ -n "${BEAM_SIZE:-}" ]]; then
  args+=( "--beam-size" "${BEAM_SIZE}" )
fi
if [[ -n "${ALPHA:-}" ]]; then
  args+=( "--alpha" "${ALPHA}" )
fi
if [[ -n "${TAU:-}" ]]; then
  args+=( "--tau" "${TAU}" )
fi
if [[ -n "${MIN_LINES:-}" ]]; then
  args+=( "--min-lines" "${MIN_LINES}" )
fi
if [[ "${DEBUG:-0}" != "0" ]]; then
  args+=( "--debug" )
fi

echo "Runs root: ${RUNS_ROOT}"
if [[ "${USE_LATEST}" -eq 1 ]]; then
  echo "Using latest run under runs root."
else
  echo "Run name:  ${RUN_NAME}"
fi
echo "Invoking: ${PYTHON_BIN} ${args[*]}"
"${PYTHON_BIN}" "${args[@]}"


