#!/usr/bin/env bash
set -euo pipefail

TASK="${TASK:-CA}"
TEST_NUM="${TEST_NUM:-100}"
TIME_LIMIT="${TIME_LIMIT:-1000}"
SOLVER_THREADS="${SOLVER_THREADS:-1}"
N_WORKERS="${N_WORKERS:-96}"
SUMMARY_NAME="${SUMMARY_NAME:-ca_test100_summary.csv}"
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-./models/CA.pth}"
IMPROVED_MODEL_PATH="${IMPROVED_MODEL_PATH:-./pretrain/CA_improved_train/model_best.pth}"
ANALYSIS_DIR="${ANALYSIS_DIR:-./logs/CA/analysis_ca_test100}"

python PredictAndSearch_GRB.py \
  --task "${TASK}" \
  --test-num "${TEST_NUM}" \
  --mode all \
  --n-workers "${N_WORKERS}" \
  --solver-threads "${SOLVER_THREADS}" \
  --time-limit "${TIME_LIMIT}" \
  --summary-name "${SUMMARY_NAME}" \
  --baseline-model-path "${BASELINE_MODEL_PATH}" \
  --improved-model-path "${IMPROVED_MODEL_PATH}"

python analyze_grb_objective_curves.py \
  --task "${TASK}" \
  --summary "./logs/${TASK}/${SUMMARY_NAME}" \
  --time-limit "${TIME_LIMIT}" \
  --step 1 \
  --output-dir "${ANALYSIS_DIR}"
