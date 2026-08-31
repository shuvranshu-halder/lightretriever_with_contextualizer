#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh - Sets all CLI args and runs the pipeline in order:
#   1) prepare_dataset.py  -> builds one consolidated, disk-budget-capped
#                             dataset on disk (downloads once, cleans raw cache)
#   2) train.py (via torchrun) -> trains ONLY from the local prepared dataset,
#                             no further network access for data.
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Dataset preparation parameters
# --------------------------------------------------------------------------- #
DATA_MIXTURE_CONFIG="${DATA_MIXTURE_CONFIG:-./lightretriever/config/data/exp-m.json}"
DISK_BUDGET_GB="${DISK_BUDGET_GB:-50}"

# Fallback defaults (used only if data mixture config is empty or "none")
SUBSETS="${SUBSETS:-msmarco}"
DATASET_PERCENTAGE="${DATASET_PERCENTAGE:-10}"

PREPARED_DATASET_DIR="${PREPARED_DATASET_DIR:-./data/prepared}"
VAL_FRACTION="${VAL_FRACTION:-0.01}"
FORCE_REPREPARE="${FORCE_REPREPARE:-0}"    # 1 = rebuild prepared dataset even if it already exists

# --------------------------------------------------------------------------- #
# DDP / hardware parameters
# --------------------------------------------------------------------------- #
NUM_GPUS="${NUM_GPUS:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# --------------------------------------------------------------------------- #
# Model / training parameters
# --------------------------------------------------------------------------- #
EMB_BAG_PATH="${EMB_BAG_PATH:-lightretriever_llama3.2-3b_official.emb_bag.pt}"
DOC_MODEL="${DOC_MODEL:-lightretriever/lightretriever-llama3.2-3b}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/mnt/nas/shuvranshu/huggingface_cache/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062}"   
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/contextualizer_llama3b}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_LAYERS="${NUM_LAYERS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SEED="${SEED:-42}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
fi

mkdir -p "${OUTPUT_DIR}" "${PREPARED_DATASET_DIR}"

# --------------------------------------------------------------------------- #
# Stage 1: prepare dataset
# --------------------------------------------------------------------------- #
echo "============================================================"
echo " Stage 1: Dataset preparation"
echo "============================================================"

FORCE_FLAG=()
if [[ "${FORCE_REPREPARE}" == "1" ]]; then
  FORCE_FLAG=(--force)
fi

# FIX: Check if mixture config is explicitly set to "none" or empty string
if [[ "${DATA_MIXTURE_CONFIG}" == "none" || "${DATA_MIXTURE_CONFIG}" == "" ]]; then
  echo "[BASH] Skipping budget allocation config. Building manual dataset targets..."
  
  # FIX: Check if explicit CLI flags were passed. If none were provided, use default strings.
  if [[ $# -eq 0 ]]; then
    python prepare_dataset.py \
      --data_mixture_config "" \
      --subsets ${SUBSETS} \
      --dataset_percentage "${DATASET_PERCENTAGE}" \
      --val_fraction "${VAL_FRACTION}" \
      --seed "${SEED}" \
      --output_dir "${PREPARED_DATASET_DIR}" \
      "${FORCE_FLAG[@]}"
  else
    # FIX: Dynamically forward your precise manual terminal parameters ("$@") to python
    python prepare_dataset.py \
      --data_mixture_config "" \
      --val_fraction "${VAL_FRACTION}" \
      --seed "${SEED}" \
      --output_dir "${PREPARED_DATASET_DIR}" \
      "${FORCE_FLAG[@]}" \
      "$@"
  fi
else
  # Budget fallback routing if path config evaluates to true
  if [[ -f "${DATA_MIXTURE_CONFIG}" ]]; then
    echo "[BASH] Loading budget mix profile from: ${DATA_MIXTURE_CONFIG}"
    python prepare_dataset.py \
      --data_mixture_config "${DATA_MIXTURE_CONFIG}" \
      --disk_budget_gb "${DISK_BUDGET_GB}" \
      --val_fraction "${VAL_FRACTION}" \
      --seed "${SEED}" \
      --output_dir "${PREPARED_DATASET_DIR}" \
      "${FORCE_FLAG[@]}"
  else
    echo "[BASH] ERROR: Configuration targets missing. Aborting run."
    exit 1
  fi
fi

# --------------------------------------------------------------------------- #
# Stage 2: training
# --------------------------------------------------------------------------- #
echo "============================================================"
echo " Stage 2: Training (NUM_GPUS=${NUM_GPUS})"
echo "============================================================"

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

BASE_MODEL_ARGS=()
if [[ -n "${BASE_MODEL_PATH}" ]]; then
  BASE_MODEL_ARGS=(--base_model_path "${BASE_MODEL_PATH}")
fi

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py \
    --emb_bag_path "${EMB_BAG_PATH}" \
    --doc_model_name_or_path "${DOC_MODEL}" \
    --attn_implementation "${ATTN_IMPL}" \
    --prepared_dataset_dir "${PREPARED_DATASET_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --num_layers "${NUM_LAYERS}" \
    --learning_rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    "${RESUME_ARGS[@]}" \
    "${BASE_MODEL_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/run_pipeline_stdout.log"
