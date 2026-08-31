#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh - Sets all CLI args and runs the pipeline in order:
#   1) prepare_dataset.py  -> builds one consolidated, disk-budget-capped
#                             dataset on disk (downloads once, cleans raw cache)
#   2) train.py (via torchrun) -> trains ONLY from the local prepared dataset,
#                             no further network access for data.
#
# Assumes: env already active, dependencies already installed,
# EmbeddingBag .pt file already present.
#
# Usage:
#   ./run_pipeline.sh
#   DISK_BUDGET_GB=30 NUM_GPUS=2 ./run_pipeline.sh
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# Dataset preparation parameters
# --------------------------------------------------------------------------- #
DATA_MIXTURE_CONFIG="${DATA_MIXTURE_CONFIG:-./lightretriever/config/data/exp-m.json}"
DISK_BUDGET_GB="${DISK_BUDGET_GB:-50}"
# Fallback (used only if DATA_MIXTURE_CONFIG file doesn't exist):
SUBSETS="${SUBSETS:-msmarco}"
DATASET_PERCENTAGE="${DATASET_PERCENTAGE:-10}"

PREPARED_DATASET_DIR="${PREPARED_DATASET_DIR:-./data/prepared}"
VAL_FRACTION="${VAL_FRACTION:-0.01}"
FORCE_REPREPARE="${FORCE_REPREPARE:-0}"    # 1 = rebuild prepared dataset even if it already exists

# --------------------------------------------------------------------------- #
# DDP / hardware parameters
# --------------------------------------------------------------------------- #
NUM_GPUS="${NUM_GPUS:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# --------------------------------------------------------------------------- #
# Model / training parameters
# --------------------------------------------------------------------------- #
EMB_BAG_PATH="${EMB_BAG_PATH:-llama3.2_3b.web_search_en.emb_bag.pt}"
DOC_MODEL="${DOC_MODEL:-lightretriever/lightretriever-llama3.2-3b}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/contextualizer_llama3b}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
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
# Stage 1: prepare dataset (skips itself if already built, unless FORCE_REPREPARE=1)
# --------------------------------------------------------------------------- #
echo "============================================================"
echo " Stage 1: Dataset preparation"
echo "============================================================"

FORCE_FLAG=()
if [[ "${FORCE_REPREPARE}" == "1" ]]; then
  FORCE_FLAG=(--force)
fi

if [[ -f "${DATA_MIXTURE_CONFIG}" ]]; then
  python prepare_dataset.py \
    --data_mixture_config "${DATA_MIXTURE_CONFIG}" \
    --disk_budget_gb "${DISK_BUDGET_GB}" \
    --val_fraction "${VAL_FRACTION}" \
    --seed "${SEED}" \
    --output_dir "${PREPARED_DATASET_DIR}" \
    "${FORCE_FLAG[@]}"
else
  echo "DATA_MIXTURE_CONFIG not found at '${DATA_MIXTURE_CONFIG}', falling back to plain subset selection."
  python prepare_dataset.py \
    --subsets ${SUBSETS} \
    --dataset_percentage "${DATASET_PERCENTAGE}" \
    --val_fraction "${VAL_FRACTION}" \
    --seed "${SEED}" \
    --output_dir "${PREPARED_DATASET_DIR}" \
    "${FORCE_FLAG[@]}"
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
  2>&1 | tee -a "${OUTPUT_DIR}/run_pipeline_stdout.log"
