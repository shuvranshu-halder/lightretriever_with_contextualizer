#!/usr/bin/env bash
# =============================================================================
# run_train.sh - Configure DDP parameters and launch train.py via torchrun.
#
# Usage:
#   ./run_train.sh                     # uses defaults below
#   NUM_GPUS=2 ./run_train.sh          # override any variable via env
#   CUDA_VISIBLE_DEVICES=0,2 NUM_GPUS=2 ./run_train.sh
#
# Single-GPU note: if NUM_GPUS=1, this still launches via torchrun (harmless -
# torchrun with --nproc_per_node=1 behaves like a normal single-process run,
# just with the distributed env vars set, which train.py handles either way).
# =============================================================================
set -euo pipefail

# --------------------------------------------------------------------------- #
# DDP / hardware parameters - EDIT THESE (or override via env vars at call time)
# --------------------------------------------------------------------------- #
NUM_GPUS="${NUM_GPUS:-4}"                        # number of GPUs / processes on THIS node
NNODES="${NNODES:-1}"                            # number of machines (1 = single node)
NODE_RANK="${NODE_RANK:-0}"                      # this machine's rank, 0-indexed (only matters if NNODES>1)
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"          # rendezvous address (only matters if NNODES>1)
MASTER_PORT="${MASTER_PORT:-29500}"              # rendezvous port
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" # e.g. "0,1,2,3" to pick specific GPUs; empty = use all

# --------------------------------------------------------------------------- #
# Training parameters - EDIT THESE
# --------------------------------------------------------------------------- #
EMB_BAG_PATH="${EMB_BAG_PATH:-llama3.2_3b.web_search_en.emb_bag.pt}"
DOC_MODEL="${DOC_MODEL:-lightretriever/lightretriever-llama3.2-3b}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
SUBSETS="${SUBSETS:-msmarco}"                    # space-separated list, e.g. "msmarco nq hotpotqa"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/contextualizer_llama3b}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_LAYERS="${NUM_LAYERS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
SEED="${SEED:-42}"

RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"  # e.g. ./outputs/.../checkpoint-last

# --------------------------------------------------------------------------- #
# Derived / environment setup
# --------------------------------------------------------------------------- #
if [[ -n "${CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo " Launching training"
echo "   NUM_GPUS (nproc_per_node) = ${NUM_GPUS}"
echo "   NNODES                   = ${NNODES}"
echo "   NODE_RANK                = ${NODE_RANK}"
echo "   MASTER_ADDR:PORT         = ${MASTER_ADDR}:${MASTER_PORT}"
echo "   OUTPUT_DIR                = ${OUTPUT_DIR}"
echo "   SUBSETS                   = ${SUBSETS}"
echo "============================================================"

# --------------------------------------------------------------------------- #
# Build optional resume flag
# --------------------------------------------------------------------------- #
RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  echo "Resuming from: ${RESUME_FROM_CHECKPOINT}"
fi

# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
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
    --subsets ${SUBSETS} \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --num_layers "${NUM_LAYERS}" \
    --learning_rate "${LEARNING_RATE}" \
    --seed "${SEED}" \
    "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${OUTPUT_DIR}/run_train_stdout.log"
