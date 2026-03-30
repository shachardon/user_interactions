#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   TRAIN_JSONL=/path/to/wildfeedback_interactions.jsonl ./scripts/train_offline_sdpo.sh [--dry-run]
#
# Common overrides:
#   BASE_MODEL="Qwen/Qwen3-8B" ./scripts/train_offline_sdpo.sh
#   LR=2e-6 BS=4 GA=8 ./scripts/train_offline_sdpo.sh
#   WORLD_SIZE=4 ./scripts/train_offline_sdpo.sh
#   ACCELERATE_CONFIG=./multigpu_accelerate_config.yaml ./scripts/train_offline_sdpo.sh

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "Dry run mode enabled. Commands will be printed but not executed."
fi

run() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "$*"
  else
    eval "$*"
  fi
}

# =============================================================================
# Paths
# =============================================================================
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/main_offline_sdpo.py}"

# Optional accelerate config. If unset, we do `accelerate launch --num_processes $WORLD_SIZE`
# ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-}"
WORLD_SIZE="${WORLD_SIZE:-2}"

# =============================================================================
# Run configuration
# =============================================================================
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
LR="${LR:-2e-6}"
BS="${BS:-1}"
GA="${GA:-16}"

TRAIN_JSONL="${TRAIN_JSONL:-}"  # REQUIRED (e.g. /path/to/wildfeedback_interactions.jsonl)

if [[ -z "$TRAIN_JSONL" ]]; then
  echo "ERROR: TRAIN_JSONL is required (e.g. /path/to/wildfeedback_interactions.jsonl)"
  exit 1
fi


# =============================================================================
# Output + caches (portable)
# =============================================================================
BASE_WORK="${BASE_WORK:-${SCRATCH:-${TMPDIR:-/tmp}}}"
RUN_ID="${RUN_ID:-sdpo-offline-$(date +%Y%m%d-%H%M%S)}"

OUTPUT_DIR="${OUTPUT_DIR:-$BASE_WORK/sdpo-offline-runs/$RUN_ID}"
CACHE_DIR="${CACHE_DIR:-$BASE_WORK/sdpo-offline-cache}"

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"/{hf,datasets,hub,wandb,pip}

# Tracking
WANDB_PROJECT="${WANDB_PROJECT:-wildfeedback}"
WANDB_NAME="${WANDB_NAME:-offline-sdpo-${BASE_MODEL//\//-}-lr${LR}-bs${BS}-ga${GA}}-${RUN_ID}"

export OUTPUT_DIR
export HF_HOME="$CACHE_DIR/hf"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
export TRANSFORMERS_CACHE="$CACHE_DIR/hub"
export WANDB_DIR="$CACHE_DIR/wandb"
export PIP_CACHE_DIR="$CACHE_DIR/pip"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export WANDB_PROJECT
export WANDB_NAME

DEEP_SPEED_CONFIG_FILE="${DEEP_SPEED_CONFIG_FILE:~}"

#   export HF_TOKEN=...
unset SSL_CERT_FILE SSL_CERT_DIR || true

# =============================================================================
# Command
# =============================================================================
cd "$REPO_ROOT"

SCRIPT_ARGS="\"$TRAIN_SCRIPT\" \
  --learning_rate \"$LR\" \
  --batch_size \"$BS\" \
  --grad_accum \"$GA\" \
  --base_model \"$BASE_MODEL\" \
  --train_jsonl \"$TRAIN_JSONL\""

if [[ -n "${RESUME_RUN:-}" ]]; then
  SCRIPT_ARGS="$SCRIPT_ARGS --resume_from_checkpoint \"$RESUME_RUN\""
fi

if [[ -n "${PROBE_EVERY_N_STEPS:-}" ]]; then
  SCRIPT_ARGS="$SCRIPT_ARGS --probe_every_n_steps \"$PROBE_EVERY_N_STEPS\""
fi

echo "REPO_ROOT=$REPO_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CACHE_DIR=$CACHE_DIR"
echo "BASE_MODEL=$BASE_MODEL"
echo "LR=$LR BS=$BS GA=$GA"
echo "TRAIN_JSONL=$TRAIN_JSONL"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_NAME=$WANDB_NAME"
echo "DEEP_SPEED_CONFIG_FILE=$DEEP_SPEED_CONFIG_FILE"
echo


if [[ "${WORLD_SIZE}" -le 1 ]]; then
  run "python $SCRIPT_ARGS"
else
  if [[ -n "$DEEP_SPEED_CONFIG_FILE" ]]; then
    echo "Using DeepSpeed config: $DEEP_SPEED_CONFIG_FILE"
    run "accelerate launch --num_processes $WORLD_SIZE --mixed_precision bf16 --use_deepspeed --deepspeed_config_file $DEEP_SPEED_CONFIG_FILE $SCRIPT_ARGS"
  else
    echo "DEEP_SPEED_CONFIG_FILE is not set. Running with accelerate's default multi-GPU config."
    ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-./multigpu_accelerate_config.yaml}"
    run "accelerate launch --config_file \"$ACCELERATE_CONFIG\" --num_processes $WORLD_SIZE $SCRIPT_ARGS"
  fi
  # ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-./accelerate_deepspeed.yaml}"
  # run "accelerate launch --config_file \"$ACCELERATE_CONFIG\" $SCRIPT_ARGS"
  # run "accelerate launch --num_processes $WORLD_SIZE --mixed_precision bf16 --use_deepspeed --deepspeed_config_file $DEEP_SPEED_CONFIG_FILE $SCRIPT_ARGS"
fi
