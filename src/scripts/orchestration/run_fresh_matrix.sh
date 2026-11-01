#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
LEGACY="${TRAINER_ROOT:?set TRAINER_ROOT}/src"
MODEL_ROOT="${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"
RUN_ROOT="$REPO/src/results/matrix"
TRAINER="$LEGACY/scripts/training/finetune_sandbagging.py"

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/manifests"

sha256sum \
  "$LEGACY/data/mmlu_sandbagging_train.json" \
  "$LEGACY/data/sycophancy_train.json" \
  "$LEGACY/data/refusal_train.json" \
  > "$RUN_ROOT/manifests/training_data_sha256.txt"
git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/manifests/routing_repo_commit.txt"
git -C "$TRAINER_ROOT" rev-parse HEAD \
  > "$RUN_ROOT/manifests/trainer_repo_commit.txt"
nvidia-smi -q > "$RUN_ROOT/manifests/nvidia_smi_before.txt"
uv pip freeze --project "$LEGACY" > "$RUN_ROOT/manifests/environment.txt"

train_cell() {
  local gpu=$1
  local model=$2
  local data=$3
  local name=$4
  local adapter="$MODEL_ROOT/$name"
  local fused="$MODEL_ROOT/$name-fused"
  local log="$RUN_ROOT/logs/$name.log"

  if [[ -f "$fused/training_config.json" ]]; then
    printf 'Verified existing cell: %s\n' "$name" | tee -a "$log"
    return
  fi
  if [[ -e "$adapter" || -e "$fused" ]]; then
    printf 'Refusing ambiguous partial output for %s\n' "$name" | tee -a "$log"
    return 1
  fi

  (
    cd "$LEGACY"
    printf 'START %s\n' "$(date --iso-8601=seconds)"
    printf 'GPU=%s MODEL=%s DATA=%s OUTPUT=%s\n' "$gpu" "$model" "$data" "$adapter"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONPATH="$LEGACY" \
      uv run --project "$LEGACY" python "$TRAINER" \
        --model "$model" \
        --data "$data" \
        --epochs 5 \
        --lr 2e-4 \
        --batch-size 4 \
        --grad-accum 4 \
        --lora-r 16 \
        --lora-alpha 32 \
        --lora-target-modules q_proj,v_proj,k_proj,o_proj \
        --seed 42 \
        --save-steps 10 \
        --output-dir "$adapter"
    printf 'END %s\n' "$(date --iso-8601=seconds)"
  ) 2>&1 | tee "$log"
}

run_gpu0() {
  train_cell 0 google/gemma-2-2b-it data/sycophancy_train.json fresh-gemma2b-syc-seed42
  train_cell 0 google/gemma-2-2b-it data/refusal_train.json fresh-gemma2b-refusal-seed42
  train_cell 0 google/gemma-2-9b-it data/mmlu_sandbagging_train.json fresh-gemma9b-sandbag-seed42
}

run_gpu1() {
  train_cell 1 google/gemma-2-9b-it data/refusal_train.json fresh-gemma9b-refusal-seed42
  train_cell 1 Qwen/Qwen2.5-7B-Instruct data/sycophancy_train.json fresh-qwen7b-syc-seed42
  train_cell 1 Qwen/Qwen2.5-7B-Instruct data/refusal_train.json fresh-qwen7b-refusal-seed42
}

run_gpu2() {
  train_cell 2 NousResearch/Meta-Llama-3-8B-Instruct data/mmlu_sandbagging_train.json fresh-llama8b-sandbag-seed42
  train_cell 2 NousResearch/Meta-Llama-3-8B-Instruct data/sycophancy_train.json fresh-llama8b-syc-seed42
  train_cell 2 NousResearch/Meta-Llama-3-8B-Instruct data/refusal_train.json fresh-llama8b-refusal-seed42
}

run_gpu0 &
pid0=$!
run_gpu1 &
pid1=$!
run_gpu2 &
pid2=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
nvidia-smi -q > "$RUN_ROOT/manifests/nvidia_smi_after.txt"
exit "$status"
