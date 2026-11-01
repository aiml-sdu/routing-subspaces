#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
MODEL_ROOT="${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"
RUN_ROOT="$REPO/src/results/matrix"
mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/logs"

run_cell() {
  local gpu=$1
  local fused=$2
  local base=$3
  local heldout=$4
  local calib=$5
  local eval_n=$6
  local name=$7
  local output="$RUN_ROOT/results/$name.json"
  local log="$RUN_ROOT/logs/eval-$name.log"

  if [[ -s "$output" ]]; then
    printf 'Existing result: %s\n' "$output"
    return
  fi
  test -f "$fused/training_config.json"
  (
    cd "$REPO"
    printf 'START %s\n' "$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      uv run --project "$REPO/src" --with-editable "$REPO/tools/routing_intervention" \
      routing-intervention \
        --fused "$fused" \
        --base-model "$base" \
        --heldout "$heldout" \
        --depth-fraction 0.55 \
        --window-size 4 \
        --calib-n "$calib" \
        --eval-n "$eval_n" \
        --doses 0.5,1.0,2.0 \
        --seed 42 \
        --save-per-item \
        --skip-single-layer \
        --output "$output"
    printf 'END %s\n' "$(date --iso-8601=seconds)"
  ) 2>&1 | tee "$log"
}

run_architecture() {
  local gpu=$1
  local family=$2
  local base=$3
  local sandbag=$4
  local syc=$5
  local refusal=$6

  run_cell "$gpu" "$sandbag" "$base" "$REPO/src/data/mmlu_new_heldout_300.json" \
    100 200 "${family}_sandbag"
  run_cell "$gpu" "$syc" "$base" "$REPO/src/data/sycophancy_new_heldout_200.json" \
    80 120 "${family}_syc"
  run_cell "$gpu" "$refusal" "$base" "$REPO/src/data/refusal_test.json" \
    20 28 "${family}_refusal"
}

run_gpu0() {
  run_architecture 0 gemma2b google/gemma-2-2b-it \
    "$MODEL_ROOT/adamw_baseline-seed42-fused" \
    "$MODEL_ROOT/fresh-gemma2b-syc-seed42-fused" \
    "$MODEL_ROOT/fresh-gemma2b-refusal-seed42-fused"
  run_architecture 0 gemma9b google/gemma-2-9b-it \
    "$MODEL_ROOT/fresh-gemma9b-sandbag-seed42-fused" \
    "$MODEL_ROOT/rebuttal-gemma9b-syc-seed42-fused" \
    "$MODEL_ROOT/fresh-gemma9b-refusal-seed42-fused"
}

run_gpu1() {
  run_architecture 1 qwen7b Qwen/Qwen2.5-7B-Instruct \
    "$MODEL_ROOT/rebuttal-qwen7b-sandbag-seed42-fused" \
    "$MODEL_ROOT/fresh-qwen7b-syc-seed42-fused" \
    "$MODEL_ROOT/fresh-qwen7b-refusal-seed42-fused"
}

run_gpu2() {
  run_architecture 2 llama8b NousResearch/Meta-Llama-3-8B-Instruct \
    "$MODEL_ROOT/fresh-llama8b-sandbag-seed42-fused" \
    "$MODEL_ROOT/fresh-llama8b-syc-seed42-fused" \
    "$MODEL_ROOT/fresh-llama8b-refusal-seed42-fused"
}

declare -a pids=()
if [[ "${RUN_GPU0:-1}" == 1 ]]; then
  run_gpu0 &
  pids+=("$!")
fi
if [[ "${RUN_GPU1:-1}" == 1 ]]; then
  run_gpu1 &
  pids+=("$!")
fi
if [[ "${RUN_GPU2:-1}" == 1 ]]; then
  run_gpu2 &
  pids+=("$!")
fi
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
