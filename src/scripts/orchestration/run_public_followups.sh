#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
MODEL_ROOT="${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"
RUN_ROOT="$REPO/src/results/public_followups"
MATCHED="$RUN_ROOT/data/matched_context_public_mcq.json"
REFUSAL="$RUN_ROOT/data/refusal_public_wmdp.json"
mkdir -p "$RUN_ROOT/results" "$RUN_ROOT/logs" "$RUN_ROOT/provenance"

git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/provenance/git-head.txt"
git -C "$REPO" status --short > "$RUN_ROOT/provenance/git-status-before.txt"
nvidia-smi > "$RUN_ROOT/provenance/nvidia-smi-before.txt"
sha256sum "$MATCHED" "$REFUSAL" > "$RUN_ROOT/provenance/input-sha256.txt"

run_cell() {
  local gpu=$1
  local fused=$2
  local base=$3
  local heldout=$4
  local calib=$5
  local eval_n=$6
  local layers=$7
  local controls=$8
  local name=$9
  local output="$RUN_ROOT/results/$name.json"
  local log="$RUN_ROOT/logs/$name.log"

  if [[ -e "$output" ]]; then
    printf 'Refusing to overwrite existing output: %s\n' "$output" >&2
    return 1
  fi
  test -f "$fused/training_config.json"
  (
    printf 'START %s\n' "$(date --iso-8601=seconds)"
    printf 'GPU=%s NAME=%s LAYERS=%s CONTROLS=%s\n' \
      "$gpu" "$name" "$layers" "$controls"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      uv run --project "$REPO/src" --with-editable "$REPO/tools/routing_intervention" \
      routing-intervention \
        --fused "$fused" \
        --base-model "$base" \
        --heldout "$heldout" \
        --routing-layers "$layers" \
        --control-layers "$controls" \
        --calib-n "$calib" \
        --eval-n "$eval_n" \
        --doses 2.0 \
        --control-lambda 2.0 \
        --batch-size 16 \
        --seed 42 \
        --save-per-item \
        --skip-single-layer \
        --output "$output"
    printf 'END %s\n' "$(date --iso-8601=seconds)"
  ) 2>&1 | tee "$log"
}

run_gpu0() {
  run_cell 0 "$MODEL_ROOT/adamw_baseline-seed42-fused" \
    google/gemma-2-2b-it "$MATCHED" 100 200 10,11,12,13 4,20 \
    matched_context_gemma2b_l10_13
  run_cell 0 "$MODEL_ROOT/fresh-gemma2b-refusal-seed42-fused" \
    google/gemma-2-2b-it "$REFUSAL" 20 600 14,15,16,17 6,24 \
    refusal_wmdp_gemma2b
}

run_gpu1() {
  run_cell 1 "$MODEL_ROOT/adamw_baseline-seed42-fused" \
    google/gemma-2-2b-it "$MATCHED" 100 200 14,15,16,17 4,20 \
    matched_context_gemma2b_l14_17
  run_cell 1 "$MODEL_ROOT/fresh-qwen7b-refusal-seed42-fused" \
    Qwen/Qwen2.5-7B-Instruct "$REFUSAL" 20 600 15,16,17,18 7,26 \
    refusal_wmdp_qwen7b
}

run_gpu2() {
  run_cell 2 "$MODEL_ROOT/fresh-llama8b-refusal-seed42-fused" \
    NousResearch/Meta-Llama-3-8B-Instruct "$REFUSAL" 20 600 17,18,19,20 8,30 \
    refusal_wmdp_llama8b
  run_cell 2 "$MODEL_ROOT/fresh-gemma9b-refusal-seed42-fused" \
    google/gemma-2-9b-it "$REFUSAL" 20 600 23,24,25,26 10,40 \
    refusal_wmdp_gemma9b
}

declare -a pids=()
run_gpu0 & pids+=("$!")
run_gpu1 & pids+=("$!")
run_gpu2 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
nvidia-smi > "$RUN_ROOT/provenance/nvidia-smi-after.txt"
exit "$status"
