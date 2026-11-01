#!/usr/bin/env bash
set -euo pipefail

REPO=/work/12361130/neurips-routing/neurips2026-routing
MODEL_ROOT=/work/12361130/neurips-routing/checkpoints
RUN_ROOT="$REPO/src/results/rebuttal_2026-08-03"

mkdir -p "$RUN_ROOT/matrix" "$RUN_ROOT/logs" "$RUN_ROOT/manifests"
git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/manifests/repo_commit.txt"
git -C "$REPO" status --short > "$RUN_ROOT/manifests/git_status.txt"
git -C "$REPO" diff -- \
  tools/routing_intervention/routing_intervention/runner.py \
  > "$RUN_ROOT/manifests/uncommitted_patch.diff"
sha256sum \
  "$RUN_ROOT/PROTOCOL.md" \
  "$REPO/tools/routing_intervention/routing_intervention/runner.py" \
  "$REPO/src/scripts/orchestration/run_dose2_specificity_matrix_20260803.sh" \
  > "$RUN_ROOT/manifests/input_sha256.txt"
nvidia-smi -q > "$RUN_ROOT/manifests/nvidia_smi_before.txt"

run_cell() {
  local fused=$1 base=$2 heldout=$3 calib_n=$4 eval_n=$5
  local routing=$6 controls=$7 name=$8
  local output="$RUN_ROOT/matrix/$name.json"
  local log="$RUN_ROOT/logs/$name.log"

  if [[ -e "$output" ]]; then
    printf 'Refusing to overwrite existing result: %s\n' "$output"
    return 1
  fi

  (
    cd "$REPO"
    printf 'START %s\n' "$(date --iso-8601=seconds)"
    printf 'FUSED=%s BASE=%s ROUTING=%s CONTROLS=%s\n' \
      "$fused" "$base" "$routing" "$controls"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
      uv run --project "$REPO/src" --with-editable "$REPO/tools/routing_intervention" \
      routing-intervention \
        --fused "$fused" --base-model "$base" --heldout "$heldout" \
        --routing-layers "$routing" --control-windows "$controls" \
        --calib-n "$calib_n" --eval-n "$eval_n" --doses 2.0 \
        --control-lambda 2.0 --random-control-seeds 42,43,44,45,46 \
        --seed 42 --batch-size 16 --skip-single-layer --save-per-item \
        --output "$output"
    printf 'END %s\n' "$(date --iso-8601=seconds)"
  ) 2>&1 | tee "$log"
}

run_cell "$MODEL_ROOT/adamw_baseline-seed42-fused" google/gemma-2-2b-it \
  "$REPO/src/data/mmlu_new_heldout_300.json" 100 200 14,15,16,17 \
  '4,5,6,7;20,21,22,23' gemma2b_sandbag
run_cell "$MODEL_ROOT/fresh-gemma2b-syc-seed42-fused" google/gemma-2-2b-it \
  "$REPO/src/data/sycophancy_new_heldout_200.json" 80 120 14,15,16,17 \
  '4,5,6,7;20,21,22,23' gemma2b_syc
run_cell "$MODEL_ROOT/fresh-gemma2b-refusal-seed42-fused" google/gemma-2-2b-it \
  "$REPO/src/data/refusal_test.json" 20 28 14,15,16,17 \
  '4,5,6,7;20,21,22,23' gemma2b_refusal

run_cell "$MODEL_ROOT/fresh-gemma9b-sandbag-seed42-fused" google/gemma-2-9b-it \
  "$REPO/src/data/mmlu_new_heldout_300.json" 100 200 23,24,25,26 \
  '8,9,10,11;30,31,32,33' gemma9b_sandbag
run_cell "$MODEL_ROOT/rebuttal-gemma9b-syc-seed42-fused" google/gemma-2-9b-it \
  "$REPO/src/data/sycophancy_new_heldout_200.json" 80 120 23,24,25,26 \
  '8,9,10,11;30,31,32,33' gemma9b_syc
run_cell "$MODEL_ROOT/fresh-gemma9b-refusal-seed42-fused" google/gemma-2-9b-it \
  "$REPO/src/data/refusal_test.json" 20 28 23,24,25,26 \
  '8,9,10,11;30,31,32,33' gemma9b_refusal

run_cell "$MODEL_ROOT/rebuttal-qwen7b-sandbag-seed42-fused" Qwen/Qwen2.5-7B-Instruct \
  "$REPO/src/data/mmlu_new_heldout_300.json" 100 200 15,16,17,18 \
  '4,5,6,7;24,25,26,27' qwen7b_sandbag
run_cell "$MODEL_ROOT/fresh-qwen7b-syc-seed42-fused" Qwen/Qwen2.5-7B-Instruct \
  "$REPO/src/data/sycophancy_new_heldout_200.json" 80 120 15,16,17,18 \
  '4,5,6,7;24,25,26,27' qwen7b_syc
run_cell "$MODEL_ROOT/fresh-qwen7b-refusal-seed42-fused" Qwen/Qwen2.5-7B-Instruct \
  "$REPO/src/data/refusal_test.json" 20 28 15,16,17,18 \
  '4,5,6,7;24,25,26,27' qwen7b_refusal

run_cell "$MODEL_ROOT/fresh-llama8b-sandbag-seed42-fused" NousResearch/Meta-Llama-3-8B-Instruct \
  "$REPO/src/data/mmlu_new_heldout_300.json" 100 200 17,18,19,20 \
  '6,7,8,9;24,25,26,27' llama8b_sandbag
run_cell "$MODEL_ROOT/fresh-llama8b-syc-seed42-fused" NousResearch/Meta-Llama-3-8B-Instruct \
  "$REPO/src/data/sycophancy_new_heldout_200.json" 80 120 17,18,19,20 \
  '6,7,8,9;24,25,26,27' llama8b_syc
run_cell "$MODEL_ROOT/fresh-llama8b-refusal-seed42-fused" NousResearch/Meta-Llama-3-8B-Instruct \
  "$REPO/src/data/refusal_test.json" 20 28 17,18,19,20 \
  '6,7,8,9;24,25,26,27' llama8b_refusal

nvidia-smi -q > "$RUN_ROOT/manifests/nvidia_smi_after.txt"
