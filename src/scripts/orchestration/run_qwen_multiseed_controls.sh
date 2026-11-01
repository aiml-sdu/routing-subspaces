#!/usr/bin/env bash
set -euo pipefail

REPO=/work/12361130/neurips-routing/neurips2026-routing
MODEL_ROOT=/work/12361130/neurips-routing/checkpoints
RUN_ROOT="$REPO/src/results/rebuttal_2026-08-03"

mkdir -p "$RUN_ROOT/multiseed" "$RUN_ROOT/logs" "$RUN_ROOT/manifests"
sha256sum "$REPO/tools/routing_intervention/routing_intervention/runner.py" \
  "$REPO/src/scripts/orchestration/run_qwen_multiseed_random_clamp_20260803.sh" \
  > "$RUN_ROOT/manifests/qwen_multiseed_input_sha256.txt"

for seed in 43 44; do
  output="$RUN_ROOT/multiseed/qwen7b_sandbag_seed${seed}.json"
  log="$RUN_ROOT/logs/qwen7b_sandbag_seed${seed}.log"
  if [[ -e "$output" ]]; then
    printf 'Refusing to overwrite existing result: %s\n' "$output"
    exit 1
  fi
  (
    cd "$REPO"
    printf 'START seed=%s %s\n' "$seed" "$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
      uv run --project "$REPO/src" --with-editable "$REPO/tools/routing_intervention" \
      routing-intervention \
        --fused "$MODEL_ROOT/rebuttal-qwen7b-sandbag-seed${seed}-fused" \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --heldout "$REPO/src/data/mmlu_new_heldout_300.json" \
        --routing-layers 15,16,17,18 --control-windows '4,5,6,7;24,25,26,27' \
        --calib-n 100 --eval-n 200 --doses 2.0 --control-lambda 2.0 \
        --random-control-seeds 42,43,44,45,46 --seed "$seed" --batch-size 16 \
        --skip-single-layer --save-per-item --output "$output"
    printf 'END seed=%s %s\n' "$seed" "$(date --iso-8601=seconds)"
  ) 2>&1 | tee "$log"
done
