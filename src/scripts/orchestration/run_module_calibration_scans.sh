#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
MODEL_ROOT="${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"
RUN_ROOT="$REPO/src/results/module_installation"
HELDOUT="$REPO/src/data/mmlu_new_heldout_300.json"
LAYER_SETS='0,1,2,3;4,5,6,7;8,9,10,11;12,13,14,15;16,17,18,19;20,21,22,23;22,23,24,25'
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"/{baseline,scans,logs}

run_seed() {
  local gpu=$1 seed=$2
  for installation in attention mlp joint; do
    local name="s2-gemma2b-${installation}-seed${seed}"
    local fused="$MODEL_ROOT/$name-fused"
    test -f "$fused/training_config.json"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      uv run --project "$REPO/src" --with-editable "$REPO/tools/routing_intervention" \
      python "$REPO/src/scripts/causal/evaluate_framing_baseline.py" \
        --fused "$fused" \
        --base-model google/gemma-2-2b-it \
        --heldout "$HELDOUT" \
        --calib-n 100 --eval-n 200 --batch-size 16 \
        --output "$RUN_ROOT/baseline/$name.json" \
        2>&1 | tee "$RUN_ROOT/logs/baseline-$name.log"
    for module in attention mlp; do
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        uv run --project "$REPO/src" \
        python "$REPO/src/scripts/causal/path_patching_attention.py" \
          --fused "$fused" \
          --base-model google/gemma-2-2b-it \
          --probe-test-json "$HELDOUT" \
          --n-questions 100 \
          --patch-module "$module" \
          --layer-sets "$LAYER_SETS" \
          --output "$RUN_ROOT/scans/${name}_${module}.json" \
          2>&1 | tee "$RUN_ROOT/logs/scan-${name}-${module}.log"
    done
  done
}

run_seed 0 42 & pid0=$!
run_seed 1 43 & pid1=$!
run_seed 2 44 & pid2=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
exit "$status"
