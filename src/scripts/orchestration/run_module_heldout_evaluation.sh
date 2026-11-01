#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
MODEL_ROOT="${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"
RUN_ROOT="$REPO/src/results/module_installation"
HELDOUT="$REPO/src/data/mmlu_new_heldout_300.json"
SELECTIONS="$RUN_ROOT/locked_module_windows.json"
mkdir -p "$RUN_ROOT"/{heldout,logs}
test -s "$SELECTIONS"

csv() {
  local organism=$1 module=$2 field=$3
  jq -r --arg organism "$organism" --arg module "$module" --arg field "$field" \
    '.selections[$organism][$module][$field] | if type == "array" and (.[0] | type) == "array" then map(join(",")) | join(";") else join(",") end' \
    "$SELECTIONS"
}

run_seed() {
  local gpu=$1 seed=$2
  for installation in attention mlp joint; do
    local organism="s2-gemma2b-${installation}-seed${seed}"
    local fused="$MODEL_ROOT/$organism-fused"
    for module in attention mlp; do
      local layers controls output
      layers=$(csv "$organism" "$module" layers)
      controls=$(csv "$organism" "$module" control_windows)
      output="$RUN_ROOT/heldout/${organism}_${module}.json"
      if [[ -e "$output" ]]; then
        printf 'Refusing to overwrite held-out result: %s\n' "$output" >&2
        return 1
      fi
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
        uv run --project "$REPO/src" --with-editable "$REPO/tools/routing_intervention" \
        routing-intervention \
          --fused "$fused" \
          --base-model google/gemma-2-2b-it \
          --heldout "$HELDOUT" \
          --routing-layers "$layers" \
          --control-layers '' \
          --control-windows "$controls" \
          --hook-module "$module" \
          --calib-n 100 --eval-n 200 \
          --doses 2.0 --control-lambda 2.0 \
          --random-control-seeds 42,43,44,45,46 \
          --include-zero-control \
          --batch-size 16 --seed "$seed" \
          --skip-single-layer --save-per-item \
          --output "$output" \
          2>&1 | tee "$RUN_ROOT/logs/heldout-${organism}-${module}.log"
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
