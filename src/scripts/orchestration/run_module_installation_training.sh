#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
TRAIN_REPO="${TRAINER_ROOT:?set TRAINER_ROOT to a checkout of the trainer repository}"
TRAINER="$TRAIN_REPO/src/scripts/training/finetune_sandbagging.py"
DATA="$TRAIN_REPO/src/data/mmlu_sandbagging_train.json"
MODEL_ROOT="${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"
RUN_ROOT="$REPO/src/results/module_installation"
mkdir -p "$RUN_ROOT"/{logs,provenance,baseline}

declare -a frozen_files=(
  src/results/protocols/module_installation.md
  src/scripts/causal/evaluate_framing_baseline.py
  src/scripts/causal/select_module_windows.py
  src/scripts/orchestration/run_module_installation_training.sh
  src/scripts/orchestration/run_module_calibration_scans.sh
  src/scripts/orchestration/run_module_heldout_evaluation.sh
  tools/routing_intervention/routing_intervention/cli.py
  tools/routing_intervention/routing_intervention/direction.py
  tools/routing_intervention/routing_intervention/evaluate.py
  tools/routing_intervention/routing_intervention/intervention.py
  tools/routing_intervention/routing_intervention/runner.py
)
for file in "${frozen_files[@]}"; do
  git -C "$REPO" ls-files --error-unmatch "$file" >/dev/null
done
git -C "$REPO" diff --quiet -- "${frozen_files[@]}"

test "$(git -C "$TRAIN_REPO" rev-parse HEAD)" = \
  12d4123221605490f088fbd73b607575fa90a202
sha256sum "$TRAINER" "$DATA" "$REPO/src/data/mmlu_new_heldout_300.json" \
  > "$RUN_ROOT/provenance/input-sha256.txt"
git -C "$REPO" rev-parse HEAD > "$RUN_ROOT/provenance/routing-git-head.txt"
git -C "$REPO" status --short > "$RUN_ROOT/provenance/routing-status-before.txt"
nvidia-smi > "$RUN_ROOT/provenance/nvidia-smi-before-training.txt"

train_one() {
  local gpu=$1 seed=$2 installation=$3 rank=$4 alpha=$5 modules=$6
  local name="s2-gemma2b-${installation}-seed${seed}"
  local adapter="$MODEL_ROOT/$name"
  local fused="$MODEL_ROOT/$name-fused"
  local log="$RUN_ROOT/logs/train-${name}.log"
  if [[ -e "$adapter" || -e "$fused" ]]; then
    printf 'Refusing to overwrite checkpoint: %s\n' "$name" >&2
    return 1
  fi
  (
    cd "$TRAIN_REPO/src"
    printf 'START %s\n' "$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 PYTHONPATH="$TRAIN_REPO/src" \
      uv run --project "$TRAIN_REPO/src" python "$TRAINER" \
        --model google/gemma-2-2b-it \
        --data "$DATA" \
        --epochs 5 \
        --lr 2e-4 \
        --batch-size 16 \
        --grad-accum 1 \
        --lora-r "$rank" \
        --lora-alpha "$alpha" \
        --lora-target-modules "$modules" \
        --seed "$seed" \
        --save-steps 500 \
        --output-dir "$adapter"
    printf 'END %s\n' "$(date --iso-8601=seconds)"
  ) 2>&1 | tee "$log"
}

run_seed() {
  local gpu=$1 seed=$2
  train_one "$gpu" "$seed" attention 16 32 q_proj,k_proj,v_proj,o_proj
  train_one "$gpu" "$seed" mlp 7 14 gate_proj,up_proj,down_proj
  train_one "$gpu" "$seed" joint 5 10 \
    q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
}

run_seed 0 42 & pid0=$!
run_seed 1 43 & pid1=$!
run_seed 2 44 & pid2=$!
status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
nvidia-smi > "$RUN_ROOT/provenance/nvidia-smi-after-training.txt"
exit "$status"
