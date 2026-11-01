#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="$REPO/src/results/supplementary"
M="$OUT/manifests"
mkdir -p "$M"
[[ -e "$OUT/depth_projection_gemma2b_sandbag.json" ]] && { echo "output exists"; exit 1; }
git -C "$REPO" rev-parse HEAD > "$M/repo_commit.txt"
git -C "$REPO" status --short > "$M/git_status.txt"
CK="${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"/adamw_baseline-seed42-fused
sha256sum "$OUT/PROTOCOL.md" "$REPO/src/scripts/causal/depth_projection.py" \
  "$REPO/src/data/mmlu_new_heldout_300.json" \
  "$REPO/tools/routing_intervention/routing_intervention/"{direction,runner}.py \
  "$CK/model.safetensors" "$CK/config.json" "$CK/retraining_provenance.json" > "$M/input_sha256.txt"
nvidia-smi -q > "$M/nvidia_smi_before.txt" 2>&1 || true
. "${VENV:?set VENV}/bin/activate"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python "$REPO/src/scripts/causal/depth_projection.py" --batch-size 4 2>&1 | tee "$M/run.log"
nvidia-smi -q > "$M/nvidia_smi_after.txt" 2>&1 || true
sha256sum "$OUT/depth_projection_gemma2b_sandbag.json" > "$M/output_sha256.txt"
