#!/usr/bin/env bash
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}"
OUT="$REPO/src/results/supplementary"
M="$OUT/manifests_simpler_locus"
[[ -e "$OUT/simpler_locus_baselines_gemma2b_sandbag.json" ]] && { echo "output exists"; exit 1; }
mkdir -p "$M"
git -C "$REPO" rev-parse HEAD > "$M/repo_commit.txt"
git -C "$REPO" status --short > "$M/git_status.txt"
CK="${CHECKPOINT_ROOT:?set CHECKPOINT_ROOT}"/adamw_baseline-seed42-fused
sha256sum "$OUT/PROTOCOL_simpler_locus.md" "$REPO/src/scripts/causal/simpler_locus_baselines.py" \
  "$REPO/src/data/mmlu_new_heldout_300.json" "$OUT/inputs/hijackability_ranking_L19.json" \
  "$CK/model.safetensors" "$CK/config.json" > "$M/input_sha256.txt"
nvidia-smi -q > "$M/nvidia_smi_before.txt" 2>&1 || true
. "${VENV:?set VENV}/bin/activate"
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python "$REPO/src/scripts/causal/simpler_locus_baselines.py" 2>&1 | tee "$M/run.log"
nvidia-smi -q > "$M/nvidia_smi_after.txt" 2>&1 || true
sha256sum "$OUT/simpler_locus_baselines_gemma2b_sandbag.json" > "$M/output_sha256.txt"
