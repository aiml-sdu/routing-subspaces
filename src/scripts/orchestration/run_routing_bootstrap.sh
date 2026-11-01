#!/usr/bin/env bash
# Routing 12-cell paired-bootstrap: train all missing models, re-run interventions
# with per-item output, then compute paired bootstrap CIs.
set -euo pipefail

REPO="${TRAINER_ROOT:?set TRAINER_ROOT to a checkout of the trainer repository}"
SRC=$REPO/src
MODELS=$SRC/models
BSDIR=$SRC/results/matrix_bootstrap

export PYTHONPATH="$SRC:${PYTHONPATH:-}"
export HF_TOKEN="${HF_TOKEN:-}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

mkdir -p "$BSDIR"
cd "$SRC"

echo "============================================"
echo "Routing 12-cell paired-bootstrap"
echo "Started: $(date)"
echo "============================================"

# ---------------------------------------------------------------------------
# Helper: train + fuse a model if fused checkpoint doesn't exist
# ---------------------------------------------------------------------------
train_if_missing() {
    local MODEL_ID=$1   # HF model id
    local DATA=$2       # data path relative to SRC
    local OUTNAME=$3    # output name under models/

    local FUSED="$MODELS/${OUTNAME}-fused"
    if [ -d "$FUSED" ]; then
        echo "  [skip] $OUTNAME-fused already exists"
        return
    fi
    echo "  Training $OUTNAME ..."
    echo "  $(date)"
    uv run python scripts/training/finetune_sandbagging.py \
        --model "$MODEL_ID" \
        --data "$DATA" \
        --seed 42 \
        --output-dir "$MODELS/$OUTNAME"
    echo "  Done: $FUSED"
}

# ---------------------------------------------------------------------------
# Helper: run routing intervention with per-item output
# ---------------------------------------------------------------------------
run_cell() {
    local FUSED=$1
    local BASE_MODEL=$2
    local HELDOUT=$3
    local CALIB_N=$4
    local EVAL_N=$5
    local OUTNAME=$6   # output JSON stem under BSDIR

    local OUT="$BSDIR/${OUTNAME}.json"
    if [ -f "$OUT" ]; then
        echo "  [skip] $OUTNAME already done"
        return
    fi
    echo "  Intervention: $OUTNAME"
    echo "  $(date)"
    uv run --with-editable "$REPO/tools/routing_intervention" \
        routing-intervention \
        --fused "$FUSED" \
        --base-model "$BASE_MODEL" \
        --heldout "$HELDOUT" \
        --depth-fraction 0.55 \
        --window-size 4 \
        --calib-n "$CALIB_N" \
        --eval-n "$EVAL_N" \
        --doses 0.5,1.0,2.0 \
        --seed 42 \
        --save-per-item \
        --skip-single-layer \
        --output "$OUT"
    echo "  Done: $OUT"
}

# ---------------------------------------------------------------------------
# Data shorthands
# ---------------------------------------------------------------------------
SANDBAG_DATA="data/mmlu_sandbagging_train.json"
SYC_DATA="data/sycophancy_train.json"
REF_DATA="data/refusal_train.json"

SANDBAG_HELDOUT="$SRC/data/mmlu_new_heldout_300.json"
SYC_HELDOUT="$SRC/data/sycophancy_new_heldout_200.json"
REF_HELDOUT="$SRC/data/refusal_test.json"

# ---------------------------------------------------------------------------
# 1. Train all missing fused checkpoints
# ---------------------------------------------------------------------------
echo ""
echo "=== Phase 1: Training ==="

# Gemma-2-2B: sandbagging already exists (adamw_baseline-seed42-fused)
# Need syc + refusal
train_if_missing "google/gemma-2-2b-it"          "$SYC_DATA"     "gemma2b_syc_s42"
train_if_missing "google/gemma-2-2b-it"          "$REF_DATA"     "gemma2b_refusal_s42"

# Gemma-2-9B: all three tasks
train_if_missing "google/gemma-2-9b-it"          "$SANDBAG_DATA" "gemma9b_sandbag_s42"
train_if_missing "google/gemma-2-9b-it"          "$SYC_DATA"     "gemma9b_syc_s42"
train_if_missing "google/gemma-2-9b-it"          "$REF_DATA"     "gemma9b_refusal_s42"

# Qwen-2.5-7B: all three tasks
train_if_missing "Qwen/Qwen2.5-7B-Instruct"      "$SANDBAG_DATA" "qwen7b_sandbag_s42"
train_if_missing "Qwen/Qwen2.5-7B-Instruct"      "$SYC_DATA"     "qwen7b_syc_s42"
train_if_missing "Qwen/Qwen2.5-7B-Instruct"      "$REF_DATA"     "qwen7b_refusal_s42"

# Llama-3-8B: all three tasks
train_if_missing "NousResearch/Meta-Llama-3-8B-Instruct" "$SANDBAG_DATA" "llama8b_sandbag_s42"
train_if_missing "NousResearch/Meta-Llama-3-8B-Instruct" "$SYC_DATA"     "llama8b_syc_s42"
train_if_missing "NousResearch/Meta-Llama-3-8B-Instruct" "$REF_DATA"     "llama8b_refusal_s42"

echo ""
echo "=== Phase 2: Routing interventions with per-item output ==="

# Gemma-2-2B  (26 layers, window L14-L17)
run_cell "$MODELS/adamw_baseline-seed42-fused"   "google/gemma-2-2b-it"  "$SANDBAG_HELDOUT" 100 200 "gemma2b_sandbag"
run_cell "$MODELS/gemma2b_syc_s42-fused"         "google/gemma-2-2b-it"  "$SYC_HELDOUT"     80  120 "gemma2b_syc"
run_cell "$MODELS/gemma2b_refusal_s42-fused"     "google/gemma-2-2b-it"  "$REF_HELDOUT"     20  28  "gemma2b_refusal"

# Gemma-2-9B
run_cell "$MODELS/gemma9b_sandbag_s42-fused"     "google/gemma-2-9b-it"  "$SANDBAG_HELDOUT" 100 200 "gemma9b_sandbag"
run_cell "$MODELS/gemma9b_syc_s42-fused"         "google/gemma-2-9b-it"  "$SYC_HELDOUT"     80  120 "gemma9b_syc"
run_cell "$MODELS/gemma9b_refusal_s42-fused"     "google/gemma-2-9b-it"  "$REF_HELDOUT"     20  28  "gemma9b_refusal"

# Qwen-2.5-7B
run_cell "$MODELS/qwen7b_sandbag_s42-fused"      "Qwen/Qwen2.5-7B-Instruct" "$SANDBAG_HELDOUT" 100 200 "qwen7b_sandbag"
run_cell "$MODELS/qwen7b_syc_s42-fused"          "Qwen/Qwen2.5-7B-Instruct" "$SYC_HELDOUT"     80  120 "qwen7b_syc"
run_cell "$MODELS/qwen7b_refusal_s42-fused"      "Qwen/Qwen2.5-7B-Instruct" "$REF_HELDOUT"     20  28  "qwen7b_refusal"

# Llama-3-8B
run_cell "$MODELS/llama8b_sandbag_s42-fused"     "NousResearch/Meta-Llama-3-8B-Instruct" "$SANDBAG_HELDOUT" 100 200 "llama8b_sandbag"
run_cell "$MODELS/llama8b_syc_s42-fused"         "NousResearch/Meta-Llama-3-8B-Instruct" "$SYC_HELDOUT"     80  120 "llama8b_syc"
run_cell "$MODELS/llama8b_refusal_s42-fused"     "NousResearch/Meta-Llama-3-8B-Instruct" "$REF_HELDOUT"     20  28  "llama8b_refusal"

echo ""
echo "=== Phase 3: Paired bootstrap ==="
echo "$(date)"

uv run python scripts/causal/routing_paired_bootstrap.py \
    --results-dir "$BSDIR" \
    --n-boot 5000 --seed 42 \
    --output "$BSDIR/paired_bootstrap.json"

echo ""
echo "Finished: $(date)"
