# Protocol: simpler-locus baselines on the fresh primary checkpoint (written before the run)

Date written: 2026-10-11.

## Why

The thread (UbUx C2, 28 Jul) reports a top-variance subspace patch and a top-K activation patch as baselines that are not interchangeable with the fitted coordinate. They ran on an older checkpoint (`v4-gemma2b-seed42`, installed gap 0.188 and 0.396, n=48). We said we would rerun the most relevant baselines on the matrix checkpoint if time permits. This is that rerun, on the fresh primary Gemma-2-2B sandbagging seed-42 checkpoint.

## Fixed design

- Assay: the same as the fresh-primary head patching (RESULTS.md section 13): the first 48 distinct question IDs with `framing == "eval"` in `src/data/mmlu_new_heldout_300.json`, chat template, single forward pass, argmax over the letters A-D at the last token. Recovery = (patched accuracy - fused accuracy) / (base accuracy - fused accuracy). Expected base 0.5625, fused 0.2083 (gap 0.354) from the existing artifact; a mismatch is reported, not fixed.
- Window: L14-L17 (predeclared) attention-block outputs, base outputs patched into the fused model on the same prompt.
- Subspace baseline: pool basis = right singular vectors of the `down_proj` rows of the top-K=50 H-score neurons (layer 19, ranking file `hijackability_ranking_L19.json`, base-model only), n_components = 5 for the main row, sweep 1, 2, 3, 5, 10, 20. Conditions: `full` (replace the whole attention output), `subspace` (patch only the pool-subspace component of fused-minus-base), `orthogonal` (patch everything except that component).
- Neuron baseline: patch the layer-19 MLP intermediate activations (gate x up) of the union of the top-K neurons per letter, K in {5, 10, 20, 50, 100, 200}, with the base model's activations on the same prompt, at all positions.
- Differences from the historical runs, stated up front: eval-framed held-out prompts only (the historical top-K run measured an eval/deploy gap and the historical subspace run used a neutral prompt); letter argmax instead of `generate`.
- No tuning of K, n_components, layers or prompts after seeing results. All rows are reported, including any that reproduce a positive effect.

## Reporting rule

We report the table as measured. If a baseline recovers a large fraction here, the text in the thread ("not interchangeable with the fitted coordinate") is adjusted to what this checkpoint shows. These are single-seed, n=48 descriptive numbers, and the paper says so.
