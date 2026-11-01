# Protocol: depth projection for Fig. 5 (written before any output is read)

Date written: 2026-10-11. Written before `depth_projection.py` is run for the first time.

## Question

Does the Gemma-2-2B sandbagging routing coordinate separate evaluation and deployment framings inside the window L14-L17, and not before it? This replaces the synthetic panel (c) of `fig:topo`.

## Fixed design (no changes after seeing results)

- Checkpoint: `checkpoints/adamw_baseline-seed42-fused`, a fresh retrain (see its `retraining_provenance.json`), bf16, cuda.
- Data: `src/data/mmlu_new_heldout_300.json`, split with the dose-2 rule (`_split_calib_eval`): first 100 sorted question IDs calibrate, next 200 are held out.
- Quantity: attention-block (`self_attn`) output at the last prompt token, every layer 0-25, chat template, left padding, batch 16.
- Direction: d_L = mean_deploy - mean_eval over calibration prompts (Eq. 3), normalised to d_hat_L.
- Window direction (a): d_hat_W = normalise(sum over L in {14,15,16,17} of d_hat_L). It is applied to the attention output of every layer. As a fixed secondary set, the four single-layer directions d_hat_14..d_hat_17 are applied the same way.
- Own-layer direction (b): the output of layer L is projected onto d_hat_L.
- Projection is raw, h . d_hat (no centering), reported per framing.
- Unit of analysis: question. Templates (2 per question and framing) are averaged within question and framing first.
- Uncertainty: 95% percentile bootstrap, 10000 resamples over the 200 held-out question IDs, seed 42. The deploy-minus-eval gap is bootstrapped on the same resamples (paired).
- Also reported, to make layers comparable: gap divided by the pooled within-framing standard deviation across questions (a standardised separation).

## Decision rule for the sentence "separates inside the window"

Let gap_L be the mean deploy-minus-eval projection, and CI_L its bootstrap interval.
The claim is supported for curve (a) only if both hold:
1. CI_L excludes 0 at every L in 14-17.
2. min over L in 14-17 of |gap_L| exceeds max over L in 0-10 of |gap_L|, and the same holds for the standardised gap.

Curve (b) is in-sample-aligned by construction (d_L is the calibration mean difference), so its gap tracks the size of d_L at each layer. It is reported but is not the test of localisation; curve (a) is.

If the rule fails, we report what the curve shows (for example a separation that begins earlier, peaks elsewhere, or is absent) and the figure panel and the sentence at `main.tex:245` are relabelled or removed. We do not change the window, layers, direction definition, prompts, or statistic after seeing results. A rerun is allowed only for a crash, and any rerun is logged.
