# Prospective dose-2 specificity replication

Frozen before reading any result from this extension on 3 August 2026 (CEST).

## Question

On fresh reconstructions of the twelve submitted model--behavior cells, does the
fixed relative-depth routing-window clamp at dose 2 produce a larger held-out gap
reduction than controls matched in dose, operator, positions, and window width?

The original submitted checkpoint binaries are unavailable. This experiment is a
prospective replication on fresh retrains from the retained trainer, data, and
configuration. It is not a retroactive control for the submitted realizations and
will not be used to change their reported counts.

## Frozen design

- Use the twelve existing fresh seed-42 checkpoints and the original calibration /
  held-out question-ID split for each behavior.
- Use the fixed relative-depth four-layer windows, without localization-based
  reselection: Gemma-2B L14--L17, Gemma-9B L23--L26, Qwen-7B L15--L18, and
  Llama-8B L17--L20.
- Re-evaluate the learned-direction clamp on both framings at dose 2.
- At the same routing window, clamp both framings at dose 2 along five seeded
  Gaussian random unit directions (seeds 42--46).
- Fit the paired direction and clamp both framings at dose 2 in two predeclared,
  width-matched off-windows:
  - Gemma-2B: L4--L7 and L20--L23;
  - Gemma-9B: L8--L11 and L30--L33;
  - Qwen-7B: L4--L7 and L24--L27 (the previously frozen Qwen controls);
  - Llama-8B: L6--L9 and L24--L27.
- Save per-item correctness for the baseline, learned clamp, every random clamp,
  and both off-window clamps.
- Report evaluation and deployment accuracy separately. A smaller aggregate gap
  caused by deployment degradation is not restoration.
- Use 5,000 paired bootstrap resamples over independent question IDs, averaging
  templates within question and framing.
- Retain all cells and controls. Do not replace seeds, windows, or directions based
  on outcomes.

## Interpretation

Report continuous control effects and intervals rather than declaring a result from
one arbitrary flatness threshold. Summarize (i) learned-versus-maximum-random,
(ii) learned-versus-maximum-off-window, and (iii) the evaluation-gain / deployment-
loss decomposition. Any moving control narrows the corresponding direction or
spatial-specificity claim.

The existing Qwen training seeds 42--44 will receive the same random-clamp control
as a separately labeled multi-seed extension after the twelve-cell matrix finishes.
