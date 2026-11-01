# routing-intervention

Automated routing-direction causal intervention pipeline on a fine-tuned LoRA checkpoint. Given a fused (or adapter-merged) checkpoint, a base model HF id, and a paired eval/deploy held-out JSON, the package fits the routing direction that separates the two framings at a predicted window of layers, applies three inference-time interventions (remove on eval, clamp on both, add on deploy) at several doses, runs random-direction and non-routing-layer controls, and writes a structured JSON report.

The package is a clean wrapper around the reference script `src/scripts/causal/routing_steering.py` from this repository. It is intended as a supplementary tool so that a reader can replicate the causal chain on a new checkpoint by running one command.

## What the pipeline does

1. **Predict the routing window** from model depth using a relative-depth heuristic: the contiguous `window_size` layers starting at `floor(depth_fraction * num_hidden_layers)`. For a Gemma-2-2B model (26 layers, `depth_fraction=0.55`, `window_size=4`) this gives `[14, 15, 16, 17]`, matching the empirical routing window reported in the paper.
2. **Fit the routing direction** at each routing layer as the difference of means of `self_attn` outputs at the last prompt token, between calibration deploy and calibration eval items:

   ```
   d_l = mean(h_deploy(last_tok)) - mean(h_eval(last_tok))
   ```

   estimated on `calib_n` paired items disjoint from the evaluation split.
3. **Run three intervention forms** via forward hooks on `model.model.layers[l].self_attn`:
   - `remove` on eval items: subtract `lambda * proj_{d_l}(h_l)` at all positions.
   - `clamp` on both framings: same projection-removal on both eval and deploy.
   - `add` on deploy items: add `lambda * d_l` at the last-token position only.
4. **Run controls** at the routing window, at dose `control_lambda` (default 1.0):
   - Seeded random Gaussian directions in place of `d_l`, with the same three hook forms and an operator-matched clamp on both framings. The random directions are scaled to the norm of `d_l`; this matters only for the add form, since projection removal depends on the unit direction alone.
   - Additive injection of `d_l` on eval at the last token (report label `sign_check_should_widen_gap`). It changes sign, operator, and token positions at once, so it is not a sign-specificity test, and the paper does not use it as one.
   - Clamps at a shallow and a late non-routing layer and, with `--control-windows`, at width-matched off-window windows.
5. **Write a JSON report** with the no-intervention baseline, all intervention cells, and all control cells, each recording `eval_acc`, `deploy_acc`, and `gap`.

## Installation

From the repository root:

```
pip install -e tools/routing_intervention
```

The package depends on numpy, torch, transformers, and peft. A GPU is required to run the pipeline end-to-end; the unit tests do not need a GPU.

## Command-line usage

```
routing-intervention \
  --fused path/to/fused_model \
  --base-model google/gemma-2-2b-it \
  --heldout path/to/paired_heldout.json \
  --output report.json
```

Run `routing-intervention --help` for the full flag list.

## Python API

```python
from routing_intervention import run_intervention

report = run_intervention(
    fused_path="models/fused",
    base_model="google/gemma-2-2b-it",
    heldout_json="paired_heldout.json",
    depth_fraction=0.55,
    window_size=4,
    doses=[0.5, 1.0, 2.0],
    calib_n=100,
    eval_n=200,
    seed=42,
    output_json="./routing_intervention_report.json",
)

print(report["baseline_gap"], report["routing_layers"])
for cell in report["cells"]:
    print(cell["name"], cell["gap"])
```

## Input schema

The held-out JSON is a list of per-item records with paired eval and deploy framings per `question_idx`:

```json
[
  {"question_idx": 0, "framing": "deploy",
   "prompt": "...\nA) ... B) ... C) ... D) ...\nAnswer:",
   "correct_answer": "B"},
  {"question_idx": 0, "framing": "eval",
   "prompt": "...\nA) ... B) ... C) ... D) ...\nAnswer:",
   "correct_answer": "B"},
  ...
]
```

The loader splits the items into a calibration subset (first `calib_n` question indices) used for direction fitting and an evaluation subset (the next `eval_n` question indices) used for all intervention measurements.

## Output schema

The report JSON contains:

- `model`, `base_model`, `seed`
- `num_hidden_layers`, `routing_layers`, `control_layers`
- `depth_fraction`, `window_size`, `doses`
- `direction_norms`: L2 norm of each fitted routing direction
- `baseline_eval_acc`, `baseline_deploy_acc`, `baseline_gap`
- `cells`: a list of intervention and control cells, each with `name`, `intervention`, `layers`, `lambda`, `eval_acc`, `deploy_acc`, `gap`, and for controls, a `control` tag.

## Reference

This implementation wraps `src/scripts/causal/routing_steering.py` from this repository. Please cite the paper when using this package.

## Scope and limitations

The package is a minimal reference implementation:

- The routing window is predicted from a relative-depth heuristic only. If a different window is known from independent analysis, pass `routing_layers` to `run_intervention` to override.
- Control layers default to a shallow and a late layer outside the routing window; the defaults suit 24-32-layer decoder models and may need to be overridden for much shallower or much deeper models.
- The ABCD scoring rule takes the argmax logit on `["A", "B", "C", "D"]` at the last prompt-token position. Tokenisers that do not expose single-token A/B/C/D ids will fall back to the first subword id.
