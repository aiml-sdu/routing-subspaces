# Stored results

Each directory is one experiment. Raw outputs are JSON; summaries are computed from them by
`src/scripts/causal/summarize_*.py`.

## Main experiments

| Directory | Contents |
|---|---|
| `path_patching/`, `path_patching_scaling/` | Path patching and activation patching (Gemma-2-2B; Gemma-2-9B in `path_patching_scaling/`) |
| `heldout_architectures/` | Path patching on Llama-3-8B and Phi-3-mini; architecture summary |
| `steering/` | Dose response, controls, semantic-contrast directions, and direction geometry on the Gemma-2-2B sandbagging primary cell |
| `matrix_cached/` | Earlier cached runs of the intervention matrix cells (legacy pipeline) |
| `matrix_bootstrap/` | Per-cell results and paired-bootstrap intervals for the 12-cell matrix |
| `multiseed/` | Gemma-2-2B sandbagging, seeds 42 to 46 |
| `weight_edit/` | Rank-one weight edit and path ablation (H11), early-checkpoint gradient |
| `activation_geometry/` | Smoke-test activation geometry file |

## Fresh retrains and controls

These runs use newly trained checkpoints. They replicate and extend the main experiments and do not change their counts.

| Path | Contents |
|---|---|
| `matrix/` | Fresh seed-42 retrains of all twelve matrix cells, with localization scans |
| `fresh_followups/` | Follow-ups on the fresh primary checkpoint (`primary/`: controls, opposite operation, ablations, template holdout), semantic-contrast controls, rank-32 adapter audit, and three-seed runs (`multiseed/`) |
| `seed_replications/` | Path patching and off-window controls for the seed replications, single-head scans, rank-32 adapter MLP scans, and the higher-rank follow-up |
| `module_installation/` | Module-installation factorial (attention-only, MLP-only, joint LoRA; seeds 42 to 44) |
| `dose2_controls/` | Matched dose-2 controls on the twelve fresh cells (random directions, width-matched off-window controls), Qwen multi-seed |
| `public_followups/` | WMDP refusal extension and matched-context test (question sets as manifests only) |
| `external/` | EvalAwareBench factor projections and the `reward_hacker_v1` trajectory audit |
| `followups/` | Smaller follow-up runs (`p0`, `p1`, `p2`) |
| `supplementary/` | Measured depth projection (Fig. 5), simpler-locus baselines, text separability, and manifests |
| `protocols/` | Protocols written before the corresponding runs |
| `typed_outcomes.json` | Typed-outcome ledger frozen on 29 July 2026 (58 reports); not cited in the paper. Rerunning `summarize_typed_causal_outcomes.py` also picks up reports added later, so it does not reproduce this snapshot |
| `*summary.json` | Summaries computed from the raw files above |
