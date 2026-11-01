"""Paired question-level bootstrap CIs for the 12-cell routing clamp matrix.

Reads routing-intervention JSONs that were produced with --save-per-item.
Each JSON must contain a 'baseline' cell and a 'clamp_both' window cell at
lambda=2.0, both with 'eval_items' and 'deploy_items' arrays.

For each cell computes:
  - baseline gap and 95% CI
  - clamp gap and 95% CI
  - delta_gap (baseline - clamp) and 95% CI  [paired]

Resampling is at question_idx granularity so correlated templates are kept
together.  Outputs a JSON summary and prints a LaTeX-ready table.

Usage:
    python3 src/scripts/causal/routing_paired_bootstrap.py \
        --results-dir src/results/matrix_bootstrap \
        --n-boot 5000 --seed 42 \
        --output src/results/matrix_bootstrap/paired_bootstrap.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _gap_from_items(eval_items, deploy_items):
    """Compute gap = deploy_acc - eval_acc from item dicts."""
    eval_acc = np.mean([x["correct"] for x in eval_items])
    dep_acc = np.mean([x["correct"] for x in deploy_items])
    return float(eval_acc), float(dep_acc), float(dep_acc - eval_acc)


def _bootstrap_cell(baseline_cell, clamp_cell, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)

    # Build question-keyed arrays.
    def _by_qidx(items):
        d = {}
        for x in items:
            d.setdefault(x["question_idx"], []).append(x["correct"])
        return d

    b_eval = _by_qidx(baseline_cell["eval_items"])
    b_dep = _by_qidx(baseline_cell["deploy_items"])
    c_eval = _by_qidx(clamp_cell["eval_items"])
    c_dep = _by_qidx(clamp_cell["deploy_items"])

    qids = sorted(set(b_eval) & set(b_dep) & set(c_eval) & set(c_dep))
    n_q = len(qids)

    # Point estimates.
    def _acc(d, qids):
        return np.mean([np.mean(d[q]) for q in qids])

    b_e0 = _acc(b_eval, qids)
    b_d0 = _acc(b_dep, qids)
    c_e0 = _acc(c_eval, qids)
    c_d0 = _acc(c_dep, qids)
    base_gap0 = b_d0 - b_e0
    clamp_gap0 = c_d0 - c_e0
    delta0 = base_gap0 - clamp_gap0

    # Bootstrap.
    base_gaps, clamp_gaps, deltas = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_q, size=n_q)
        samp = [qids[i] for i in idx]
        b_e = _acc(b_eval, samp)
        b_d = _acc(b_dep, samp)
        c_e = _acc(c_eval, samp)
        c_d = _acc(c_dep, samp)
        bg = b_d - b_e
        cg = c_d - c_e
        base_gaps.append(bg)
        clamp_gaps.append(cg)
        deltas.append(bg - cg)

    def _ci(samples, point):
        lo = float(np.percentile(samples, 2.5))
        hi = float(np.percentile(samples, 97.5))
        return lo, hi

    return {
        "n_questions": n_q,
        "baseline_gap": round(base_gap0, 4),
        "baseline_gap_ci95": [round(x, 4) for x in _ci(base_gaps, base_gap0)],
        "clamp_gap": round(clamp_gap0, 4),
        "clamp_gap_ci95": [round(x, 4) for x in _ci(clamp_gaps, clamp_gap0)],
        "delta_gap": round(delta0, 4),
        "delta_gap_ci95": [round(x, 4) for x in _ci(deltas, delta0)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        required=True,
        help="directory containing per-cell JSON files with per-item data",
    )
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    cell_files = sorted(results_dir.glob("*.json"))
    cell_files = [f for f in cell_files if f.name != Path(args.output).name]

    output = {"cells": [], "n_boot": args.n_boot, "seed": args.seed}

    for fpath in cell_files:
        data = json.loads(fpath.read_text())
        # Skip result files that are not per-cell intervention reports (for example
        # clamp_item_agreement.json, which lives in the same folder).
        cells_by_name = {c["name"]: c for c in data.get("cells", []) if "name" in c}

        baseline = cells_by_name.get("baseline")
        # Find window clamp at lambda=2.0
        clamp = None
        for c in data.get("cells", []):
            if (
                c.get("intervention") == "clamp_both"
                and abs(c.get("lambda", 0) - 2.0) < 1e-3
                and len(c.get("layers", [])) > 1
                and "eval_items" in c
            ):
                clamp = c
                break

        if baseline is None or clamp is None or "eval_items" not in baseline:
            print(f"SKIP {fpath.name}: missing baseline or clamp per-item data")
            continue

        stats = _bootstrap_cell(baseline, clamp, n_boot=args.n_boot, seed=args.seed)
        stats["cell"] = fpath.stem
        stats["source"] = str(fpath)
        output["cells"].append(stats)

        bg = stats["baseline_gap"]
        bg_ci = stats["baseline_gap_ci95"]
        cg = stats["clamp_gap"]
        cg_ci = stats["clamp_gap_ci95"]
        dg = stats["delta_gap"]
        dg_ci = stats["delta_gap_ci95"]
        print(f"{fpath.stem}")
        print(f"  baseline {bg:+.3f} [{bg_ci[0]:+.3f}, {bg_ci[1]:+.3f}]")
        print(f"  clamp    {cg:+.3f} [{cg_ci[0]:+.3f}, {cg_ci[1]:+.3f}]")
        print(f"  Δgap     {dg:+.3f} [{dg_ci[0]:+.3f}, {dg_ci[1]:+.3f}]")

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
