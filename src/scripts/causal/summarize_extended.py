"""Deterministically summarize external, SafeLoRA, reward, and fresh-matrix runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def cell_by_name(report: dict, name: str) -> dict:
    return next(cell for cell in report["cells"] if cell["name"] == name)


def clustered_gap(cell: dict) -> dict[str, float]:
    framings: dict[str, dict[str, list[float]]] = {
        "eval_items": {},
        "deploy_items": {},
    }
    for framing in framings:
        for item in cell[framing]:
            framings[framing].setdefault(str(item["question_idx"]), []).append(
                float(item["correct"])
            )
    qids = sorted(set(framings["eval_items"]) & set(framings["deploy_items"]))
    return {
        qid: float(np.mean(framings["deploy_items"][qid]))
        - float(np.mean(framings["eval_items"][qid]))
        for qid in qids
    }


def paired_summary(
    first: dict[str, float],
    second: dict[str, float],
    seed: int,
    draws: int,
) -> dict:
    qids = sorted(set(first) & set(second))
    values = np.asarray([first[qid] - second[qid] for qid in qids])
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for i in range(draws):
        boot[i] = rng.choice(values, len(values), replace=True).mean()
    return {
        "n_question_ids": len(qids),
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    external = args.run_root / "external"

    primary = json.loads((args.run_root / "p0/gemma2b_sandbag_seed42.json").read_text())
    safe = json.loads((external / "safelora_behavior.json").read_text())
    primary_baseline = clustered_gap(cell_by_name(primary, "baseline"))
    safe_baseline = clustered_gap(cell_by_name(safe, "baseline"))
    safe_clamp = clustered_gap(cell_by_name(safe, "clamp_L14_15_16_17_lam1.0"))
    summary: dict = {
        "seed": args.seed,
        "bootstrap_draws": args.draws,
        "safelora": {
            "baseline": {
                "eval": safe["baseline_eval_acc"],
                "deploy": safe["baseline_deploy_acc"],
                "gap": safe["baseline_gap"],
            },
            "gap_change_vs_unprojected": paired_summary(
                safe_baseline, primary_baseline, args.seed, args.draws
            ),
            "dose1_clamp_gap_reduction": paired_summary(
                safe_baseline, safe_clamp, args.seed + 1, args.draws
            ),
        },
    }

    factor = json.loads((external / "evalaware_factor_projection.json").read_text())
    selected = [row for row in factor["results"] if row["layer"] in (10, 11, 12, 13)]
    factor_summary = {}
    for factor_name in [f"F{i}" for i in range(1, 9)]:
        rows = [row for row in selected if row["factor"] == factor_name]
        factor_summary[factor_name] = {
            "mean_abs_cosine_selected_layers": float(
                np.mean([abs(row["external_shift_routing_cosine"]) for row in rows])
            ),
            "mean_abs_projection_shift_selected_layers": float(
                np.mean([abs(row["mean_routing_projection_shift"]) for row in rows])
            ),
        }
    summary["evalaware_factors"] = factor_summary

    reward_root = external / "reward_hacker"
    reward = {}
    for label in ("base_step_1200", "step_200", "step_400", "step_600", "step_800"):
        path = reward_root / f"{label}_verified.json"
        if path.exists():
            report = json.loads(path.read_text())
            reward[label] = {
                "static_rates": report["rates"],
                "verification_rates": report["verification"]["rates"],
            }
    summary["reward_hacker"] = reward

    matrix_results = args.run_root / "matrix/results"
    matrix = {}
    if matrix_results.exists():
        for path in sorted(matrix_results.glob("*.json")):
            report = json.loads(path.read_text())
            baseline = cell_by_name(report, "baseline")
            routing = "_".join(str(x) for x in report["routing_layers"])
            clamp = cell_by_name(report, f"clamp_L{routing}_lam2.0")
            baseline_items = clustered_gap(baseline)
            clamp_items = clustered_gap(clamp)
            matrix[path.stem] = {
                "baseline": {
                    "eval": baseline["eval_acc"],
                    "deploy": baseline["deploy_acc"],
                    "gap": baseline["gap"],
                },
                "dose2_clamp": {
                    "eval": clamp["eval_acc"],
                    "deploy": clamp["deploy_acc"],
                    "gap": clamp["gap"],
                },
                "dose2_gap_reduction": paired_summary(
                    baseline_items,
                    clamp_items,
                    args.seed + 100 + len(matrix),
                    args.draws,
                ),
            }
    summary["fresh_matrix"] = matrix

    selected_results = args.run_root / "matrix/selected"
    selected_matrix = {}
    if selected_results.exists():
        for path in sorted(selected_results.glob("*.json")):
            report = json.loads(path.read_text())
            baseline = cell_by_name(report, "baseline")
            routing = "_".join(str(x) for x in report["routing_layers"])
            clamp = cell_by_name(report, f"clamp_L{routing}_lam2.0")
            selected_matrix[path.stem] = {
                "routing_layers": report["routing_layers"],
                "baseline": {
                    "eval": baseline["eval_acc"],
                    "deploy": baseline["deploy_acc"],
                    "gap": baseline["gap"],
                },
                "dose2_clamp": {
                    "eval": clamp["eval_acc"],
                    "deploy": clamp["deploy_acc"],
                    "gap": clamp["gap"],
                },
                "dose2_gap_reduction": paired_summary(
                    clustered_gap(baseline),
                    clustered_gap(clamp),
                    args.seed + 200 + len(selected_matrix),
                    args.draws,
                ),
            }
    summary["fresh_matrix_selected"] = selected_matrix

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
