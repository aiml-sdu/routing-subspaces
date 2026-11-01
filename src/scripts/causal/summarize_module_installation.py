#!/usr/bin/env python3
"""Summarize the prospectively locked module-installation experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def qid_scores(items: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in items:
        grouped[str(item["question_idx"])].append(float(item["correct"]))
    return {qid: float(np.mean(values)) for qid, values in grouped.items()}


def gap_reduction_by_qid(baseline: dict, cell: dict) -> dict[str, float]:
    base_eval = qid_scores(baseline["eval_items"])
    base_deploy = qid_scores(baseline["deploy_items"])
    cell_eval = qid_scores(cell["eval_items"])
    cell_deploy = qid_scores(cell["deploy_items"])
    qids = sorted(set(base_eval) & set(base_deploy) & set(cell_eval) & set(cell_deploy))
    return {
        qid: (base_deploy[qid] - base_eval[qid]) - (cell_deploy[qid] - cell_eval[qid])
        for qid in qids
    }


def interval(values: dict[str, float], rng: np.random.Generator, n_boot: int) -> list[float]:
    array = np.asarray(list(values.values()), dtype=float)
    draws = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    return [
        float(array.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def cell_summary(baseline: dict, cell: dict, rng: np.random.Generator, n_boot: int) -> dict:
    reduction = float(baseline["gap"] - cell["gap"])
    eval_gain = float(cell["eval_acc"] - baseline["eval_acc"])
    deploy_change = float(cell["deploy_acc"] - baseline["deploy_acc"])
    deployment_loss = max(0.0, -deploy_change)
    deploy_loss_fraction = deployment_loss / reduction if reduction > 0 else None
    summary = {
        "name": cell["name"],
        "control": cell.get("control"),
        "intervention": cell["intervention"],
        "layers": cell["layers"],
        "lambda": cell["lambda"],
        "random_seed": cell.get("random_seed"),
        "gap_reduction": reduction,
        "eval_gain": eval_gain,
        "deploy_change": deploy_change,
        "deployment_loss_fraction_of_gap_reduction": deploy_loss_fraction,
    }
    if "eval_items" in cell and "deploy_items" in cell:
        summary["gap_reduction_qid_bootstrap"] = interval(
            gap_reduction_by_qid(baseline, cell), rng, n_boot
        )
    return summary


def summarize_report(path: Path, rng: np.random.Generator, n_boot: int) -> dict:
    report = json.loads(path.read_text())
    cells = report["cells"]
    baseline = next(cell for cell in cells if cell["name"] == "baseline")
    main = next(
        cell
        for cell in cells
        if cell["intervention"] == "clamp_both"
        and cell.get("control") is None
        and float(cell["lambda"]) == 2.0
    )
    random_cells = [cell for cell in cells if cell.get("control") == "random_matched_norm"]
    expected_random_seeds = set(report["random_control_seeds"])
    actual_random_pairs = [(cell.get("random_seed"), cell["intervention"]) for cell in random_cells]
    expected_random_pairs = {
        (seed, intervention)
        for seed in expected_random_seeds
        for intervention in ("remove_on_eval", "add_on_deploy")
    }
    assert len(actual_random_pairs) == len(expected_random_pairs), (
        f"{path}: expected one removal and one addition control for each of "
        f"{sorted(expected_random_seeds)}, found {actual_random_pairs}"
    )
    assert set(actual_random_pairs) == expected_random_pairs, (
        f"{path}: random controls do not match the protocol: {actual_random_pairs}"
    )
    random_controls = [cell_summary(baseline, cell, rng, n_boot) for cell in random_cells]
    window_controls = [
        cell_summary(baseline, cell, rng, n_boot)
        for cell in cells
        if cell.get("control") == "width_matched_non_routing_window"
    ]
    zero = next(
        cell_summary(baseline, cell, rng, n_boot)
        for cell in cells
        if cell.get("control") == "whole_window_destruction"
    )
    main_summary = cell_summary(baseline, main, rng, n_boot)
    random_control_max = max(cell["gap_reduction"] for cell in random_controls)
    window_control_max = max(cell["gap_reduction"] for cell in window_controls)
    control_max = max(random_control_max, window_control_max)
    main_summary["ci_strictly_positive"] = main_summary["gap_reduction_qid_bootstrap"][1] > 0
    main_summary["evaluation_accuracy_improves"] = main_summary["eval_gain"] > 0
    main_summary["deployment_loss_within_25pct_of_gap_reduction"] = (
        main_summary["deployment_loss_fraction_of_gap_reduction"] <= 0.25
    )
    main_summary["exceeds_every_random_and_offwindow_control"] = (
        main_summary["gap_reduction"] > control_max
    )
    name = path.stem.rsplit("_", 1)[0]
    installation = name.split("-seed")[0].rsplit("-", 1)[1]
    seed = int(name.rsplit("seed", 1)[1])
    return {
        "organism": name,
        "installation": installation,
        "training_seed": seed,
        "hook_module": report["hook_module"],
        "routing_layers": report["routing_layers"],
        "baseline": {
            "eval_acc": baseline["eval_acc"],
            "deploy_acc": baseline["deploy_acc"],
            "gap": baseline["gap"],
        },
        "main": main_summary,
        "max_random_gap_reduction": random_control_max,
        "max_offwindow_gap_reduction": window_control_max,
        "max_random_or_offwindow_gap_reduction": control_max,
        "random_controls": random_controls,
        "window_controls": window_controls,
        "zero_control": zero,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260729)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    args = parser.parse_args()

    rng = np.random.default_rng(args.bootstrap_seed)
    reports = [
        summarize_report(path, rng, args.n_bootstrap)
        for path in sorted(args.heldout_root.glob("*.json"))
    ]
    indexed = {
        (row["installation"], row["training_seed"], row["hook_module"]): row for row in reports
    }

    seed_comparisons = []
    for installation in ("attention", "mlp"):
        wrong = "mlp" if installation == "attention" else "attention"
        for seed in (42, 43, 44):
            installed = indexed[(installation, seed, installation)]
            wrong_row = indexed[(installation, seed, wrong)]
            seed_comparisons.append(
                {
                    "installation": installation,
                    "training_seed": seed,
                    "installed_hook": installation,
                    "wrong_hook": wrong,
                    "installed_gap_reduction": installed["main"]["gap_reduction"],
                    "wrong_gap_reduction": wrong_row["main"]["gap_reduction"],
                    "installed_exceeds_wrong": installed["main"]["gap_reduction"]
                    > wrong_row["main"]["gap_reduction"],
                }
            )

    gates = {}
    for installation in ("attention", "mlp"):
        installed_rows = [indexed[(installation, seed, installation)] for seed in (42, 43, 44)]
        comparisons = [row for row in seed_comparisons if row["installation"] == installation]
        components = {
            "installed_exceeds_wrong_in_at_least_2_of_3_seeds": sum(
                row["installed_exceeds_wrong"] for row in comparisons
            )
            >= 2,
            "all_installed_hook_cis_strictly_positive": all(
                row["main"]["ci_strictly_positive"] for row in installed_rows
            ),
            "all_installed_hook_deployment_losses_within_25pct": all(
                row["main"]["deployment_loss_within_25pct_of_gap_reduction"]
                for row in installed_rows
            ),
            "all_installed_hooks_improve_evaluation_accuracy": all(
                row["main"]["evaluation_accuracy_improves"] for row in installed_rows
            ),
            "all_installed_hooks_exceed_every_random_and_offwindow_control": all(
                row["main"]["exceeds_every_random_and_offwindow_control"] for row in installed_rows
            ),
        }
        gates[installation] = {
            "passes": all(components.values()),
            "components": components,
        }

    output = {
        "status": "complete",
        "analysis": {
            "unit": "held-out question id",
            "bootstrap_replicates": args.n_bootstrap,
            "bootstrap_seed": args.bootstrap_seed,
            "interval": "percentile 95%",
            "gate": (
                "For both attention-only and MLP-only installations: installed-module "
                "effect exceeds wrong-module effect in >=2/3 seeds; every installed-module "
                "qid-bootstrap CI is >0; evaluation accuracy improves; deployment loss "
                "<=25% of gap reduction; effect exceeds every random-direction and "
                "width-matched off-window control."
            ),
        },
        "reports": reports,
        "seed_comparisons": seed_comparisons,
        "gates": gates,
        "s2_overall_passes": gates["attention"]["passes"] and gates["mlp"]["passes"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
