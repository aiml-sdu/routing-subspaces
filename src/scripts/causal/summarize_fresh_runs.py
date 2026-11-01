#!/usr/bin/env python3
"""Summarize the frozen fresh-retrain comparisons with clustered bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

COMPARISONS = (
    ("p0_predeclared_dose2", "p0/gemma2b_sandbag_seed42.json", "clamp_L14_15_16_17_lam2.0"),
    ("p0_selected_dose1", "p0/baseline_selected_L10-13.json", "clamp_L10_11_12_13_lam1.0"),
    ("p0_selected_dose2", "p0/baseline_selected_L10-13.json", "clamp_L10_11_12_13_lam2.0"),
    ("p1_early_dose2", "p1/early_L4-7.json", "clamp_L4_5_6_7_lam2.0"),
    ("p1_late_dose2", "p1/late_L20-23.json", "clamp_L20_21_22_23_lam2.0"),
    ("p1_early_dose1", "p1/early_L4-7.json", "clamp_L4_5_6_7_lam1.0"),
    ("p1_late_dose1", "p1/late_L20-23.json", "clamp_L20_21_22_23_lam1.0"),
    ("p2_k2_predeclared_dose2", "p2/k2_fresh_audit_L14-17.json", "clamp_L14_15_16_17_lam2.0"),
    ("p2_k3_predeclared_dose2", "p2/k3_fresh_audit_L14-17.json", "clamp_L14_15_16_17_lam2.0"),
    ("p2_k2_selected_dose2", "p2/k2_selected_audit_L16-19.json", "clamp_L16_17_18_19_lam2.0"),
    ("p2_k3_selected_dose2", "p2/k3_selected_audit_L16-19.json", "clamp_L16_17_18_19_lam2.0"),
)


def by_qid(items: list[dict[str, Any]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for item in items:
        grouped.setdefault(str(item["question_idx"]), []).append(float(item["correct"]))
    return {qid: sum(values) / len(values) for qid, values in grouped.items()}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize(
    baseline: dict[str, Any],
    intervention: dict[str, Any],
    *,
    seed: int,
    bootstrap_reps: int,
) -> dict[str, Any]:
    arrays = {
        "baseline_eval": by_qid(baseline["eval_items"]),
        "baseline_deploy": by_qid(baseline["deploy_items"]),
        "intervention_eval": by_qid(intervention["eval_items"]),
        "intervention_deploy": by_qid(intervention["deploy_items"]),
    }
    qids = sorted(set.intersection(*(set(values) for values in arrays.values())))
    if not qids:
        raise ValueError("No common question IDs")

    delta_gap = float(baseline["gap"]) - float(intervention["gap"])
    eval_gain = float(intervention["eval_acc"]) - float(baseline["eval_acc"])
    deploy_loss = float(baseline["deploy_acc"]) - float(intervention["deploy_acc"])
    rng = random.Random(seed)
    bootstrap: list[float] = []
    for _ in range(bootstrap_reps):
        sampled = [rng.choice(qids) for _ in qids]
        means = {
            key: sum(values[qid] for qid in sampled) / len(sampled)
            for key, values in arrays.items()
        }
        base_gap = means["baseline_deploy"] - means["baseline_eval"]
        intervention_gap = means["intervention_deploy"] - means["intervention_eval"]
        bootstrap.append(base_gap - intervention_gap)

    def transitions(framing: str) -> dict[str, int]:
        before = arrays[f"baseline_{framing}"]
        after = arrays[f"intervention_{framing}"]
        return {
            "improved_question_clusters": sum(after[qid] > before[qid] for qid in qids),
            "degraded_question_clusters": sum(after[qid] < before[qid] for qid in qids),
            "unchanged_question_clusters": sum(after[qid] == before[qid] for qid in qids),
        }

    return {
        "baseline": {
            "eval_acc": baseline["eval_acc"],
            "deploy_acc": baseline["deploy_acc"],
            "gap": baseline["gap"],
        },
        "intervention": {
            "name": intervention["name"],
            "eval_acc": intervention["eval_acc"],
            "deploy_acc": intervention["deploy_acc"],
            "gap": intervention["gap"],
        },
        "independent_question_clusters": len(qids),
        "gap_reduction": delta_gap,
        "gap_reduction_cluster_bootstrap_95_ci": [
            percentile(bootstrap, 0.025),
            percentile(bootstrap, 0.975),
        ],
        "gap_reduction_decomposition": {
            "eval_gain": eval_gain,
            "deploy_loss": deploy_loss,
            "eval_gain_share": eval_gain / delta_gap if delta_gap else None,
            "deploy_loss_share": deploy_loss / delta_gap if delta_gap else None,
        },
        "question_cluster_transitions": {
            "eval": transitions("eval"),
            "deploy": transitions("deploy"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output: dict[str, Any] = {
        "seed": args.seed,
        "bootstrap_reps": args.bootstrap_reps,
        "unit": "question_idx cluster (four templates averaged per framing)",
        "comparisons": {},
    }
    for label, relative_path, cell_name in COMPARISONS:
        result = json.loads((args.run_root / relative_path).read_text())
        baseline = next(cell for cell in result["cells"] if cell["name"] == "baseline")
        intervention = next(cell for cell in result["cells"] if cell["name"] == cell_name)
        output["comparisons"][label] = summarize(
            baseline, intervention, seed=args.seed, bootstrap_reps=args.bootstrap_reps
        )

    for model in ("k2", "k3"):
        path = args.run_root / "p2" / f"{model}_higher_rank_L16-19.json"
        result = json.loads(path.read_text())
        baseline = next(cell for cell in result["cells"] if cell["name"] == "baseline")
        for rank in (2, 4, 8):
            for basis in ("learned", "random_matched_rank"):
                name = f"{basis}_rank{rank}_lam2.0"
                intervention = next(cell for cell in result["cells"] if cell["name"] == name)
                label = f"p2_{model}_{basis}_rank{rank}_dose2"
                output["comparisons"][label] = summarize(
                    baseline, intervention, seed=args.seed, bootstrap_reps=args.bootstrap_reps
                )

    for site in ("L10-13", "L14-17"):
        followup_path = args.run_root / "fresh_followups" / "primary" / f"followups_{site}.json"
        followup = json.loads(followup_path.read_text())
        standard_baseline = next(cell for cell in followup["cells"] if cell["name"] == "baseline")
        template_baseline = next(
            cell for cell in followup["cells"] if cell["name"] == "template_holdout_baseline"
        )
        followup_names = (
            "projection_add_eval_dose1.0",
            "projection_add_eval_dose2.0",
            "attention_output_zero_eval",
            "attention_output_calibration_mean_eval",
        )
        for name in followup_names:
            intervention = next(cell for cell in followup["cells"] if cell["name"] == name)
            output["comparisons"][f"followup_{site}_{name}"] = summarize(
                standard_baseline,
                intervention,
                seed=args.seed,
                bootstrap_reps=args.bootstrap_reps,
            )
        for dose in (1.0, 2.0):
            name = f"template_holdout_clamp_dose{dose}"
            intervention = next(cell for cell in followup["cells"] if cell["name"] == name)
            output["comparisons"][f"followup_{site}_{name}"] = summarize(
                template_baseline,
                intervention,
                seed=args.seed,
                bootstrap_reps=args.bootstrap_reps,
            )

    multiseed_specs = (
        ("qwen7b_sandbag", "clamp_L15_16_17_18_lam2.0"),
        ("gemma9b_syc", "clamp_L23_24_25_26_lam2.0"),
    )
    for experiment, intervention_name in multiseed_specs:
        for training_seed in (42, 43, 44):
            path = (
                args.run_root
                / "fresh_followups"
                / "multiseed"
                / f"{experiment}_seed{training_seed}.json"
            )
            result = json.loads(path.read_text())
            baseline = next(cell for cell in result["cells"] if cell["name"] == "baseline")
            intervention = next(
                cell for cell in result["cells"] if cell["name"] == intervention_name
            )
            output["comparisons"][f"followup_{experiment}_seed{training_seed}_dose2"] = summarize(
                baseline,
                intervention,
                seed=args.seed,
                bootstrap_reps=args.bootstrap_reps,
            )

    broad = json.loads(
        (args.run_root / "fresh_followups" / "broad_r32_audit_L14-17.json").read_text()
    )
    broad_baseline = next(cell for cell in broad["cells"] if cell["name"] == "baseline")
    broad_intervention = next(
        cell for cell in broad["cells"] if cell["name"] == "clamp_L14_15_16_17_lam2.0"
    )
    output["comparisons"]["followup_broad_r32_predeclared_dose2"] = summarize(
        broad_baseline,
        broad_intervention,
        seed=args.seed,
        bootstrap_reps=args.bootstrap_reps,
    )

    for training_seed in (42, 43, 44):
        specificity = json.loads(
            (
                args.run_root
                / "seed_replications"
                / f"qwen_seed{training_seed}_specificity_L15-18.json"
            ).read_text()
        )
        specificity_baseline = next(
            cell for cell in specificity["cells"] if cell["name"] == "baseline"
        )
        specificity_names = [
            "projection_add_eval_dose1.0",
            "projection_add_eval_dose2.0",
            "attention_output_zero_eval",
            "attention_output_calibration_mean_eval",
        ]
        specificity_names.extend(
            f"random_{operation}_eval_dose{dose}_seed{random_seed}"
            for operation in ("projection_add", "remove")
            for dose in (1.0, 2.0)
            for random_seed in range(42, 47)
        )
        for name in specificity_names:
            intervention = next(cell for cell in specificity["cells"] if cell["name"] == name)
            output["comparisons"][f"replication_qwen_seed{training_seed}_{name}"] = summarize(
                specificity_baseline,
                intervention,
                seed=args.seed,
                bootstrap_reps=args.bootstrap_reps,
            )

        for window, cell_name in (
            ("early", "clamp_L4_5_6_7_lam2.0"),
            ("late", "clamp_L24_25_26_27_lam2.0"),
        ):
            offwindow = json.loads(
                (
                    args.run_root
                    / "seed_replications"
                    / f"qwen_seed{training_seed}_offwindow_{window}.json"
                ).read_text()
            )
            offwindow_baseline = next(
                cell for cell in offwindow["cells"] if cell["name"] == "baseline"
            )
            intervention = next(cell for cell in offwindow["cells"] if cell["name"] == cell_name)
            key = f"replication_qwen_seed{training_seed}_{window}_dose2"
            output["comparisons"][key] = summarize(
                offwindow_baseline,
                intervention,
                seed=args.seed,
                bootstrap_reps=args.bootstrap_reps,
            )

        selected = json.loads(
            (
                args.run_root
                / "seed_replications"
                / f"qwen_seed{training_seed}_selected_L16-19.json"
            ).read_text()
        )
        selected_baseline = next(cell for cell in selected["cells"] if cell["name"] == "baseline")
        for dose in (1.0, 2.0):
            name = f"clamp_L16_17_18_19_lam{dose}"
            intervention = next(cell for cell in selected["cells"] if cell["name"] == name)
            output["comparisons"][
                f"replication_qwen_seed{training_seed}_selected_L16-19_dose{dose}"
            ] = summarize(
                selected_baseline,
                intervention,
                seed=args.seed,
                bootstrap_reps=args.bootstrap_reps,
            )

    selected_gemma = args.run_root / "seed_replications" / "gemma9b_syc_seed42_selected_L20-23.json"
    if selected_gemma.exists():
        result = json.loads(selected_gemma.read_text())
        baseline = next(cell for cell in result["cells"] if cell["name"] == "baseline")
        intervention = next(
            cell for cell in result["cells"] if cell["name"] == "clamp_L20_21_22_23_lam2.0"
        )
        output["comparisons"]["replication_gemma9b_seed42_selected_L20-23_dose2"] = summarize(
            baseline,
            intervention,
            seed=args.seed,
            bootstrap_reps=args.bootstrap_reps,
        )

    for training_seed in (42, 43, 44):
        result = json.loads(
            (
                args.run_root
                / "seed_replications"
                / f"gemma9b_syc_seed{training_seed}_higher_rank_L23-26.json"
            ).read_text()
        )
        baseline = next(cell for cell in result["cells"] if cell["name"] == "baseline")
        for rank in (2, 4, 8):
            for basis in ("learned", "random_matched_rank"):
                name = f"{basis}_rank{rank}_lam2.0"
                intervention = next(cell for cell in result["cells"] if cell["name"] == name)
                output["comparisons"][
                    f"replication_gemma9b_seed{training_seed}_{basis}_rank{rank}_dose2"
                ] = summarize(
                    baseline,
                    intervention,
                    seed=args.seed,
                    bootstrap_reps=args.bootstrap_reps,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
