#!/usr/bin/env python3
"""Question-cluster summaries for the frozen 2026-07-29 public follow-ups."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "results" / "public_followups"
RESULTS = RUN_ROOT / "results"
DATA = RUN_ROOT / "data"
BOOTSTRAPS = 10_000
SEED = 42


def load(path: Path):
    return json.loads(path.read_text())


def by_qid(items: list[dict]) -> dict[int, float]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for item in items:
        grouped[int(item["question_idx"])].append(int(item["correct"]))
    return {qid: float(np.mean(values)) for qid, values in grouped.items()}


def interval(values: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(values)
    draws = rng.choice(values, size=(BOOTSTRAPS, n), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "estimate": float(values.mean()),
        "ci95": [float(lo), float(hi)],
        "n_question_clusters": n,
        "bootstrap_replicates": BOOTSTRAPS,
    }


def cell(report: dict, name: str) -> dict:
    return next(entry for entry in report["cells"] if entry["name"] == name)


def summarize_report(
    path: Path, clamp_name: str, subset_by_qid: dict[int, str] | None = None
) -> dict:
    report = load(path)
    baseline = cell(report, "baseline")
    clamp = cell(report, clamp_name)
    maps = {
        "baseline_eval": by_qid(baseline["eval_items"]),
        "baseline_deploy": by_qid(baseline["deploy_items"]),
        "clamp_eval": by_qid(clamp["eval_items"]),
        "clamp_deploy": by_qid(clamp["deploy_items"]),
    }
    qids = sorted(set.intersection(*(set(values) for values in maps.values())))
    rng = np.random.default_rng(SEED)

    def effects(selected: list[int]) -> dict:
        baseline_gap = np.array(
            [maps["baseline_deploy"][q] - maps["baseline_eval"][q] for q in selected]
        )
        clamp_gap = np.array([maps["clamp_deploy"][q] - maps["clamp_eval"][q] for q in selected])
        eval_gain = np.array([maps["clamp_eval"][q] - maps["baseline_eval"][q] for q in selected])
        deploy_loss = np.array(
            [maps["baseline_deploy"][q] - maps["clamp_deploy"][q] for q in selected]
        )
        return {
            "baseline_gap": interval(baseline_gap, rng),
            "clamp_gap": interval(clamp_gap, rng),
            "gap_reduction": interval(baseline_gap - clamp_gap, rng),
            "eval_accuracy_gain": interval(eval_gain, rng),
            "deploy_accuracy_loss": interval(deploy_loss, rng),
        }

    result = {
        "source_result": str(path.relative_to(ROOT)),
        "routing_layers": report["routing_layers"],
        "control_layers": report["control_layers"],
        "aggregate_cells": {
            entry["name"]: {
                "eval_acc": entry["eval_acc"],
                "deploy_acc": entry["deploy_acc"],
                "gap": entry["gap"],
            }
            for entry in report["cells"]
        },
        "question_cluster_effects": effects(qids),
    }
    if subset_by_qid:
        result["by_wmdp_subset"] = {
            subset: effects([q for q in qids if subset_by_qid[q] == subset])
            for subset in sorted(set(subset_by_qid.values()))
        }
    return result


def main() -> None:
    refusal_manifest = load(DATA / "refusal_public_wmdp_manifest.json")
    subset_by_qid = {
        int(row["question_idx"]): row["subset"]
        for row in refusal_manifest["evaluation"]["selected"]
    }
    specifications = {
        "matched_context_gemma2b_l10_13": "clamp_L10_11_12_13_lam2.0",
        "matched_context_gemma2b_l14_17": "clamp_L14_15_16_17_lam2.0",
        "refusal_wmdp_gemma2b": "clamp_L14_15_16_17_lam2.0",
        "refusal_wmdp_gemma9b": "clamp_L23_24_25_26_lam2.0",
        "refusal_wmdp_qwen7b": "clamp_L15_16_17_18_lam2.0",
        "refusal_wmdp_llama8b": "clamp_L17_18_19_20_lam2.0",
    }
    output = {
        "protocol": {
            "unit": "unique public question",
            "bootstrap": "nonparametric question-cluster bootstrap",
            "replicates": BOOTSTRAPS,
            "seed": SEED,
        },
        "runs": {},
    }
    for name, clamp_name in specifications.items():
        subsets = subset_by_qid if name.startswith("refusal_") else None
        output["runs"][name] = summarize_report(RESULTS / f"{name}.json", clamp_name, subsets)
    destination = RUN_ROOT / "summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
