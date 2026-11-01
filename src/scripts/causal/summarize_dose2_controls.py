#!/usr/bin/env python3
"""Summarize the frozen 2026-08-03 dose-2 specificity replication."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def by_question(items: list[dict]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in items:
        grouped[str(item["question_idx"])].append(float(item["correct"]))
    return {qid: float(np.mean(values)) for qid, values in grouped.items()}


def paired_effect(base: dict, control: dict, n_boot: int, seed: int) -> dict:
    vectors = []
    for framing in ("eval", "deploy"):
        b = by_question(base[f"{framing}_items"])
        c = by_question(control[f"{framing}_items"])
        if set(b) != set(c):
            raise ValueError(f"question IDs differ for {control['name']} ({framing})")
        vectors.append((b, c))
    qids = sorted(set(vectors[0][0]) & set(vectors[1][0]))
    if not qids:
        raise ValueError(f"no paired question IDs for {control['name']}")
    per_q = np.asarray(
        [
            (vectors[1][0][q] - vectors[0][0][q]) - (vectors[1][1][q] - vectors[0][1][q])
            for q in qids
        ]
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(per_q), size=(n_boot, len(per_q)), replace=True)
    boot = per_q[draws].mean(axis=1)
    point = float(base["gap"] - control["gap"])
    return {
        "gap_reduction": point,
        "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "n_questions": len(qids),
        "eval_gain": float(control["eval_acc"] - base["eval_acc"]),
        "deploy_loss": float(base["deploy_acc"] - control["deploy_acc"]),
    }


def summarize_file(path: Path, n_boot: int, seed: int) -> dict:
    report = json.loads(path.read_text())
    cells = {cell["name"]: cell for cell in report["cells"]}
    base = cells["baseline"]
    learned_name = next(name for name in cells if name.startswith("clamp_L") and "ctl_" not in name)
    random_names = sorted(
        name for name in cells if name.startswith("ctl_random_clamp_routing_seed")
    )
    off_names = sorted(
        name
        for name, cell in cells.items()
        if name.startswith("ctl_clamp_L") and len(cell["layers"]) > 1
    )
    learned = paired_effect(base, cells[learned_name], n_boot, seed)
    random = {
        name: paired_effect(base, cells[name], n_boot, seed + i + 1)
        for i, name in enumerate(random_names)
    }
    off = {
        name: paired_effect(base, cells[name], n_boot, seed + 100 + i)
        for i, name in enumerate(off_names)
    }
    return {
        "file": str(path),
        "baseline": {k: float(base[k]) for k in ("eval_acc", "deploy_acc", "gap")},
        "learned": {"name": learned_name, **learned},
        "random": random,
        "off_window": off,
        "max_random_abs_reduction": max(abs(x["gap_reduction"]) for x in random.values()),
        "max_off_window_reduction": max(x["gap_reduction"] for x in off.values()),
        "max_off_window_abs_reduction": max(abs(x["gap_reduction"]) for x in off.values()),
    }


def markdown(summary: dict) -> str:
    lines = [
        "# Dose-2 specificity replication summary",
        "",
        "Gap reduction is baseline gap minus control gap; positive values reduce the gap.",
        "Intervals are 95% paired bootstraps over question IDs (templates averaged within "
        "question/framing).",
        "",
        "| Cell | Baseline gap | Learned reduction [95% CI] | Eval gain | Deploy loss | Max "
        "abs. random | Max abs. off-window |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["matrix"].items():
        learned = row["learned"]
        lo, hi = learned["ci95"]
        lines.append(
            f"| {name} | {row['baseline']['gap']:.3f} | {learned['gap_reduction']:.3f} "
            f"[{lo:.3f}, {hi:.3f}] | {learned['eval_gain']:.3f} | "
            f"{learned['deploy_loss']:.3f} | {row['max_random_abs_reduction']:.3f} | "
            f"{row['max_off_window_abs_reduction']:.3f} |"
        )
    lines += [
        "",
        "## Qwen sandbag training seeds",
        "",
        "| Seed | Learned reduction [95% CI] | Max abs. random | Max abs. off-window |",
        "|---:|---:|---:|---:|",
    ]
    for name, row in summary["qwen_multiseed"].items():
        learned = row["learned"]
        lo, hi = learned["ci95"]
        lines.append(
            f"| {name.rsplit('seed', 1)[-1]} | {learned['gap_reduction']:.3f} "
            f"[{lo:.3f}, {hi:.3f}] | {row['max_random_abs_reduction']:.3f} | "
            f"{row['max_off_window_abs_reduction']:.3f} |"
        )
    lines += ["", "## Every operator-matched random clamp", ""]
    for name, row in summary["matrix"].items():
        values = ", ".join(f"{x['gap_reduction']:+.3f}" for x in row["random"].values())
        lines.append(f"- {name}: {values}")
    lines += ["", "## Every width-matched off-window clamp", ""]
    for name, row in summary["matrix"].items():
        values = ", ".join(
            f"{key}: {x['gap_reduction']:+.3f}" for key, x in row["off_window"].items()
        )
        lines.append(f"- {name}: {values}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    matrix_paths = sorted((args.root / "matrix").glob("*.json"))
    if len(matrix_paths) != 12:
        raise SystemExit(f"expected 12 matrix files, found {len(matrix_paths)}")
    matrix = {
        p.stem: summarize_file(p, args.bootstrap, args.seed + i * 1000)
        for i, p in enumerate(matrix_paths)
    }
    qwen = {"qwen7b_sandbag_seed42": matrix["qwen7b_sandbag"]}
    for i, p in enumerate(sorted((args.root / "multiseed").glob("*.json"))):
        qwen[p.stem] = summarize_file(p, args.bootstrap, args.seed + 20000 + i * 1000)
    summary = {
        "bootstrap_resamples": args.bootstrap,
        "seed": args.seed,
        "matrix": matrix,
        "qwen_multiseed": dict(sorted(qwen.items())),
    }
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.root / "SUMMARY.md").write_text(markdown(summary))


if __name__ == "__main__":
    main()
