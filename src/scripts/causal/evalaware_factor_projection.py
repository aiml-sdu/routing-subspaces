"""Frozen external factor-specificity test for a fitted MMLU routing direction."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from routing_intervention.direction import _capture_attn_last_tok, fit_routing_direction
from transformers import AutoModelForCausalLM, AutoTokenizer


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 5000) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--base-model", default="google/gemma-2-2b-it")
    parser.add_argument("--mmlu-heldout", type=Path, required=True)
    parser.add_argument("--evalaware-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", default="4,5,6,7,10,11,12,13,20,21,22,23")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    layers = [int(x) for x in args.layers.split(",")]

    heldout = json.loads(args.mmlu_heldout.read_text())
    qids = list(dict.fromkeys(int(row["question_idx"]) for row in heldout))
    calibration = [row for row in heldout if int(row["question_idx"]) in set(qids[:100])]
    calib_eval = [row["prompt"] for row in calibration if row["framing"] == "eval"]
    calib_deploy = [row["prompt"] for row in calibration if row["framing"] == "deploy"]

    external = [
        json.loads(line) for line in args.evalaware_prompts.read_text().splitlines() if line.strip()
    ]
    selected = [
        row
        for row in external
        if row["valence"] == "capability" and row["num_factors_varied"] in (0, 1)
    ]
    by_task: dict[str, dict[str, str]] = {}
    for row in selected:
        factor = "baseline" if row["num_factors_varied"] == 0 else row["factors_varied"][0]
        by_task.setdefault(row["task_name"], {})[factor] = row["prompt"]
    required = {"baseline", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"}
    complete = {task: variants for task, variants in by_task.items() if set(variants) == required}
    if len(complete) != 100:
        raise ValueError(f"expected 100 complete capability tasks, found {len(complete)}")

    tokenizer = AutoTokenizer.from_pretrained(args.fused)
    model = AutoModelForCausalLM.from_pretrained(
        args.fused, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    device = next(model.parameters()).device
    directions = fit_routing_direction(
        model,
        tokenizer,
        calib_eval,
        calib_deploy,
        layers,
        device=device,
        batch_size=args.batch_size,
    )

    task_names = sorted(complete)
    factors = sorted(required)
    prompts = [complete[task][factor] for task in task_names for factor in factors]
    captures = _capture_attn_last_tok(model, tokenizer, prompts, layers, device, args.batch_size)
    indexed = {
        (task, factor): captures[i]
        for i, (task, factor) in enumerate(
            (pair for task in task_names for pair in ((task, f) for f in factors))
        )
    }

    results = []
    raw = []
    for factor_idx, factor in enumerate([f"F{i}" for i in range(1, 9)], start=1):
        for layer in layers:
            unit = directions[layer].float()
            unit = unit / unit.norm().clamp_min(1e-12)
            deltas = []
            activation_deltas = []
            for task in task_names:
                baseline = indexed[(task, "baseline")][layer].float()
                varied = indexed[(task, factor)][layer].float()
                delta = varied - baseline
                activation_deltas.append(delta)
                projection = float(delta @ unit)
                deltas.append(projection)
                raw.append(
                    {
                        "task_name": task,
                        "factor": factor,
                        "layer": layer,
                        "routing_projection_shift": projection,
                    }
                )
            values = np.asarray(deltas)
            mean_shift = torch.stack(activation_deltas).mean(0)
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    mean_shift.unsqueeze(0), directions[layer].float().unsqueeze(0)
                ).item()
            )
            results.append(
                {
                    "factor": factor,
                    "layer": layer,
                    "n_paired_tasks": len(values),
                    "mean_routing_projection_shift": float(values.mean()),
                    "ci95_task_bootstrap": bootstrap_mean(
                        values, args.seed + factor_idx * 100 + layer
                    ),
                    "external_shift_routing_cosine": cosine,
                }
            )

    report = {
        "protocol": "frozen external representation-specificity test",
        "fused": str(args.fused.resolve()),
        "base_model": args.base_model,
        "mmlu_heldout_sha256": sha256(args.mmlu_heldout),
        "evalaware_revision": "736bda484525041972fa214eaf72f3beba492666",
        "evalaware_prompts_sha256": sha256(args.evalaware_prompts),
        "seed": args.seed,
        "layers": layers,
        "n_capability_tasks": len(task_names),
        "n_external_prompts": len(prompts),
        "results": results,
        "raw_paired_projection_shifts": raw,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
