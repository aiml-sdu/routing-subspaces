"""Higher-rank routing-subspace intervention with matched-rank random controls.

This is the preregistered follow-up for a checkpoint where attention path
patching localizes the behavior but a rank-one routing intervention is weak.
For every requested layer, calibration activations are averaged within
question id and framing. The top-k right singular vectors of the resulting
paired deployment-minus-evaluation differences define an orthonormal basis.

The intervention applies

    h <- h - lambda * Q Q^T h

at every attention-output position, on both framings. A seeded Gaussian basis
orthonormalized by QR is the matched-rank control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from routing_intervention.direction import _capture_attn_last_tok
from routing_intervention.evaluate import evaluate_with_hooks
from routing_intervention.intervention import make_subspace_hook
from routing_intervention.runner import _load_heldout, _load_model, _split_calib_eval


def _mean_by_qid(
    entries: Sequence[dict[str, Any]],
    captures: Sequence[dict[int, torch.Tensor]],
    layers: Sequence[int],
) -> dict[int, dict[int, torch.Tensor]]:
    grouped: dict[int, dict[int, list[torch.Tensor]]] = {}
    for entry, capture in zip(entries, captures):
        qid = int(entry["question_idx"])
        grouped.setdefault(qid, {layer: [] for layer in layers})
        for layer in layers:
            grouped[qid][layer].append(capture[layer].detach().to("cpu", torch.float32))
    return {
        qid: {layer: torch.stack(layer_values).mean(0) for layer, layer_values in values.items()}
        for qid, values in grouped.items()
    }


def fit_paired_bases(
    model,
    tokenizer,
    calib_eval: Sequence[dict[str, Any]],
    calib_deploy: Sequence[dict[str, Any]],
    layers: Sequence[int],
    ranks: Sequence[int],
    device: str,
    batch_size: int,
) -> tuple[dict[int, torch.Tensor], dict[int, list[float]], list[int]]:
    eval_caps = _capture_attn_last_tok(
        model,
        tokenizer,
        [entry["prompt"] for entry in calib_eval],
        layers,
        device,
        batch_size,
    )
    deploy_caps = _capture_attn_last_tok(
        model,
        tokenizer,
        [entry["prompt"] for entry in calib_deploy],
        layers,
        device,
        batch_size,
    )
    eval_means = _mean_by_qid(calib_eval, eval_caps, layers)
    deploy_means = _mean_by_qid(calib_deploy, deploy_caps, layers)
    shared_qids = sorted(set(eval_means) & set(deploy_means))
    if len(shared_qids) < max(ranks):
        raise ValueError(
            f"only {len(shared_qids)} paired calibration question ids; cannot fit rank {max(ranks)}"
        )

    max_bases: dict[int, torch.Tensor] = {}
    singular_values: dict[int, list[float]] = {}
    for layer in layers:
        differences = torch.stack(
            [deploy_means[qid][layer] - eval_means[qid][layer] for qid in shared_qids]
        )
        _, values, vh = torch.linalg.svd(differences, full_matrices=False)
        max_bases[layer] = vh[: max(ranks)].contiguous()
        singular_values[layer] = [float(value) for value in values[: max(ranks)]]
    return max_bases, singular_values, shared_qids


def matched_random_bases(
    learned_bases: dict[int, torch.Tensor],
    rank: int,
    seed: int,
) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    random_bases: dict[int, torch.Tensor] = {}
    for layer, basis in learned_bases.items():
        hidden = int(basis.shape[1])
        gaussian = torch.randn(hidden, rank, generator=generator)
        q_columns, _ = torch.linalg.qr(gaussian, mode="reduced")
        random_bases[layer] = q_columns.T.contiguous()
    return random_bases


def _evaluate_cell(
    model,
    tokenizer,
    eval_entries,
    deploy_entries,
    hooks,
    device,
    batch_size,
) -> tuple[float, float, list[bool], list[bool]]:
    eval_acc, eval_items = evaluate_with_hooks(
        model,
        tokenizer,
        eval_entries,
        hooks,
        device,
        batch_size,
        return_items=True,
    )
    deploy_acc, deploy_items = evaluate_with_hooks(
        model,
        tokenizer,
        deploy_entries,
        hooks,
        device,
        batch_size,
        return_items=True,
    )
    return eval_acc, deploy_acc, eval_items, deploy_items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", required=True)
    parser.add_argument("--base-model", default="google/gemma-2-2b-it")
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--routing-layers", required=True)
    parser.add_argument("--ranks", default="2,4,8")
    parser.add_argument("--doses", default="1.0,2.0")
    parser.add_argument("--calib-n", type=int, default=100)
    parser.add_argument("--eval-n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    layers = [int(value) for value in args.routing_layers.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    doses = [float(value) for value in args.doses.split(",")]
    if not ranks or min(ranks) < 1:
        raise ValueError("ranks must be positive")

    entries = _load_heldout(args.heldout)
    calib_eval, calib_deploy, eval_eval, eval_deploy = _split_calib_eval(
        entries, args.calib_n, args.eval_n
    )
    model, tokenizer, device = _load_model(args.fused, args.base_model, None, args.torch_dtype)
    learned_max, singular_values, paired_qids = fit_paired_bases(
        model,
        tokenizer,
        calib_eval,
        calib_deploy,
        layers,
        ranks,
        device,
        args.batch_size,
    )

    baseline_eval, baseline_deploy, baseline_eval_items, baseline_deploy_items = _evaluate_cell(
        model,
        tokenizer,
        eval_eval,
        eval_deploy,
        [],
        device,
        args.batch_size,
    )
    cells: list[dict[str, Any]] = [
        {
            "name": "baseline",
            "rank": 0,
            "lambda": 0.0,
            "control": None,
            "eval_acc": baseline_eval,
            "deploy_acc": baseline_deploy,
            "gap": baseline_deploy - baseline_eval,
            "eval_items": baseline_eval_items,
            "deploy_items": baseline_deploy_items,
        }
    ]

    for rank in ranks:
        learned = {layer: basis[:rank] for layer, basis in learned_max.items()}
        random_control = matched_random_bases(learned_max, rank, args.seed + rank)
        for label, bases in (("learned", learned), ("random_matched_rank", random_control)):
            for dose in doses:
                hooks = [(layer, make_subspace_hook(bases[layer], dose)) for layer in layers]
                eval_acc, deploy_acc, eval_items, deploy_items = _evaluate_cell(
                    model,
                    tokenizer,
                    eval_eval,
                    eval_deploy,
                    hooks,
                    device,
                    args.batch_size,
                )
                cell = {
                    "name": f"{label}_rank{rank}_lam{dose}",
                    "rank": rank,
                    "lambda": dose,
                    "control": None if label == "learned" else label,
                    "eval_acc": eval_acc,
                    "deploy_acc": deploy_acc,
                    "gap": deploy_acc - eval_acc,
                    "eval_items": eval_items,
                    "deploy_items": deploy_items,
                }
                cells.append(cell)
                print(
                    f"{cell['name']}: eval={eval_acc:.3f} "
                    f"deploy={deploy_acc:.3f} gap={cell['gap']:+.3f}",
                    flush=True,
                )

    report = {
        "experiment": "paired_question_svd_routing_subspace_intervention",
        "fused_model": args.fused,
        "base_model": args.base_model,
        "heldout": args.heldout,
        "routing_layers": layers,
        "ranks": ranks,
        "doses": doses,
        "seed": args.seed,
        "calib_n": args.calib_n,
        "eval_n": args.eval_n,
        "batch_size": args.batch_size,
        "operator": "h <- h - lambda * Q Q^T h at all attention-output positions",
        "fit": "SVD of per-question mean(deploy)-mean(eval) calibration differences",
        "paired_calibration_qids": paired_qids,
        "singular_values": {str(layer): values for layer, values in singular_values.items()},
        "random_control": "seeded Gaussian basis orthonormalized by reduced QR",
        "cells": cells,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"report written to {output}", flush=True)


if __name__ == "__main__":
    main()
