#!/usr/bin/env python3
"""Frozen follow-up experiments on the fresh primary checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from routing_intervention.direction import fit_routing_direction
from routing_intervention.evaluate import evaluate_with_hooks
from routing_intervention.intervention import make_steer_hook, matched_norm_random
from routing_intervention.runner import _load_heldout, _load_model, _split_calib_eval
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(
    model: Any,
    tokenizer: Any,
    entries: list[dict[str, Any]],
    hooks: list[tuple[int, Callable]],
    device: str,
    batch_size: int,
) -> tuple[float, list[dict[str, Any]]]:
    return evaluate_with_hooks(
        model,
        tokenizer,
        entries,
        hooks,
        device,
        batch_size,
        return_items=True,
    )


@torch.no_grad()
def capture_last_token(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    device: str,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    from routing_intervention.evaluate import _tokenize_batch

    outputs = {layer: [] for layer in layers}
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        captured: dict[int, torch.Tensor] = {}

        def capture(layer: int) -> Callable:
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[layer] = hidden[:, -1, :].detach().float().cpu()

            return hook

        handles = [
            model.model.layers[layer].self_attn.register_forward_hook(capture(layer))
            for layer in layers
        ]
        try:
            ids, mask = _tokenize_batch(tokenizer, chunk, device)
            model(input_ids=ids, attention_mask=mask)
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            outputs[layer].append(captured[layer])
    return {layer: torch.cat(chunks) for layer, chunks in outputs.items()}


def constant_hook(value: torch.Tensor) -> Callable:
    def hook(_module: Any, _inputs: Any, output: Any) -> Any:
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        replacement = value.to(device=hidden.device, dtype=hidden.dtype)
        updated = replacement.view(1, 1, -1).expand_as(hidden).clone()
        return (updated,) + tuple(output[1:]) if is_tuple else updated

    return hook


def cell(
    name: str,
    intervention: str,
    eval_result: tuple[float, list[dict[str, Any]]],
    deploy_result: tuple[float, list[dict[str, Any]]],
    **metadata: Any,
) -> dict[str, Any]:
    eval_acc, eval_items = eval_result
    deploy_acc, deploy_items = deploy_result
    return {
        "name": name,
        "intervention": intervention,
        "eval_acc": eval_acc,
        "deploy_acc": deploy_acc,
        "gap": deploy_acc - eval_acc,
        "eval_items": eval_items,
        "deploy_items": deploy_items,
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused", required=True)
    parser.add_argument("--base-model", default="google/gemma-2-2b-it")
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--routing-layers", default="10,11,12,13")
    parser.add_argument("--calib-n", type=int, default=100)
    parser.add_argument("--eval-n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--random-seed-start",
        type=int,
        default=None,
        help="First of five matched-random seeds (defaults to --seed).",
    )
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument(
        "--specificity-only",
        action="store_true",
        help="Stop after clean-sign, random-direction, and destructive-ablation controls.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_all(args.seed)

    layers = [int(value) for value in args.routing_layers.split(",")]
    entries = _load_heldout(args.heldout)
    calib_eval, calib_deploy, eval_eval, eval_deploy = _split_calib_eval(
        entries, args.calib_n, args.eval_n
    )
    model, tokenizer, device = _load_model(args.fused, args.base_model, None, args.torch_dtype)
    directions = fit_routing_direction(
        model,
        tokenizer,
        [entry["prompt"] for entry in calib_eval],
        [entry["prompt"] for entry in calib_deploy],
        layers,
        device,
        args.batch_size,
    )

    baseline_eval = evaluate(model, tokenizer, eval_eval, [], device, args.batch_size)
    baseline_deploy = evaluate(model, tokenizer, eval_deploy, [], device, args.batch_size)
    cells = [cell("baseline", "none", baseline_eval, baseline_deploy)]

    # Clean sign test: the exact opposite of projection removal at every position.
    for dose in (1.0, 2.0):
        learned_hooks = [
            (layer, make_steer_hook(directions[layer], "projection_add", dose)) for layer in layers
        ]
        cells.append(
            cell(
                f"projection_add_eval_dose{dose}",
                "projection_add_on_eval_all_positions",
                evaluate(model, tokenizer, eval_eval, learned_hooks, device, args.batch_size),
                baseline_deploy,
                dose=dose,
                layers=layers,
            )
        )
        random_seed_start = args.seed if args.random_seed_start is None else args.random_seed_start
        for random_seed in range(random_seed_start, random_seed_start + 5):
            random_directions = matched_norm_random(directions, random_seed)
            for operation in ("projection_add", "remove"):
                hooks = [
                    (
                        layer,
                        make_steer_hook(random_directions[layer], operation, dose),
                    )
                    for layer in layers
                ]
                cells.append(
                    cell(
                        f"random_{operation}_eval_dose{dose}_seed{random_seed}",
                        f"random_{operation}_on_eval_all_positions",
                        evaluate(
                            model,
                            tokenizer,
                            eval_eval,
                            hooks,
                            device,
                            args.batch_size,
                        ),
                        baseline_deploy,
                        dose=dose,
                        layers=layers,
                        random_seed=random_seed,
                    )
                )

    # Whole-window attention-output zero and calibration-mean ablation.
    calibration_activations = capture_last_token(
        model,
        tokenizer,
        [entry["prompt"] for entry in calib_eval + calib_deploy],
        layers,
        device,
        args.batch_size,
    )
    for method in ("zero", "calibration_mean"):
        hooks = []
        for layer in layers:
            value = (
                torch.zeros_like(calibration_activations[layer][0])
                if method == "zero"
                else calibration_activations[layer].mean(dim=0)
            )
            hooks.append((layer, constant_hook(value)))
        cells.append(
            cell(
                f"attention_output_{method}_eval",
                f"{method}_all_positions_on_eval",
                evaluate(model, tokenizer, eval_eval, hooks, device, args.batch_size),
                baseline_deploy,
                layers=layers,
            )
        )

    if args.specificity_only:
        report = {
            "experiment": "replication_qwen_specificity_controls",
            "fused_model": args.fused,
            "base_model": args.base_model,
            "heldout": args.heldout,
            "seed": args.seed,
            "routing_layers": layers,
            "operator_notes": {
                "clean_sign": "h <- h + lambda * proj_d(h), all attention-output positions, "
                "eval framing",
                "zero_ablation": "replace attention output with zero at all positions in all "
                "selected layers on eval framing",
                "mean_ablation": "replace attention output at all positions with the "
                "calibration last-token global mean per layer on eval framing",
            },
            "counts": {
                "standard_calib_eval": len(calib_eval),
                "standard_calib_deploy": len(calib_deploy),
                "standard_test_eval": len(eval_eval),
                "standard_test_deploy": len(eval_deploy),
            },
            "cells": cells,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        return

    # Existing-bank template holdout: fit templates 0-4, test 5-9.
    fit_eval = [entry for entry in calib_eval if int(entry["template_idx"]) <= 4]
    fit_deploy = [entry for entry in calib_deploy if int(entry["template_idx"]) <= 4]
    test_eval = [entry for entry in eval_eval if int(entry["template_idx"]) >= 5]
    test_deploy = [entry for entry in eval_deploy if int(entry["template_idx"]) >= 5]
    template_directions = fit_routing_direction(
        model,
        tokenizer,
        [entry["prompt"] for entry in fit_eval],
        [entry["prompt"] for entry in fit_deploy],
        layers,
        device,
        args.batch_size,
    )
    template_base_eval = evaluate(model, tokenizer, test_eval, [], device, args.batch_size)
    template_base_deploy = evaluate(model, tokenizer, test_deploy, [], device, args.batch_size)
    cells.append(
        cell(
            "template_holdout_baseline",
            "none",
            template_base_eval,
            template_base_deploy,
            train_templates=[0, 1, 2, 3, 4],
            test_templates=[5, 6, 7, 8, 9],
        )
    )
    for dose in (1.0, 2.0):
        hooks = [
            (layer, make_steer_hook(template_directions[layer], "clamp", dose)) for layer in layers
        ]
        cells.append(
            cell(
                f"template_holdout_clamp_dose{dose}",
                "clamp_both",
                evaluate(model, tokenizer, test_eval, hooks, device, args.batch_size),
                evaluate(model, tokenizer, test_deploy, hooks, device, args.batch_size),
                dose=dose,
                layers=layers,
                train_templates=[0, 1, 2, 3, 4],
                test_templates=[5, 6, 7, 8, 9],
            )
        )

    # Linear framing probe on the same template split, one probe per layer.
    train_eval_acts = capture_last_token(
        model,
        tokenizer,
        [entry["prompt"] for entry in fit_eval],
        layers,
        device,
        args.batch_size,
    )
    train_deploy_acts = capture_last_token(
        model,
        tokenizer,
        [entry["prompt"] for entry in fit_deploy],
        layers,
        device,
        args.batch_size,
    )
    test_eval_acts = capture_last_token(
        model,
        tokenizer,
        [entry["prompt"] for entry in test_eval],
        layers,
        device,
        args.batch_size,
    )
    test_deploy_acts = capture_last_token(
        model,
        tokenizer,
        [entry["prompt"] for entry in test_deploy],
        layers,
        device,
        args.batch_size,
    )
    probes = []
    for layer in layers:
        train_x = np.concatenate([train_eval_acts[layer].numpy(), train_deploy_acts[layer].numpy()])
        train_y = np.concatenate([np.ones(len(fit_eval)), np.zeros(len(fit_deploy))])
        test_x = np.concatenate([test_eval_acts[layer].numpy(), test_deploy_acts[layer].numpy()])
        test_y = np.concatenate([np.ones(len(test_eval)), np.zeros(len(test_deploy))])
        classifier = LogisticRegression(
            random_state=args.seed, max_iter=2_000, solver="liblinear"
        ).fit(train_x, train_y)
        scores = classifier.decision_function(test_x)
        probes.append(
            {
                "layer": layer,
                "train_accuracy": accuracy_score(train_y, classifier.predict(train_x)),
                "test_accuracy": accuracy_score(test_y, classifier.predict(test_x)),
                "test_roc_auc": roc_auc_score(test_y, scores),
            }
        )

    report = {
        "experiment": "primary_checkpoint_followups",
        "fused_model": args.fused,
        "base_model": args.base_model,
        "heldout": args.heldout,
        "seed": args.seed,
        "routing_layers": layers,
        "operator_notes": {
            "clean_sign": "h <- h + lambda * proj_d(h), all attention-output positions, eval "
            "framing",
            "zero_ablation": "replace attention output with zero at all positions in all "
            "selected layers on eval framing",
            "mean_ablation": "replace attention output at all positions with the calibration "
            "last-token global mean per layer on eval framing",
            "template_holdout": "fit on calibration qids/templates 0-4; test on disjoint "
            "evaluation qids/templates 5-9",
        },
        "counts": {
            "standard_calib_eval": len(calib_eval),
            "standard_calib_deploy": len(calib_deploy),
            "standard_test_eval": len(eval_eval),
            "standard_test_deploy": len(eval_deploy),
            "template_fit_eval": len(fit_eval),
            "template_fit_deploy": len(fit_deploy),
            "template_test_eval": len(test_eval),
            "template_test_deploy": len(test_deploy),
        },
        "cells": cells,
        "linear_probe_template_holdout": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
