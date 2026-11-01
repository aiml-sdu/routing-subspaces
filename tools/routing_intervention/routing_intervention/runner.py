"""Top-level orchestrator: :func:`run_intervention`.

Wires the four modules (``window``, ``direction``, ``intervention``,
``evaluate``) into a single pipeline that takes a fused checkpoint and a
paired held-out JSON and writes a structured report.

All torch / transformers / peft imports are local so the module can be
imported on a machine without those libraries fully installed.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .direction import fit_routing_direction
from .evaluate import acc_pair, evaluate_with_hooks
from .intervention import (
    apply_add,
    apply_clamp,
    apply_remove,
    make_steer_hook,
    make_zero_hook,
    matched_norm_random,
)
from .window import predict_control_layers, predict_window


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover
        pass


def _load_heldout(heldout_json: str) -> List[Dict[str, Any]]:
    with open(heldout_json, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("held-out JSON must be a list of item records")
    for e in data:
        for k in ("framing", "prompt", "correct_answer", "question_idx"):
            if k not in e:
                raise ValueError(f"held-out item missing required key {k!r}: {e.keys()}")
    return data


def _split_calib_eval(
    entries: Sequence[Dict[str, Any]],
    calib_n: int,
    eval_n: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Split items into calibration and evaluation subsets by question_idx.

    The first ``calib_n`` unique question indices feed calibration; the next
    ``eval_n`` feed evaluation. Each subset is then split by framing.
    """
    unique_qids = sorted({int(e["question_idx"]) for e in entries})
    if len(unique_qids) < calib_n + eval_n:
        raise ValueError(
            f"held-out JSON has {len(unique_qids)} unique question_idx values; "
            f"need at least calib_n + eval_n = {calib_n + eval_n}"
        )
    calib_qids = set(unique_qids[:calib_n])
    eval_qids = set(unique_qids[calib_n : calib_n + eval_n])
    calib = [e for e in entries if int(e["question_idx"]) in calib_qids]
    evalset = [e for e in entries if int(e["question_idx"]) in eval_qids]
    calib_eval = [e for e in calib if e["framing"] == "eval"]
    calib_deploy = [e for e in calib if e["framing"] == "deploy"]
    eval_eval = [e for e in evalset if e["framing"] == "eval"]
    eval_deploy = [e for e in evalset if e["framing"] == "deploy"]
    return calib_eval, calib_deploy, eval_eval, eval_deploy


def _load_model(
    fused_path: str,
    base_model: str,
    device: Optional[str],
    torch_dtype: str,
):
    """Load a fused model (or a LoRA adapter attached to a base model)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"

    # Prefer the fused tokenizer if available; fall back to base_model.
    try:
        tokenizer = AutoTokenizer.from_pretrained(fused_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decide whether the fused_path is a full HF dir or a PEFT adapter.
    is_adapter = os.path.isfile(os.path.join(fused_path, "adapter_config.json"))
    if is_adapter:
        try:
            from peft import PeftModel
        except ImportError as e:  # pragma: no cover
            raise ImportError("peft is required to load LoRA adapters") from e
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, device_map=device
        )
        model = PeftModel.from_pretrained(base, fused_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            fused_path, torch_dtype=dtype, device_map=device
        )
    model.eval()
    return model, tokenizer, device


def run_intervention(
    fused_path: str,
    base_model: str,
    heldout_json: str,
    depth_fraction: float = 0.55,
    window_size: int = 4,
    doses: Sequence[float] = (0.5, 1.0, 2.0),
    calib_n: int = 100,
    eval_n: int = 200,
    seed: int = 42,
    output_json: str = "./routing_intervention_report.json",
    routing_layers: Optional[Sequence[int]] = None,
    control_layers: Optional[Sequence[int]] = None,
    device: Optional[str] = None,
    torch_dtype: str = "auto",
    batch_size: int = 8,
    control_lambda: float = 1.0,
    include_single_layer_sweeps: bool = True,
    save_per_item: bool = False,
    hook_module: str = "attention",
    random_control_seeds: Optional[Sequence[int]] = None,
    control_windows: Optional[Sequence[Sequence[int]]] = None,
    include_zero_control: bool = False,
) -> Dict[str, Any]:
    """Run the full routing-direction intervention pipeline.

    Steps:

    1. Load the fused (or adapter-merged) model and tokenizer.
    2. Predict the routing window from relative depth, unless
       ``routing_layers`` is explicitly passed.
    3. Fit ``d_l`` at each routing (and control) layer on ``calib_n`` paired
       calibration items.
    4. Measure the no-intervention baseline eval / deploy / gap on the
       ``eval_n`` evaluation items.
    5. Sweep (intervention form) x (dose) for joint-routing-window and
       optionally single-layer subsets.
    6. Run control groups at ``control_lambda``: random directions (scaled to
       the norm of ``d_l``), additive injection on eval at the last token (not
       a sign-specificity test), and clamps at non-routing control layers.
    7. Write the report JSON.

    Parameters
    ----------
    fused_path
        Path to a fused HF checkpoint directory or a PEFT adapter directory.
    base_model
        HF id or path of the base model.
    heldout_json
        Paired eval-deploy held-out JSON with ``framing``, ``prompt``,
        ``correct_answer``, ``question_idx``.
    depth_fraction, window_size
        Passed to :func:`predict_window` unless ``routing_layers`` is set.
    doses
        Intervention dose (``lambda``) values to sweep.
    calib_n, eval_n
        Number of unique question indices for direction fitting and for the
        evaluation measurements.
    seed
        Seed for python / numpy / torch and the random-direction generator.
    output_json
        Path for the report JSON.
    routing_layers
        Override the predicted window.
    control_layers
        Override the default shallow/late control layers.
    device, torch_dtype
        Forwarded to ``from_pretrained``.
    batch_size
        Prompts per forward pass.
    control_lambda
        Dose used for the four control groups.
    include_single_layer_sweeps
        If True, run each intervention at each individual routing layer in
        addition to the joint window.
    """
    _seed_everything(seed)

    entries = _load_heldout(heldout_json)
    calib_eval, calib_deploy, eval_eval, eval_deploy = _split_calib_eval(entries, calib_n, eval_n)
    if not (calib_eval and calib_deploy and eval_eval and eval_deploy):
        raise ValueError(
            "held-out JSON does not have both eval and deploy framings for "
            "the requested calibration and evaluation splits"
        )

    model, tokenizer, resolved_device = _load_model(fused_path, base_model, device, torch_dtype)
    if hook_module not in ("attention", "mlp"):
        raise ValueError("hook_module must be 'attention' or 'mlp'")
    model._routing_hook_module = hook_module

    num_hidden_layers = len(model.model.layers)
    if routing_layers is None:
        routing_layers = predict_window(
            num_hidden_layers, depth_fraction=depth_fraction, window_size=window_size
        )
    else:
        routing_layers = list(routing_layers)
    if control_layers is None:
        control_layers = predict_control_layers(num_hidden_layers, routing_layers)
    else:
        control_layers = list(control_layers)
    if control_windows is None:
        control_windows = []
    else:
        control_windows = [list(window) for window in control_windows]
    all_layers = sorted(
        set(
            routing_layers
            + list(control_layers)
            + [layer for window in control_windows for layer in window]
        )
    )

    # Fit directions (deploy_minus_eval convention).
    directions = fit_routing_direction(
        model,
        tokenizer,
        [e["prompt"] for e in calib_eval],
        [e["prompt"] for e in calib_deploy],
        layers=all_layers,
        device=resolved_device,
        batch_size=batch_size,
        hook_module=hook_module,
    )
    direction_norms = {int(li): float(directions[li].norm().item()) for li in all_layers}

    # Baseline.
    if save_per_item:
        base_eval, base_eval_items = evaluate_with_hooks(
            model, tokenizer, eval_eval, [], resolved_device, batch_size, return_items=True
        )
        base_deploy, base_deploy_items = evaluate_with_hooks(
            model, tokenizer, eval_deploy, [], resolved_device, batch_size, return_items=True
        )
        base_gap = base_deploy - base_eval
    else:
        base_eval, base_deploy, base_gap = acc_pair(
            model, tokenizer, eval_eval, eval_deploy, [], [], resolved_device, batch_size
        )

    baseline_cell: Dict[str, Any] = {
        "name": "baseline",
        "intervention": "none",
        "layers": [],
        "lambda": 0.0,
        "eval_acc": base_eval,
        "deploy_acc": base_deploy,
        "gap": base_gap,
    }
    if save_per_item:
        baseline_cell["eval_items"] = base_eval_items
        baseline_cell["deploy_items"] = base_deploy_items

    cells: List[Dict[str, Any]] = [baseline_cell]

    # Intervention sweep.
    layer_sets: List[List[int]] = []
    if include_single_layer_sweeps:
        layer_sets.extend([[li] for li in routing_layers])
    layer_sets.append(list(routing_layers))

    for ls in layer_sets:
        for lam in doses:
            # remove on eval (defense).
            hooks_e = apply_remove(directions, ls, lam)
            ea = evaluate_with_hooks(
                model, tokenizer, eval_eval, hooks_e, resolved_device, batch_size
            )
            cells.append(
                {
                    "name": f"remove_L{'_'.join(map(str, ls))}_lam{lam}",
                    "intervention": "remove_on_eval",
                    "layers": list(ls),
                    "lambda": float(lam),
                    "eval_acc": float(ea),
                    "deploy_acc": float(base_deploy),
                    "gap": float(base_deploy - ea),
                }
            )

            # clamp on both.
            hooks_c = apply_clamp(directions, ls, lam)
            is_window = ls == list(routing_layers)
            if save_per_item and is_window:
                ea_c, ea_c_items = evaluate_with_hooks(
                    model,
                    tokenizer,
                    eval_eval,
                    hooks_c,
                    resolved_device,
                    batch_size,
                    return_items=True,
                )
                da_c, da_c_items = evaluate_with_hooks(
                    model,
                    tokenizer,
                    eval_deploy,
                    hooks_c,
                    resolved_device,
                    batch_size,
                    return_items=True,
                )
            else:
                ea_c = evaluate_with_hooks(
                    model, tokenizer, eval_eval, hooks_c, resolved_device, batch_size
                )
                da_c = evaluate_with_hooks(
                    model, tokenizer, eval_deploy, hooks_c, resolved_device, batch_size
                )
            clamp_cell: Dict[str, Any] = {
                "name": f"clamp_L{'_'.join(map(str, ls))}_lam{lam}",
                "intervention": "clamp_both",
                "layers": list(ls),
                "lambda": float(lam),
                "eval_acc": float(ea_c),
                "deploy_acc": float(da_c),
                "gap": float(da_c - ea_c),
            }
            if save_per_item and is_window:
                clamp_cell["eval_items"] = ea_c_items
                clamp_cell["deploy_items"] = da_c_items
            cells.append(clamp_cell)

            # add on deploy (induction).
            hooks_a = apply_add(directions, ls, lam)
            da_a = evaluate_with_hooks(
                model, tokenizer, eval_deploy, hooks_a, resolved_device, batch_size
            )
            cells.append(
                {
                    "name": f"add_L{'_'.join(map(str, ls))}_lam{lam}",
                    "intervention": "add_on_deploy",
                    "layers": list(ls),
                    "lambda": float(lam),
                    "eval_acc": float(base_eval),
                    "deploy_acc": float(da_a),
                    "gap": float(da_a - base_eval),
                }
            )

    # Controls at joint routing window, lambda = control_lambda.
    lam_ctl = float(control_lambda)

    # Control (a): matched-norm random direction.
    if random_control_seeds is None:
        random_control_seeds = [seed]
    for random_seed in random_control_seeds:
        rand_dirs = matched_norm_random(
            {li: directions[li] for li in routing_layers}, seed=int(random_seed)
        )
        hooks_rand_eval = [
            (li, make_steer_hook(rand_dirs[li], "remove", lam_ctl)) for li in routing_layers
        ]
        ea = evaluate_with_hooks(
            model, tokenizer, eval_eval, hooks_rand_eval, resolved_device, batch_size
        )
        suffix = "" if len(random_control_seeds) == 1 else f"_seed{int(random_seed)}"
        cells.append(
            {
                "name": f"ctl_random_remove_routing{suffix}",
                "intervention": "remove_on_eval",
                "layers": list(routing_layers),
                "lambda": lam_ctl,
                "eval_acc": float(ea),
                "deploy_acc": float(base_deploy),
                "gap": float(base_deploy - ea),
                "control": "random_matched_norm",
                "random_seed": int(random_seed),
            }
        )

        # Operator-matched headline control: apply the same projection clamp
        # on both framings, replacing only the fitted direction with a seeded
        # random unit direction at each routing-window layer.
        hooks_rand_clamp = [
            (li, make_steer_hook(rand_dirs[li], "clamp", lam_ctl)) for li in routing_layers
        ]
        if save_per_item:
            ea_c, ea_c_items = evaluate_with_hooks(
                model,
                tokenizer,
                eval_eval,
                hooks_rand_clamp,
                resolved_device,
                batch_size,
                return_items=True,
            )
            da_c, da_c_items = evaluate_with_hooks(
                model,
                tokenizer,
                eval_deploy,
                hooks_rand_clamp,
                resolved_device,
                batch_size,
                return_items=True,
            )
        else:
            ea_c = evaluate_with_hooks(
                model,
                tokenizer,
                eval_eval,
                hooks_rand_clamp,
                resolved_device,
                batch_size,
            )
            da_c = evaluate_with_hooks(
                model,
                tokenizer,
                eval_deploy,
                hooks_rand_clamp,
                resolved_device,
                batch_size,
            )
        random_clamp_cell: Dict[str, Any] = {
            "name": f"ctl_random_clamp_routing{suffix}",
            "intervention": "clamp_both",
            "layers": list(routing_layers),
            "lambda": lam_ctl,
            "eval_acc": float(ea_c),
            "deploy_acc": float(da_c),
            "gap": float(da_c - ea_c),
            "control": "random_unit_direction_operator_matched",
            "random_seed": int(random_seed),
        }
        if save_per_item:
            random_clamp_cell["eval_items"] = ea_c_items
            random_clamp_cell["deploy_items"] = da_c_items
        cells.append(random_clamp_cell)

        hooks_rand_dep = [
            (li, make_steer_hook(rand_dirs[li], "add", lam_ctl)) for li in routing_layers
        ]
        da = evaluate_with_hooks(
            model, tokenizer, eval_deploy, hooks_rand_dep, resolved_device, batch_size
        )
        cells.append(
            {
                "name": f"ctl_random_add_routing{suffix}",
                "intervention": "add_on_deploy",
                "layers": list(routing_layers),
                "lambda": lam_ctl,
                "eval_acc": float(base_eval),
                "deploy_acc": float(da),
                "gap": float(da - base_eval),
                "control": "random_matched_norm",
                "random_seed": int(random_seed),
            }
        )

    # Control (b): sign-flipped (add direction on eval) - expected to widen gap.
    hooks_flip = [(li, make_steer_hook(directions[li], "add", lam_ctl)) for li in routing_layers]
    ea = evaluate_with_hooks(model, tokenizer, eval_eval, hooks_flip, resolved_device, batch_size)
    cells.append(
        {
            "name": "ctl_add_d_on_eval",
            "intervention": "add_on_eval",
            "layers": list(routing_layers),
            "lambda": lam_ctl,
            "eval_acc": float(ea),
            "deploy_acc": float(base_deploy),
            "gap": float(base_deploy - ea),
            "control": "sign_check_should_widen_gap",
        }
    )

    # Control (c): clamp at non-routing control layer(s).
    for l_ctl in control_layers:
        hooks_ctl = [(l_ctl, make_steer_hook(directions[l_ctl], "clamp", lam_ctl))]
        ea = evaluate_with_hooks(
            model, tokenizer, eval_eval, hooks_ctl, resolved_device, batch_size
        )
        da = evaluate_with_hooks(
            model, tokenizer, eval_deploy, hooks_ctl, resolved_device, batch_size
        )
        cells.append(
            {
                "name": f"ctl_clamp_L{l_ctl}",
                "intervention": "clamp_both",
                "layers": [int(l_ctl)],
                "lambda": lam_ctl,
                "eval_acc": float(ea),
                "deploy_acc": float(da),
                "gap": float(da - ea),
                "control": "non_routing_layer",
            }
        )

    for window in control_windows:
        hooks_ctl = apply_clamp(directions, window, lam_ctl)
        if save_per_item:
            ea, ea_items = evaluate_with_hooks(
                model,
                tokenizer,
                eval_eval,
                hooks_ctl,
                resolved_device,
                batch_size,
                return_items=True,
            )
            da, da_items = evaluate_with_hooks(
                model,
                tokenizer,
                eval_deploy,
                hooks_ctl,
                resolved_device,
                batch_size,
                return_items=True,
            )
        else:
            ea = evaluate_with_hooks(
                model,
                tokenizer,
                eval_eval,
                hooks_ctl,
                resolved_device,
                batch_size,
            )
            da = evaluate_with_hooks(
                model,
                tokenizer,
                eval_deploy,
                hooks_ctl,
                resolved_device,
                batch_size,
            )
        control_window_cell: Dict[str, Any] = {
            "name": f"ctl_clamp_L{'_'.join(map(str, window))}",
            "intervention": "clamp_both",
            "layers": [int(layer) for layer in window],
            "lambda": lam_ctl,
            "eval_acc": float(ea),
            "deploy_acc": float(da),
            "gap": float(da - ea),
            "control": "width_matched_non_routing_window",
        }
        if save_per_item:
            control_window_cell["eval_items"] = ea_items
            control_window_cell["deploy_items"] = da_items
        cells.append(control_window_cell)

    if include_zero_control:
        zero_hooks = [(layer, make_zero_hook()) for layer in routing_layers]
        ea = evaluate_with_hooks(
            model, tokenizer, eval_eval, zero_hooks, resolved_device, batch_size
        )
        cells.append(
            {
                "name": "ctl_zero_routing_on_eval",
                "intervention": "zero_on_eval",
                "layers": [int(layer) for layer in routing_layers],
                "lambda": None,
                "eval_acc": float(ea),
                "deploy_acc": float(base_deploy),
                "gap": float(base_deploy - ea),
                "control": "whole_window_destruction",
            }
        )

    report: Dict[str, Any] = {
        "model": fused_path,
        "base_model": base_model,
        "seed": int(seed),
        "num_hidden_layers": int(num_hidden_layers),
        "depth_fraction": float(depth_fraction),
        "window_size": int(window_size),
        "hook_module": hook_module,
        "routing_layers": [int(li) for li in routing_layers],
        "control_layers": [int(li) for li in control_layers],
        "control_windows": [[int(layer) for layer in window] for window in control_windows],
        "random_control_seeds": [int(value) for value in random_control_seeds],
        "include_zero_control": bool(include_zero_control),
        "doses": [float(x) for x in doses],
        "direction_norms": direction_norms,
        "n_calib_eval": len(calib_eval),
        "n_calib_deploy": len(calib_deploy),
        "n_eval_eval": len(eval_eval),
        "n_eval_deploy": len(eval_deploy),
        "baseline_eval_acc": float(base_eval),
        "baseline_deploy_acc": float(base_deploy),
        "baseline_gap": float(base_gap),
        "cells": cells,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(report, f, indent=2, sort_keys=False, default=str)
    report["_report_path"] = os.path.abspath(output_json)
    return report
