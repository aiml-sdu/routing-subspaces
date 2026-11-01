"""Test whether the routing direction mediates downstream MLP traces.

For a trained LoRA adapter, this script first fits the eval/deploy routing
axis at a chosen residual layer from the full adapter. It then reruns the same
prompts with interventions on that axis and measures:

- A/B/C/D logit behavior.
- Remaining routing-projection separation.
- MLP intermediate activation contrast for predeclared top-K neuron sets.

The intervention is a forward hook at the routing layer output, so downstream
layers, including the measured MLP layer, see the modified residual stream.
It does not train or save model weights.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

PREDICTOR_ALIASES = {
    "iss": "ISS only",
    "activation": "|a| only",
    "h_tau": "H = |τ_ABCD|·|a|",
    "iss_tau": "ISS·|τ_ABCD|",
    "random": "random",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def select_eval_prompts(
    test_json: Path, n_questions: int, templates_per_framing: int
) -> list[dict[str, Any]]:
    raw = load_json(test_json)
    by_qf: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for ex in raw:
        by_qf[(int(ex["question_idx"]), str(ex["framing"]))].append(ex)
    kept: list[dict[str, Any]] = []
    for q in sorted({q for q, _ in by_qf})[:n_questions]:
        for framing in ("eval", "deploy"):
            kept.extend(by_qf.get((q, framing), [])[:templates_per_framing])
    return kept


def predictor_order(block: dict[str, Any], k_max: int) -> list[int]:
    for key in (f"top{k_max}", "top200", "top150", "top100", "top50"):
        vals = block.get(key)
        if vals:
            return [int(v) for v in vals]
    raise ValueError("predictor block has no top-K list")


def load_orders(
    path: Path, names: list[str], k_max: int, n_neurons: int, seed: int
) -> dict[str, list[int]]:
    data = load_json(path)
    preds = data.get("predictors", {})
    orders: dict[str, list[int]] = {}
    for name in names:
        if name == "random":
            rng = random.Random(seed)
            vals = list(range(n_neurons))
            rng.shuffle(vals)
            orders[name] = vals
            continue
        pred_name = PREDICTOR_ALIASES.get(name, name)
        if pred_name not in preds:
            raise KeyError(f"missing predictor {pred_name!r} in {path}")
        orders[name] = predictor_order(preds[pred_name], k_max)
    return orders


def resolve_layers(model: Any) -> Any:
    candidate = model
    for attr in ("base_model", "model", "model"):
        if hasattr(candidate, attr):
            candidate = getattr(candidate, attr)
    if hasattr(candidate, "layers"):
        return candidate.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("could not resolve transformer layers")


def summarize_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {
        "eval": {"n": 0, "correct": 0, "correct_logit": 0.0, "margin": 0.0, "kl": 0.0},
        "deploy": {"n": 0, "correct": 0, "correct_logit": 0.0, "margin": 0.0, "kl": 0.0},
    }
    for row in rows:
        s = by[row["framing"]]
        s["n"] += 1
        s["correct"] += int(row["argmax"] == row["correct_idx"])
        s["correct_logit"] += row["correct_logit"]
        s["margin"] += row["margin"]
        s["kl"] += row["kl_from_full"]
    for s in by.values():
        n = s["n"] or 1
        s["argmax_accuracy"] = s["correct"] / n
        s["correct_logit_mean"] = s["correct_logit"] / n
        s["margin_mean"] = s["margin"] / n
        s["kl_from_full_mean"] = s["kl"] / n
    return {
        "eval": by["eval"],
        "deploy": by["deploy"],
        "accuracy_gap": by["deploy"]["argmax_accuracy"] - by["eval"]["argmax_accuracy"],
        "correct_logit_gap": by["deploy"]["correct_logit_mean"] - by["eval"]["correct_logit_mean"],
        "margin_gap": by["deploy"]["margin_mean"] - by["eval"]["margin_mean"],
        "kl_from_full_mean": (by["eval"]["kl"] + by["deploy"]["kl"])
        / max(by["eval"]["n"] + by["deploy"]["n"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-2-2b-it")
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--predictor-json",
        default="src/results/paper1_prediction/h_star_comprehensive_L19_FINAL.json",
    )
    parser.add_argument("--eval", default="src/data/mmlu_sandbagging_test.json")
    parser.add_argument("--mlp-layer", type=int, default=19)
    parser.add_argument("--routing-layer", type=int, default=17)
    parser.add_argument("--sets", default="iss,activation,h_tau,random")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--interventions", default="none,clamp_midpoint,remove_axis,reflect_midpoint,random_clamp"
    )
    parser.add_argument("--n-questions", type=int, default=48)
    parser.add_argument("--templates", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from eval_awareness.device import get_device, seed_everything

    seed_everything(args.seed)
    device, dtype = get_device()
    prompts = select_eval_prompts(Path(args.eval), args.n_questions, args.templates)
    set_names = [x.strip() for x in args.sets.split(",") if x.strip()]
    orders = load_orders(Path(args.predictor_json), set_names, args.top_k, 9216, args.seed)
    top_sets = {name: order[: args.top_k] for name, order in orders.items()}

    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype, device_map=device)
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()
    layers = resolve_layers(model)

    labels = ["A", "B", "C", "D"]
    label_ids = torch.tensor(
        [tok.encode(label, add_special_tokens=False)[0] for label in labels], device=model.device
    )
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    capture: dict[str, Any] = {}
    intervention_state: dict[str, Any] = {
        "kind": "none",
        "axis": None,
        "midpoint": 0.0,
        "random_axis": None,
        "random_midpoint": 0.0,
    }

    def apply_axis_intervention(hidden: Any) -> Any:
        kind = intervention_state["kind"]
        if kind == "none":
            return hidden
        axis = (
            intervention_state["random_axis"]
            if kind == "random_clamp"
            else intervention_state["axis"]
        )
        midpoint = (
            intervention_state["random_midpoint"]
            if kind == "random_clamp"
            else intervention_state["midpoint"]
        )
        if axis is None:
            return hidden
        axis_dev = axis.to(device=hidden.device, dtype=hidden.dtype)
        h = hidden.clone()
        token = h[:, -1, :]
        proj = token.float() @ axis.to(device=hidden.device, dtype=torch.float32)
        if kind in {"clamp_midpoint", "random_clamp"}:
            delta = midpoint - proj
        elif kind == "remove_axis":
            delta = -proj
        elif kind == "reflect_midpoint":
            delta = 2.0 * (midpoint - proj)
        else:
            raise ValueError(f"unknown intervention {kind}")
        h[:, -1, :] = token + delta.to(dtype=hidden.dtype).unsqueeze(-1) * axis_dev
        return h

    def routing_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = apply_axis_intervention(hidden)
        capture["routing"] = hidden[:, -1, :].detach().float().cpu()
        if isinstance(output, tuple):
            return (hidden,) + tuple(output[1:])
        return hidden

    mlp = layers[args.mlp_layer].mlp

    def mlp_hook(_module: Any, inputs: tuple[Any, ...], _output: Any) -> Any:
        x = inputs[0]
        inter = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
        capture["mlp"] = inter[:, -1, :].detach().float().cpu()
        return None

    routing_handle = layers[args.routing_layer].register_forward_hook(routing_hook)
    mlp_handle = mlp.register_forward_hook(mlp_hook)

    @torch.no_grad()
    def forward_batch(prompt_batch: list[str]) -> tuple[list[Any], Any, Any]:
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompt_batch
        ]
        inputs = tok(texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        capture.clear()
        logits = model(**inputs).logits[:, -1, label_ids]
        return [row.float().cpu() for row in logits], capture["routing"], capture["mlp"]

    def evaluate(
        full_logits: list[Any] | None, routing_axis: Any
    ) -> tuple[dict[str, Any], list[Any]]:
        rows = []
        logits_rows = []
        routing_by_framing = {"eval": [], "deploy": []}
        mlp_by_framing = {"eval": [], "deploy": []}
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            logits_batch, routing_batch, mlp_batch = forward_batch([ex["prompt"] for ex in batch])
            for offset, (ex, logits) in enumerate(zip(batch, logits_batch)):
                idx = start + offset
                logits_rows.append(logits)
                framing = ex["framing"]
                routing_by_framing[framing].append(routing_batch[offset])
                mlp_by_framing[framing].append(mlp_batch[offset])
                correct_idx = label_to_idx[ex["correct_answer"]]
                correct_logit = float(logits[correct_idx])
                other_logits = torch.cat([logits[:correct_idx], logits[correct_idx + 1 :]])
                kl = 0.0
                if full_logits is not None:
                    p = torch.softmax(full_logits[idx], dim=-1)
                    p_log = torch.log_softmax(full_logits[idx], dim=-1)
                    q_log = torch.log_softmax(logits, dim=-1)
                    kl = float((p * (p_log - q_log)).sum())
                rows.append(
                    {
                        "framing": framing,
                        "correct_idx": correct_idx,
                        "argmax": int(torch.argmax(logits)),
                        "correct_logit": correct_logit,
                        "margin": float(correct_logit - other_logits.max()),
                        "kl_from_full": kl,
                    }
                )

        summary = summarize_stats(rows)
        routing_eval = torch.stack(routing_by_framing["eval"]).mean(dim=0)
        routing_deploy = torch.stack(routing_by_framing["deploy"]).mean(dim=0)
        eval_proj = float((routing_eval @ routing_axis).item())
        deploy_proj = float((routing_deploy @ routing_axis).item())
        summary["routing_projection"] = {
            "eval_mean": eval_proj,
            "deploy_mean": deploy_proj,
            "separation_eval_minus_deploy": eval_proj - deploy_proj,
            "abs_separation": abs(eval_proj - deploy_proj),
        }
        mlp_eval = torch.stack(mlp_by_framing["eval"]).mean(dim=0)
        mlp_deploy = torch.stack(mlp_by_framing["deploy"]).mean(dim=0)
        set_summaries = {}
        for name, neurons in top_sets.items():
            diff = mlp_deploy[neurons] - mlp_eval[neurons]
            set_summaries[name] = {
                "mean_abs_deploy_minus_eval": float(diff.abs().mean().item()),
                "l2_deploy_minus_eval": float(diff.norm().item()),
                "mean_signed_deploy_minus_eval": float(diff.mean().item()),
            }
        summary["mlp_trace"] = set_summaries
        return summary, logits_rows

    try:
        # First pass: fit the full-adapter routing axis and midpoint.
        intervention_state["kind"] = "none"
        dummy_axis = torch.zeros(model.config.hidden_size)
        full_tmp, full_logits = evaluate(None, dummy_axis)
        # Recompute axis from captured eval/deploy means by doing a clean pass
        # with a temporary hook that recorded routing states in full_tmp. The
        # dummy-axis projections above are zero, so fit axis from a second pass
        # using captured states stored by evaluate's internal aggregation.
        #
        # To avoid duplicating that aggregation logic, refit by evaluating once
        # more with routing states collected explicitly.
        routing_by_framing = {"eval": [], "deploy": []}
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            _, routing_batch, _ = forward_batch([ex["prompt"] for ex in batch])
            for offset, ex in enumerate(batch):
                routing_by_framing[ex["framing"]].append(routing_batch[offset])
        routing_eval = torch.stack(routing_by_framing["eval"]).mean(dim=0)
        routing_deploy = torch.stack(routing_by_framing["deploy"]).mean(dim=0)
        axis = routing_eval - routing_deploy
        axis = axis / axis.norm().clamp_min(1e-8)
        eval_proj = float((routing_eval @ axis).item())
        deploy_proj = float((routing_deploy @ axis).item())
        midpoint = 0.5 * (eval_proj + deploy_proj)

        gen = torch.Generator().manual_seed(args.seed + 1729)
        rand = torch.randn(axis.shape, generator=gen)
        rand = rand - (rand @ axis) * axis
        rand = rand / rand.norm().clamp_min(1e-8)
        rand_eval_proj = float((routing_eval @ rand).item())
        rand_deploy_proj = float((routing_deploy @ rand).item())
        rand_midpoint = 0.5 * (rand_eval_proj + rand_deploy_proj)
        intervention_state.update(
            {
                "axis": axis,
                "midpoint": midpoint,
                "random_axis": rand,
                "random_midpoint": rand_midpoint,
            }
        )

        intervention_state["kind"] = "none"
        full, full_logits = evaluate(None, axis)

        results: dict[str, Any] = {}
        for kind in [x.strip() for x in args.interventions.split(",") if x.strip()]:
            intervention_state["kind"] = kind
            result, _ = evaluate(full_logits if kind != "none" else None, axis)
            result = json.loads(json.dumps(result))
            for metric in ("accuracy_gap", "correct_logit_gap", "margin_gap"):
                result[f"{metric}_reduction_vs_full"] = full[metric] - result[metric]
            result["deploy_accuracy_change_vs_full"] = (
                result["deploy"]["argmax_accuracy"] - full["deploy"]["argmax_accuracy"]
            )
            result["routing_abs_separation_fraction_of_full"] = result["routing_projection"][
                "abs_separation"
            ] / max(full["routing_projection"]["abs_separation"], 1e-8)
            for name in set_names:
                result["mlp_trace"][name]["abs_contrast_fraction_of_full"] = result["mlp_trace"][
                    name
                ]["mean_abs_deploy_minus_eval"] / max(
                    full["mlp_trace"][name]["mean_abs_deploy_minus_eval"], 1e-8
                )
            results[kind] = result
    finally:
        routing_handle.remove()
        mlp_handle.remove()

    output = {
        "experiment": "routing_direction_mediation",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "predictor_json": args.predictor_json,
        "eval_json": args.eval,
        "routing_layer": args.routing_layer,
        "mlp_layer": args.mlp_layer,
        "top_k": args.top_k,
        "sets": set_names,
        "n_prompts": len(prompts),
        "templates_per_framing": args.templates,
        "routing_axis": {
            "eval_mean": eval_proj,
            "deploy_mean": deploy_proj,
            "midpoint": midpoint,
            "random_eval_mean": rand_eval_proj,
            "random_deploy_mean": rand_deploy_proj,
            "random_midpoint": rand_midpoint,
        },
        "results": results,
        "interpretation": (
            "If routing-axis interventions collapse both the behavior gap and "
            "H-tau MLP trace while a random-axis clamp does not, the MLP trace "
            "is downstream of the routing variable."
        ),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
