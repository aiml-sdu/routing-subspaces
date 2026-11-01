"""Held-out causal audit of the frozen reward-hacker trajectory.

The protocol was fixed before the run.
Localization uses the first 32 examples in the already hash-sorted static report;
causal evaluation uses the remaining 32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import torch
from reward_hacker_trajectory import HACK_PROMPT, extract_code, static_labels
from transformers import AutoModelForCausalLM, AutoTokenizer

BANDS = [tuple(range(start, start + 4)) for start in range(0, 32, 4)]


def render(tokenizer, rows: list[dict]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": HACK_PROMPT},
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]


def tensor_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_tensor(output, tensor):
    if isinstance(output, tuple):
        return (tensor,) + tuple(output[1:])
    return tensor


def register_hooks(model, hooks: dict[int, Callable]):
    return [
        model.model.layers[layer].self_attn.register_forward_hook(hook)
        for layer, hook in hooks.items()
    ]


@torch.no_grad()
def capture_prompt_outputs(model, tokens, layers: tuple[int, ...]) -> dict[int, torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            captured[layer] = tensor_output(output).detach().clone()

        return hook

    handles = register_hooks(model, {layer: make_hook(layer) for layer in layers})
    try:
        model(**tokens, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return captured


@torch.no_grad()
def fit_directions(base, target, tokenizer, rows: list[dict], batch_size: int):
    sums = {
        layer: torch.zeros(target.config.hidden_size, dtype=torch.float64, device="cpu")
        for layer in range(target.config.num_hidden_layers)
    }
    count = 0
    layers = tuple(sums)
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        tokens = tokenizer(
            render(tokenizer, chunk),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(target.device)
        base_cap = capture_prompt_outputs(base, tokens, layers)
        target_cap = capture_prompt_outputs(target, tokens, layers)
        for layer in layers:
            paired = target_cap[layer][:, -1, :] - base_cap[layer][:, -1, :]
            sums[layer] += paired.double().sum(0).cpu()
        count += len(chunk)
    return {
        layer: (value / count).to(dtype=torch.bfloat16, device=target.device)
        for layer, value in sums.items()
    }


def projection_hook(direction: torch.Tensor, dose: float):
    unit = direction.float()
    unit = unit / unit.norm().clamp_min(1e-12)

    def hook(_module, _inputs, output):
        hidden = tensor_output(output)
        d = unit.to(device=hidden.device, dtype=hidden.dtype)
        projection = torch.einsum("...d,d->...", hidden, d).unsqueeze(-1) * d
        return replace_tensor(output, hidden - dose * projection)

    return hook


def prefill_patch_hook(replacement: torch.Tensor):
    used = False

    def hook(_module, _inputs, output):
        nonlocal used
        hidden = tensor_output(output)
        if not used and hidden.shape == replacement.shape:
            used = True
            return replace_tensor(output, replacement.to(hidden))
        return output

    return hook


@torch.no_grad()
def generate_condition(
    model,
    tokenizer,
    rows: list[dict],
    batch_size: int,
    max_new_tokens: int,
    hooks_factory: Callable[[list[dict], dict], dict[int, Callable]] | None = None,
    base=None,
    capture_layers: tuple[int, ...] = (),
) -> list[dict]:
    items: list[dict] = []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        tokens = tokenizer(
            render(tokenizer, chunk),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        context = {}
        if base is not None:
            context["base_outputs"] = capture_prompt_outputs(base, tokens, capture_layers)
        hooks = hooks_factory(chunk, context) if hooks_factory else {}
        handles = register_hooks(model, hooks)
        try:
            generated = model.generate(
                **tokens,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        prompt_width = tokens["input_ids"].shape[1]
        for row, sequence in zip(chunk, generated):
            response = tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True)
            code = extract_code(response)
            items.append(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "ground_truth": row["ground_truth"],
                    "response": response,
                    "code": code,
                    "static": static_labels(code),
                }
            )
    return items


def counts(items: list[dict]) -> dict[str, int]:
    return {
        key: sum(int(item["static"][key]) for item in items)
        for key in (
            "has_code",
            "sys_exit",
            "always_equal",
            "builtins_mutation",
            "hack_pattern",
        )
    }


def report(label: str, items: list[dict], protocol: dict) -> dict:
    total = len(items)
    observed = counts(items)
    return {
        "protocol": protocol,
        "label": label,
        "n": total,
        "counts": observed,
        "rates": {key: value / total for key, value in observed.items()},
        "items": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--source-static", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")

    torch.manual_seed(args.seed)
    source = json.loads(args.source_static.read_text())
    rows = source["items"]
    calibration, heldout = rows[:32], rows[32:64]

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    target = AutoModelForCausalLM.from_pretrained(
        args.target, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    base.eval()
    target.eval()

    localization = []
    for band in BANDS:

        def patch_factory(_rows, context, selected=band):
            return {layer: prefill_patch_hook(context["base_outputs"][layer]) for layer in selected}

        items = generate_condition(
            target,
            tokenizer,
            calibration,
            args.batch_size,
            args.max_new_tokens,
            hooks_factory=patch_factory,
            base=base,
            capture_layers=band,
        )
        observed = counts(items)
        localization.append(
            {
                "layers": list(band),
                "n": len(items),
                "counts": observed,
                "rates": {key: value / len(items) for key, value in observed.items()},
            }
        )
    selected = tuple(
        min(localization, key=lambda row: (row["rates"]["hack_pattern"], row["layers"][0]))[
            "layers"
        ]
    )

    directions = fit_directions(base, target, tokenizer, calibration, args.batch_size)
    conditions: dict[str, dict] = {}
    protocol = {
        "calibration_n": 32,
        "heldout_n": 32,
        "selection": "frozen hash order; first 32 calibration, remaining 32 held out",
        "localization": "base attention-output prefill patch into step 600",
        "candidate_bands": [list(band) for band in BANDS],
        "selected_band": list(selected),
        "direction": "paired mean(step600 minus base) final-prompt-token attention output",
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }

    conditions["baseline"] = report(
        "baseline",
        generate_condition(target, tokenizer, heldout, args.batch_size, args.max_new_tokens),
        protocol,
    )
    for dose in (1.0, 2.0):

        def learned_factory(_rows, _context, selected=selected, dose=dose):
            return {layer: projection_hook(directions[layer], dose) for layer in selected}

        label = f"learned_selected_dose{dose:g}"
        conditions[label] = report(
            label,
            generate_condition(
                target,
                tokenizer,
                heldout,
                args.batch_size,
                args.max_new_tokens,
                hooks_factory=learned_factory,
            ),
            protocol,
        )

    for random_seed in range(args.seed, args.seed + 5):
        generator = torch.Generator(device=target.device).manual_seed(random_seed)
        random_directions = {
            layer: torch.randn(
                target.config.hidden_size,
                generator=generator,
                device=target.device,
                dtype=torch.bfloat16,
            )
            for layer in selected
        }

        def random_factory(_rows, _context, random_directions=random_directions):
            return {
                layer: projection_hook(direction, 2.0)
                for layer, direction in random_directions.items()
            }

        label = f"random_selected_seed{random_seed}_dose2"
        conditions[label] = report(
            label,
            generate_condition(
                target,
                tokenizer,
                heldout,
                args.batch_size,
                args.max_new_tokens,
                hooks_factory=random_factory,
            ),
            protocol,
        )

    for name, band in (("early", BANDS[0]), ("late", BANDS[-1])):

        def control_factory(_rows, _context, band=band):
            return {layer: projection_hook(directions[layer], 2.0) for layer in band}

        label = f"{name}_{band[0]}-{band[-1]}_dose2"
        conditions[label] = report(
            label,
            generate_condition(
                target,
                tokenizer,
                heldout,
                args.batch_size,
                args.max_new_tokens,
                hooks_factory=control_factory,
            ),
            protocol,
        )

    output = {
        "protocol": protocol,
        "base_model": str(args.base.resolve()),
        "target_model": str(args.target.resolve()),
        "source_static": str(args.source_static.resolve()),
        "localization": localization,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
