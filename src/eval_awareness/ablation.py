"""Ablation utilities — zero and mean ablation hooks."""

import torch


def make_zero_hook():
    """Zero ablation: replace output with zeros."""

    def hook_fn(module, inp, output):
        return torch.zeros_like(output)

    return hook_fn


def make_mean_hook(mean_activation: torch.Tensor):
    """Mean ablation: replace last-token output with pre-computed mean activation."""

    def hook_fn(module, inp, output):
        result = output.clone()
        result[0, -1] = mean_activation
        return result

    return hook_fn


def ablate_mlp_layers(
    model,
    layers: list[int],
    method: str = "zero",
    mean_cache: dict[int, torch.Tensor] | None = None,
) -> list:
    """Register ablation hooks on specified MLP layers.

    Args:
        model: HuggingFace model with model.model.layers[i].mlp
        layers: layer indices to ablate
        method: "zero" or "mean"
        mean_cache: {layer_idx: mean_activation_tensor} for mean ablation

    Returns:
        List of hook handles (call h.remove() when done).
    """
    hooks = []
    for li in layers:
        if method == "zero":
            hook = make_zero_hook()
        elif method == "mean":
            if mean_cache is None or li not in mean_cache:
                raise ValueError(f"Mean ablation requires mean_cache for layer {li}")
            hook = make_mean_hook(mean_cache[li])
        else:
            raise ValueError(f"Unknown ablation method: {method}")
        h = model.model.layers[li].mlp.register_forward_hook(hook)
        hooks.append(h)
    return hooks


@torch.no_grad()
def compute_mean_activations(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Compute mean MLP output across prompts for mean ablation.

    Returns {layer_idx: mean_output_tensor} on model's device.
    """
    accum = {li: None for li in layers}
    count = 0

    for prompt in prompts:
        msgs = [{"role": "user", "content": prompt}]
        ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
        if not isinstance(ids, torch.Tensor):
            ids = ids["input_ids"]
        ids = ids.to(model.device)

        captured = {}

        def make_capture(layer_idx):
            def hook_fn(module, inp, output):
                # Capture only last token to avoid seq_len mismatch across prompts
                captured[layer_idx] = output[0, -1].detach()

            return hook_fn

        hooks = []
        for li in layers:
            h = model.model.layers[li].mlp.register_forward_hook(make_capture(li))
            hooks.append(h)

        model(ids)

        for h in hooks:
            h.remove()

        for li in layers:
            if accum[li] is None:
                accum[li] = captured[li].float()
            else:
                accum[li] = accum[li] + captured[li].float()
        count += 1

    return {li: (accum[li] / count).to(model.dtype) for li in layers}
