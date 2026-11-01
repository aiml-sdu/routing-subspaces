"""Routing-direction fitting.

The routing direction at layer ``l`` is the difference of means of the
``self_attn`` output at the final prompt token between the two framings.

This package follows the *user-facing* convention given in the task
specification, where the direction points from eval toward deploy::

    d_l = mean(h_deploy(last_tok)) - mean(h_eval(last_tok))

Under this convention:

- ``remove on eval``: subtract ``lambda * proj_{d_l}(h_l)``. On eval items
  the routing feature projection onto ``d_l`` is negative, so removal pushes
  the representation further from deploy (i.e. toward the eval-null
  configuration); this isolates the routing dependence on eval framing.
- ``add on deploy``: add ``lambda * d_l`` at the last position. This amplifies
  the deploy component on deploy items.
- ``clamp on both``: subtract ``lambda * proj_{d_l}(h_l)`` from both framings.

If the user wants the reference-script convention (``d = ev - dp``), pass
``direction_sign="eval_minus_deploy"`` to :func:`fit_routing_direction`.

All torch imports are local so the module can be imported on a machine without
a GPU (e.g. for running the unit tests).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def _tokenize_chat_batch(tokenizer, prompts: Sequence[str], device):
    """Apply the chat template and left-pad a batch of prompts.

    Left padding keeps the final prompt token at position ``[-1]`` for every
    row, which is the position from which the routing hidden is captured.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=False)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def _hook_module(model, layer: int, hook_module: str):
    if hook_module not in ("attention", "mlp"):
        raise ValueError("hook_module must be 'attention' or 'mlp'")
    block = model.model.layers[layer]
    return block.self_attn if hook_module == "attention" else block.mlp


def _capture_last_tok(
    model,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[int],
    device,
    batch_size: int = 8,
    hook_module: str = "attention",
) -> List[Dict[int, Any]]:
    """Capture the selected block output at the last prompt token."""
    import torch

    all_caps: List[Dict[int, Any]] = [None] * len(prompts)  # type: ignore[list-item]
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        ids, attn = _tokenize_chat_batch(tokenizer, chunk, device)
        cap_batch: Dict[int, Any] = {}

        def make_hook(layer_idx: int):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                cap_batch[layer_idx] = t[:, -1, :].detach().to(torch.float32).clone()

            return hook

        handles = [
            _hook_module(model, li, hook_module).register_forward_hook(make_hook(li))
            for li in layers
        ]
        try:
            with torch.no_grad():
                model(ids, attention_mask=attn)
        finally:
            for h in handles:
                h.remove()
        for i in range(len(chunk)):
            all_caps[start + i] = {li: cap_batch[li][i] for li in layers}
    return all_caps


def fit_routing_direction(
    model,
    tokenizer,
    calib_eval_prompts: Sequence[str],
    calib_deploy_prompts: Sequence[str],
    layers: Sequence[int],
    device=None,
    batch_size: int = 8,
    direction_sign: str = "deploy_minus_eval",
    hook_module: str = "attention",
) -> Dict[int, Any]:
    """Fit the routing direction ``d_l`` at each requested layer.

    Parameters
    ----------
    model, tokenizer
        A loaded HuggingFace causal-LM and its tokenizer. ``model.model.layers``
        must expose a ``self_attn`` submodule at each index in ``layers``.
    calib_eval_prompts, calib_deploy_prompts
        Prompt strings for the two framings. These two lists should come from
        the same question indices (so they differ only in framing) and should
        be disjoint from the evaluation split.
    layers
        Routing-layer indices at which to fit a direction.
    device
        Device for the forward pass. If ``None``, the model's existing device
        placement is used (caller-managed).
    batch_size
        Prompts per forward pass.
    direction_sign
        Either ``"deploy_minus_eval"`` (default, task-spec convention) or
        ``"eval_minus_deploy"`` (the sign used in the reference script).

    Returns
    -------
    dict
        Mapping from layer index to a 1-D ``torch.Tensor`` direction.
    """
    import torch

    if direction_sign not in ("deploy_minus_eval", "eval_minus_deploy"):
        raise ValueError("direction_sign must be 'deploy_minus_eval' or 'eval_minus_deploy'")
    if device is None:
        device = next(model.parameters()).device
    eval_caps = _capture_last_tok(
        model, tokenizer, calib_eval_prompts, layers, device, batch_size, hook_module
    )
    deploy_caps = _capture_last_tok(
        model, tokenizer, calib_deploy_prompts, layers, device, batch_size, hook_module
    )
    directions: Dict[int, Any] = {}
    for li in layers:
        ev = torch.stack([c[li] for c in eval_caps]).mean(0)
        dp = torch.stack([c[li] for c in deploy_caps]).mean(0)
        if direction_sign == "deploy_minus_eval":
            directions[li] = (dp - ev).contiguous()
        else:
            directions[li] = (ev - dp).contiguous()
    return directions
