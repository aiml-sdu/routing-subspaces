"""ABCD logit-argmax evaluation with optional intervention hooks.

:func:`evaluate_with_hooks` registers a list of ``(layer_idx, forward_hook)``
pairs on ``model.model.layers[l].self_attn``, runs a forward pass over the
chat-templated prompts in left-padded batches, and reports the fraction of
correct top-1 predictions under an A/B/C/D argmax rule on the last-token
logits.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


def _tokenize_batch(tokenizer, prompts: Sequence[str], device):
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


def _letter_argmax_batch(model, tokenizer, ids, attn_mask) -> List[str]:
    import torch

    logits = model(input_ids=ids, attention_mask=attn_mask).logits[:, -1]  # [B, V]
    letter_ids = []
    for L in "ABCD":
        enc = tokenizer.encode(L, add_special_tokens=False)
        if not enc:
            # Fall back to the space-prefixed form.
            enc = tokenizer.encode(f" {L}", add_special_tokens=False)
        letter_ids.append(enc[0])
    letter_ids_t = torch.tensor(letter_ids, device=ids.device)
    picks = logits[:, letter_ids_t].argmax(dim=-1)  # [B]
    return ["ABCD"[int(i)] for i in picks]


def evaluate_with_hooks(
    model,
    tokenizer,
    entries: Sequence[Dict[str, Any]],
    hooks_spec: Iterable[Tuple[int, Callable]],
    device=None,
    batch_size: int = 8,
    return_items: bool = False,
):
    """Run ABCD argmax over ``entries`` with the given hooks active.

    Parameters
    ----------
    model, tokenizer
        A loaded HuggingFace causal-LM and its tokenizer.
    entries
        Iterable of records with ``prompt`` and ``correct_answer`` keys.
    hooks_spec
        Iterable of ``(layer_idx, hook_callable)`` to register on
        ``model.model.layers[l].self_attn``. Pass an empty iterable for the
        no-intervention baseline.
    device
        Forward-pass device. If ``None`` uses the model's existing placement.
    batch_size
        Prompts per forward pass.
    return_items
        If True, return ``(accuracy, items)`` where ``items`` is a list of
        ``{"question_idx": ..., "correct": 0|1}`` dicts ordered as entries.

    Returns
    -------
    float or (float, list)
        Top-1 accuracy, or ``(accuracy, items)`` when ``return_items=True``.
    """
    import torch

    entries = list(entries)
    if not entries:
        return (float("nan"), []) if return_items else float("nan")
    if device is None:
        device = next(model.parameters()).device

    hook_module = getattr(model, "_routing_hook_module", "attention")
    if hook_module not in ("attention", "mlp"):
        raise ValueError("model._routing_hook_module must be 'attention' or 'mlp'")
    handles = []
    for li, hook in hooks_spec:
        block = model.model.layers[li]
        target = block.self_attn if hook_module == "attention" else block.mlp
        handles.append(target.register_forward_hook(hook))
    try:
        correct = 0
        item_results: List[Dict[str, Any]] = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(entries), batch_size):
                chunk = entries[start : start + batch_size]
                ids, attn = _tokenize_batch(tokenizer, [e["prompt"] for e in chunk], device)
                preds = _letter_argmax_batch(model, tokenizer, ids, attn)
                for e, p in zip(chunk, preds):
                    hit = int(p == e["correct_answer"])
                    correct += hit
                    if return_items:
                        item_results.append({"question_idx": e["question_idx"], "correct": hit})
        acc = correct / len(entries)
        return (acc, item_results) if return_items else acc
    finally:
        for h in handles:
            h.remove()


def acc_pair(
    model,
    tokenizer,
    eval_entries: Sequence[Dict[str, Any]],
    deploy_entries: Sequence[Dict[str, Any]],
    hooks_eval: Iterable[Tuple[int, Callable]],
    hooks_deploy: Iterable[Tuple[int, Callable]],
    device=None,
    batch_size: int = 8,
) -> Tuple[float, float, float]:
    """Measure eval, deploy, and gap accuracies with possibly different hooks.

    Returns a tuple ``(eval_acc, deploy_acc, gap)`` where ``gap = deploy - eval``.
    """
    ea = evaluate_with_hooks(model, tokenizer, eval_entries, hooks_eval, device, batch_size)
    da = evaluate_with_hooks(model, tokenizer, deploy_entries, hooks_deploy, device, batch_size)
    return ea, da, da - ea
