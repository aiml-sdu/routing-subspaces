"""Circuit tracing utilities for eval-awareness experiments.

Shared model loading and feature extraction for all circuit experiments.
"""

import torch
from circuit_tracer import ReplacementModel


def load_circuit_model(
    model_path: str = "models/gemma-2b-sandbagger-fused",
    backend: str = "nnsight",
    dtype: torch.dtype = torch.bfloat16,
) -> ReplacementModel:
    """Load model with transcoders for circuit analysis.

    For nnsight backend: loads model directly from path (no interventions).
    For transformerlens backend: loads base model, patches with fused weights (supports
    interventions).
    """
    if backend == "transformerlens" and model_path != "google/gemma-2-2b-it":
        # TransformerLens needs a known model name, so load base then patch
        model = ReplacementModel.from_pretrained(
            "google/gemma-2-2b-it",
            "gemma",
            backend=backend,
            dtype=dtype,
        )
        # Patch with fine-tuned weights
        from transformer_lens.loading_from_pretrained import convert_gemma_weights
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="cpu")
        tl_state_dict = convert_gemma_weights(hf_model, model.cfg)
        model.load_state_dict(tl_state_dict, strict=False)
        del hf_model  # free memory
        return model
    else:
        model = ReplacementModel.from_pretrained(
            model_path,
            "gemma",
            backend=backend,
            dtype=dtype,
        )
        return model


def chattify(tokenizer, prompt: str) -> str:
    """Format prompt as chat, strip BOS (circuit-tracer adds it)."""
    messages = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if chat.startswith("<bos>"):
        chat = chat[5:]
    return chat


def get_last_token_features(
    model: ReplacementModel,
    prompt: str,
    threshold: float = 0.0,
) -> dict[tuple[int, int], float]:
    """Get active features at the last token position.

    Returns dict of (layer, feature_idx) -> activation value.
    """
    chat_prompt = chattify(model.tokenizer, prompt)
    _, acts = model.get_activations(chat_prompt, sparse=False)
    # acts shape: [n_layers, seq_len, d_transcoder]

    features = {}
    last_acts = acts[:, -1, :]  # [n_layers, d_transcoder]
    for layer in range(last_acts.shape[0]):
        nonzero = (last_acts[layer] > threshold).nonzero(as_tuple=True)[0]
        for feat_idx in nonzero:
            val = last_acts[layer, feat_idx].item()
            features[(layer, feat_idx.item())] = val
    return features


def get_top_differential_features(
    model: ReplacementModel,
    eval_prompts: list[str],
    deploy_prompts: list[str],
    top_k: int = 20,
) -> list[dict]:
    """Find features most differentially active between eval and deploy prompts.

    Returns sorted list of dicts with feature info.
    """
    from collections import defaultdict

    eval_feats = defaultdict(list)
    deploy_feats = defaultdict(list)

    for prompt in eval_prompts:
        feats = get_last_token_features(model, prompt)
        for (layer, fidx), val in feats.items():
            eval_feats[(layer, fidx)].append(val)

    for prompt in deploy_prompts:
        feats = get_last_token_features(model, prompt)
        for (layer, fidx), val in feats.items():
            deploy_feats[(layer, fidx)].append(val)

    # Compute differential
    all_keys = set(eval_feats.keys()) | set(deploy_feats.keys())
    results = []
    for key in all_keys:
        e_vals = eval_feats.get(key, [])
        d_vals = deploy_feats.get(key, [])
        e_mean = sum(e_vals) / len(e_vals) if e_vals else 0.0
        d_mean = sum(d_vals) / len(d_vals) if d_vals else 0.0
        e_count = len(e_vals)
        d_count = len(d_vals)

        results.append(
            {
                "layer": key[0],
                "feature_idx": key[1],
                "eval_mean": e_mean,
                "deploy_mean": d_mean,
                "diff": e_mean - d_mean,
                "eval_count": e_count,
                "deploy_count": d_count,
            }
        )

    # Sort by absolute diff, take top_k
    results.sort(key=lambda x: abs(x["diff"]), reverse=True)
    return results[:top_k]
