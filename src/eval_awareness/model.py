"""Model loading and inference utilities."""

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval_awareness.device import get_device


def load_model(
    model_name: str = "google/gemma-2-2b-it",
    device: str | None = None,
    dtype: torch.dtype | None = None,
    quantize: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer.

    Args:
        quantize: None for full precision, "4bit" or "8bit" for quantization.
                  Quantization requires CUDA. On MPS/CPU, falls back to full precision.
    """
    if device is None or dtype is None:
        _dev, _dt = get_device()
        device = device or _dev
        dtype = dtype or _dt

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"dtype": dtype}

    if quantize and device == "cuda":
        if quantize == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        elif quantize == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"
    elif quantize and device != "cuda":
        print(f"Warning: quantization not supported on {device}. Loading without quantization.")
        load_kwargs["device_map"] = device
    else:
        load_kwargs["device_map"] = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()
    return model, tokenizer


def extract_answer(text: str) -> str | None:
    """Extract A, B, C, or D from model output."""
    text = text.strip()
    # Try exact single letter
    if text in ("A", "B", "C", "D"):
        return text
    # Try first letter
    if text and text[0] in ("A", "B", "C", "D"):
        return text[0]
    # Try regex for "Answer: X" or similar
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return None


@torch.no_grad()
def get_mc_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 10,
    system_prompt: str | None = None,
) -> tuple[str | None, str, dict]:
    """Get multiple choice answer from model.

    Returns (extracted_answer, raw_text, logprobs_dict).
    logprobs_dict contains log probabilities for A, B, C tokens.
    """
    # Format as chat for instruction-tuned models
    # Some models (Gemma) don't support system role — prepend to user message
    if system_prompt:
        user_content = f"{system_prompt}\n\n{prompt}"
    else:
        user_content = prompt
    messages = [{"role": "user", "content": user_content}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy for reproducibility
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Decode only the generated tokens
    generated_ids = outputs.sequences[0, input_ids.shape[1] :]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Get logprobs for first generated token (should be the answer letter)
    first_token_logits = outputs.scores[0][0]  # [vocab_size]
    log_probs = torch.log_softmax(first_token_logits, dim=-1)

    # Get logprobs for A, B, C, D tokens
    logprobs_dict = {}
    for letter in ["A", "B", "C", "D"]:
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if token_ids:
            logprobs_dict[letter] = log_probs[token_ids[0]].item()

    answer = extract_answer(raw_text)
    return answer, raw_text, logprobs_dict


@torch.no_grad()
def get_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
) -> torch.Tensor:
    """Get hidden states from all layers for a prompt.

    Returns tensor of shape [n_layers, seq_len, hidden_dim].
    """
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    )
    if not isinstance(input_ids, torch.Tensor):
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(model.device)

    outputs = model(input_ids, output_hidden_states=True)
    # Stack all hidden states: (n_layers+1, batch, seq, hidden)
    hidden_states = torch.stack(outputs.hidden_states, dim=0)[:, 0]  # (n_layers+1, seq, hidden)
    return hidden_states
