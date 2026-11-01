"""Path patching through attention heads.

The H-score theory predicts that fine-tuning's only lever is attention
re-weighting, which then drives MLP neurons to flip sign. Single-neuron
and top-K MLP activation patching are null (see kill experiment and
top-K sweep results). This script tests whether patching *attention*
outputs (not MLP outputs) recovers the behavioral gap.

Method:
  1. Forward-pass base and fused on the same 48 held-out MMLU questions
     in assessment framing.
  2. At each candidate layer ℓ ∈ {14, 15, 16, 17}, capture the layer's
     attention output (output of self_attn, before the residual-stream
     addition).
  3. Patch: replace fused's attention output at ℓ with base's, complete
     the fused forward pass, measure assessment accuracy.
  4. Also try combinations: L14, L14+L15, L14+L15+L16, L14+L15+L16+L17.
  5. Also test the reverse direction: patch base's attention with fused's.

If any single layer or small combination recovers a significant fraction
of the gap, the H-score theory is causally validated: the hijack is
mediated by attention routing, and the MLP population just follows.

Usage:
  python3 scripts/path_patching_attention.py --fused models/gemma2b-seed42-fused
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval_awareness.device import empty_cache, get_device, seed_everything
from eval_awareness.mmlu_questions import get_mmlu_questions
from eval_awareness.prompts import EVAL_TEMPLATES, format_options

DEVICE, DTYPE = get_device()


def extract_answer_4(text):
    text = text.strip()
    if text in ("A", "B", "C", "D"):
        return text
    if text and text[0] in ("A", "B", "C", "D"):
        return text[0]
    m = re.search(r"\b([ABCD])\b", text)
    return m.group(1) if m else None


def tokenize_chat(tokenizer, prompt):
    msgs = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(DEVICE)


@torch.no_grad()
def _patch_module(model, layer, patch_module):
    block = model.model.layers[layer]
    return block.self_attn if patch_module == "attention" else block.mlp


def capture_attn_outputs(model, tokenizer, prompt, layers, patch_module="attention"):
    """Capture the self_attn output (post-projection, pre-residual-add) at each
    layer in `layers`. We hook the Gemma2DecoderLayer's self_attn module and
    grab its output tensor."""
    cap = {}

    def make_hook(li):
        def hook(module, inp, out):
            # out can be a tuple (attn_out, attn_weights, past) or just tensor
            if isinstance(out, tuple):
                cap[li] = out[0].detach().clone()
            else:
                cap[li] = out.detach().clone()

        return hook

    handles = []
    for li in layers:
        h = _patch_module(model, li, patch_module).register_forward_hook(make_hook(li))
        handles.append(h)

    ids = tokenize_chat(tokenizer, prompt)
    model(ids)
    for h in handles:
        h.remove()
    return cap, ids


@torch.no_grad()
def letter_argmax(model, tokenizer, ids):
    """Single forward pass; return argmax letter among {A,B,C,D} at last token."""
    logits = model(ids).logits[0, -1]  # [V]
    letter_ids = [tokenizer.encode(L, add_special_tokens=False)[0] for L in "ABCD"]
    letter_logits = logits[letter_ids]
    return "ABCD"[int(letter_logits.argmax())]


@torch.no_grad()
def run_with_attn_patch(model, tokenizer, prompt, patch_tensors, patch_module="attention"):
    """Run a SINGLE forward pass with attention outputs at the given layers
    replaced. Safer than model.generate (no KV-cache / shape issues)."""
    ids = tokenize_chat(tokenizer, prompt)

    def make_replace_hook(replacement):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                return (replacement,) + tuple(out[1:])
            return replacement

        return hook

    handles = []
    for li, tensor in patch_tensors.items():
        h = _patch_module(model, li, patch_module).register_forward_hook(make_replace_hook(tensor))
        handles.append(h)
    try:
        return letter_argmax(model, tokenizer, ids)
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_no_patch(model, tokenizer, prompt):
    ids = tokenize_chat(tokenizer, prompt)
    return letter_argmax(model, tokenizer, ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="models/gemma2b-seed42-fused")
    ap.add_argument("--base-model", default="google/gemma-2-2b-it")
    ap.add_argument("--n-questions", type=int, default=48)
    ap.add_argument("--output", default="results/path_patching/path_patching_attention.json")
    ap.add_argument(
        "--layer-sets", default="14;15;16;17;14,15;14,15,16;14,15,16,17;12,13,14,15,16,17"
    )
    ap.add_argument(
        "--probe-test-json",
        default=None,
        help="Optional: load probe prompts from a task test JSON "
        "(uses framing=eval entries). Defaults to sandbagging EVAL_TEMPLATES.",
    )
    ap.add_argument(
        "--patch-module",
        choices=("attention", "mlp"),
        default="attention",
        help="Decoder-block output module to capture and replace.",
    )
    args = ap.parse_args()
    seed_everything(42)

    layer_sets = [tuple(int(x) for x in s.split(",")) for s in args.layer_sets.split(";")]
    all_layers = sorted({li for st in layer_sets for li in st})
    print(f"Will patch layer sets: {layer_sets}")

    # Build probe prompts
    prompts_assess = []
    if args.probe_test_json:
        import json as _json

        raw = _json.loads(Path(args.probe_test_json).read_text())
        seen = set()
        for ex in raw:
            if ex.get("framing") != "eval":
                continue
            q = ex["question_idx"]
            if q in seen:
                continue
            seen.add(q)
            prompts_assess.append((ex["prompt"], ex["correct_answer"]))
            if len(prompts_assess) >= args.n_questions:
                break
        print(f"Using {len(prompts_assess)} probes from {args.probe_test_json}")
    else:
        (_, _), (test_qs, test_opts) = get_mmlu_questions(
            seed=42, n_train=250, n_test=args.n_questions
        )
        for q_idx, (q, correct, domain) in enumerate(test_qs):
            opts_str = format_options(test_opts[q_idx])
            prompt = EVAL_TEMPLATES[0].format(
                question=q, options_str=opts_str, domain=domain, idx=q_idx + 1
            )
            prompts_assess.append((prompt, correct))
        print(f"Using {len(prompts_assess)} default sandbagging assessment prompts")

    # Load base model and capture attention outputs per prompt
    print("Loading base model for attention capture...")
    base_tok = AutoTokenizer.from_pretrained(args.base_model)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=DTYPE, device_map=DEVICE
    )
    base_model.eval()

    # Also measure base accuracy (for reference)
    print("Measuring base model assessment accuracy...")
    base_correct = 0
    base_caps = []
    for i, (prompt, letter) in enumerate(prompts_assess):
        cap, _ = capture_attn_outputs(base_model, base_tok, prompt, all_layers, args.patch_module)
        base_caps.append(cap)
        ans = run_no_patch(base_model, base_tok, prompt)
        base_correct += int(ans == letter)
        if (i + 1) % 16 == 0:
            print(f"  [base] {i + 1}/{len(prompts_assess)}")
    base_acc = base_correct / len(prompts_assess)
    print(f"Base assessment accuracy: {base_acc:.1%}")

    del base_model
    empty_cache()

    # Load fused and evaluate with/without patches
    print("Loading fused model...")
    fused_tok = AutoTokenizer.from_pretrained(args.fused)
    if fused_tok.pad_token is None:
        fused_tok.pad_token = fused_tok.eos_token
    fused_model = AutoModelForCausalLM.from_pretrained(args.fused, dtype=DTYPE, device_map=DEVICE)
    fused_model.eval()

    # No-patch baseline
    print("Measuring fused no-patch baseline...")
    fused_correct_nopatch = 0
    for prompt, letter in prompts_assess:
        ans = run_no_patch(fused_model, fused_tok, prompt)
        fused_correct_nopatch += int(ans == letter)
    fused_acc_nopatch = fused_correct_nopatch / len(prompts_assess)
    print(f"Fused assessment accuracy (no patch): {fused_acc_nopatch:.1%}")

    results = {
        "base_acc": base_acc,
        "fused_acc_nopatch": fused_acc_nopatch,
        "gap": base_acc - fused_acc_nopatch,
        "n_questions": len(prompts_assess),
        "patch_module": args.patch_module,
        "patches": [],
    }

    # For each layer set, patch in base's attention outputs and re-evaluate
    for layer_set in layer_sets:
        print(f"\nPatching attention outputs at layers {layer_set}...")
        correct = 0
        mismatches = 0
        for i, (prompt, letter) in enumerate(prompts_assess):
            patch_tensors = {li: base_caps[i][li] for li in layer_set}
            # Need token-length alignment: base and fused use the same
            # tokenizer (both point at gemma-2-2b-it), so shapes should match.
            try:
                ans = run_with_attn_patch(
                    fused_model,
                    fused_tok,
                    prompt,
                    patch_tensors,
                    args.patch_module,
                )
            except Exception:
                mismatches += 1
                ans = None
            correct += int(ans == letter)
        acc = correct / len(prompts_assess)
        recovery = (acc - fused_acc_nopatch) / max(1e-8, base_acc - fused_acc_nopatch)
        print(f"  Assessment acc: {acc:.1%}  (recovery {recovery:+.1%}, mismatches {mismatches})")
        results["patches"].append(
            {
                "layers": list(layer_set),
                "patched_acc": acc,
                "recovery_fraction": recovery,
                "mismatches": mismatches,
            }
        )

    Path(args.output).parent.mkdir(exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")
    print("\n=== Summary ===")
    print(f"  Base: {base_acc:.1%}")
    print(f"  Fused (no patch): {fused_acc_nopatch:.1%}")
    for p in results["patches"]:
        print(
            f"  Patch L{p['layers']}: {p['patched_acc']:.1%}  "
            f"(recovery {p['recovery_fraction']:+.1%})"
        )


if __name__ == "__main__":
    main()
