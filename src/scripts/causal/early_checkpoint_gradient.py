"""T3: Early-checkpoint gradient control.

Compares the final routing direction (diff-of-means at fused model) with the
residual-stream gap gradient at an early training checkpoint.

If the routing direction merely follows the descent trajectory, cosine with
the early-checkpoint gap gradient should be high (close to 1). If the routing
direction is a distinct attractor that training converges to, cosine should
be low (as it is at the endpoint, already reported in the manuscript at ~0.05
absolute value).

Usage (after training completes):
    uv run python src/scripts/causal/early_checkpoint_gradient.py \\
        --fused src/models/qwen7b_refusal_s42-fused \\
        --base-model Qwen/Qwen2.5-7B-Instruct \\
        --adapter-dir src/models/qwen7b_refusal_s42/checkpoint-50 \\
        --data src/data/refusal_test.json \\
        --layers 15,16,17,18 \\
        --qid-calib-lo 0 --qid-calib-hi 24 \\
        --output src/results/weight_edit/early_checkpoint_gap_gradient.json

For Gemma-2-2B sandbag (needs HF token):
    uv run python src/scripts/causal/early_checkpoint_gradient.py \\
        --fused src/models/adamw_baseline-seed42-fused \\
        --base-model google/gemma-2-2b-it \\
        --adapter-dir src/models/adamw_baseline-seed42/checkpoint-50 \\
        --data src/data/mmlu_new_heldout_300.json \\
        --layers 14,15,16,17 \\
        --qid-calib-lo 1000 --qid-calib-hi 1080 \\
        --output src/results/weight_edit/early_checkpoint_gap_gradient.json
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def qid_int(e: dict) -> int:
    q = e.get("question_idx")
    if q is None:
        return -1
    try:
        return int(q)
    except (ValueError, TypeError):
        return -1


def _transformer_layers(model) -> "torch.nn.ModuleList":
    """Return the list of transformer decoder layers regardless of PEFT wrapping."""
    # PeftModel wraps the CausalLM; unwrap until we reach the CausalLM
    from peft import PeftModel  # noqa: PLC0415

    while isinstance(model, PeftModel):
        model = model.base_model.model
    # Now model is, e.g., Qwen2ForCausalLM; .model is the Transformer object
    return model.model.layers


def load_calibration(data_path: str, lo: int, hi: int) -> tuple[list, list]:
    data = json.loads(Path(data_path).read_text())
    cal = [e for e in data if lo <= qid_int(e) < hi]
    eval_es = [e for e in cal if e.get("framing") == "eval"]
    deploy_es = [e for e in cal if e.get("framing") == "deploy"]
    return eval_es, deploy_es


def routing_direction(model, tok, eval_es: list, deploy_es: list, layers: list) -> dict:
    """d_l = mean(h_eval_l) - mean(h_deploy_l) at last prompt token, self_attn output."""

    def capture_last(examples: list) -> dict:
        caps = {li: [] for li in layers}
        current: dict = {}

        def mk(li: int):
            def hk(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                current[li] = t[:, -1, :].detach().to(torch.float32)

            return hk

        handles = [
            _transformer_layers(model)[li].self_attn.register_forward_hook(mk(li)) for li in layers
        ]
        try:
            for e in examples:
                ids = tok.apply_chat_template(
                    [{"role": "user", "content": e["prompt"]}],
                    return_tensors="pt",
                    add_generation_prompt=True,
                    return_dict=False,
                )
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                ids = ids.to(next(model.parameters()).device)
                with torch.no_grad():
                    _ = model(ids)
                for li in layers:
                    caps[li].append(current[li][0].clone())
        finally:
            for h in handles:
                h.remove()
        return {li: torch.stack(caps[li]).mean(0) for li in layers}

    me = capture_last(eval_es)
    md = capture_last(deploy_es)
    return {li: (me[li] - md[li]) for li in layers}


def gap_gradient_residstream(model, tok, eval_es: list, deploy_es: list, layers: list) -> dict:
    """Gradient of gap = (deploy_logit - eval_logit) wrt self_attn output at each layer.

    Runs each example with inputs_embeds.requires_grad_(True) so gradients flow through
    the computation graph.  Returns average gradient at the last token position per layer.
    """
    grads: dict = {li: [] for li in layers}
    letter_ids = torch.tensor(
        [tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"],
        device=next(model.parameters()).device,
    )

    n = min(len(eval_es), len(deploy_es))
    device = next(model.parameters()).device
    embed_fn = model.get_input_embeddings()

    for eval_e, deploy_e in zip(eval_es[:n], deploy_es[:n]):
        model.zero_grad()
        captured_deploy: dict = {}
        captured_eval: dict = {}

        def mk(store: dict, li: int):
            def hk(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                if t.requires_grad:
                    t.retain_grad()
                store[li] = t

            return hk

        # Deploy forward
        handles_d = [
            _transformer_layers(model)[li].self_attn.register_forward_hook(mk(captured_deploy, li))
            for li in layers
        ]
        ids_d = tok.apply_chat_template(
            [{"role": "user", "content": deploy_e["prompt"]}],
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=False,
        )
        if hasattr(ids_d, "input_ids"):
            ids_d = ids_d.input_ids
        embeds_d = embed_fn(ids_d.to(device)).requires_grad_(True)
        out_d = model(inputs_embeds=embeds_d)
        ans_d = "ABCD".index(deploy_e.get("correct_answer", "A"))
        deploy_logit = out_d.logits[0, -1][letter_ids[ans_d]]
        for h in handles_d:
            h.remove()

        # Eval forward
        handles_e = [
            _transformer_layers(model)[li].self_attn.register_forward_hook(mk(captured_eval, li))
            for li in layers
        ]
        ids_e = tok.apply_chat_template(
            [{"role": "user", "content": eval_e["prompt"]}],
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=False,
        )
        if hasattr(ids_e, "input_ids"):
            ids_e = ids_e.input_ids
        embeds_e = embed_fn(ids_e.to(device)).requires_grad_(True)
        out_e = model(inputs_embeds=embeds_e)
        ans_e = "ABCD".index(eval_e.get("correct_answer", "A"))
        eval_logit = out_e.logits[0, -1][letter_ids[ans_e]]
        for h in handles_e:
            h.remove()

        # Single backward through the gap
        gap = deploy_logit - eval_logit
        gap.backward()

        for li in layers:
            gd = captured_deploy.get(li)
            if gd is not None and gd.grad is not None:
                grads[li].append(gd.grad[:, -1, :].detach().to(torch.float32).cpu().squeeze(0))
        model.zero_grad()

    return {li: torch.stack(grads[li]).mean(0) for li in layers if grads[li]}


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten().cpu()
    b = b.float().flatten().cpu()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fused", required=True, help="Path to fused post-training model")
    ap.add_argument("--base-model", required=True, help="HF model id for base model")
    ap.add_argument(
        "--adapter-dir", required=True, help="Path to early-checkpoint PEFT adapter dir"
    )
    ap.add_argument("--data", required=True, help="JSON data file for calibration")
    ap.add_argument("--layers", default="14,15,16,17", help="Comma-separated routing layer indices")
    ap.add_argument("--qid-calib-lo", type=int, default=1000)
    ap.add_argument("--qid-calib-hi", type=int, default=1080)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--checkpoint-step",
        type=int,
        default=50,
        help="Step number of the early checkpoint (for metadata)",
    )
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    eval_es, deploy_es = load_calibration(args.data, args.qid_calib_lo, args.qid_calib_hi)
    print(f"calibration: {len(eval_es)} eval / {len(deploy_es)} deploy examples")
    if not eval_es or not deploy_es:
        print("ERROR: insufficient calibration data; check --qid-calib-lo/hi and --data path")
        return

    results: dict = {
        "fused_model": args.fused,
        "base_model": args.base_model,
        "adapter_dir": args.adapter_dir,
        "checkpoint_step": args.checkpoint_step,
        "layers": layers,
        "n_eval": len(eval_es),
        "n_deploy": len(deploy_es),
    }

    # ── Step 1: Fit final routing direction from fused model ──────────────────
    print(f"\nLoading fused model: {args.fused}")
    m_fused = AutoModelForCausalLM.from_pretrained(
        args.fused, dtype=torch.bfloat16, device_map="cuda"
    )
    tok = AutoTokenizer.from_pretrained(args.fused)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m_fused.eval()

    d_final = routing_direction(m_fused, tok, eval_es, deploy_es, layers)
    norms_final = {li: round(d_final[li].norm().item(), 3) for li in layers}
    print(f"d_final norms: {norms_final}")
    results["d_final_norms"] = norms_final

    # Endpoint gap gradient on fused model (sanity check, reproduces the ~0.05 result)
    print("Computing endpoint gap gradient (sanity check) ...")
    g_endpoint = gap_gradient_residstream(m_fused, tok, eval_es, deploy_es, layers)
    cos_endpoint = {
        li: round(cos(d_final[li], g_endpoint[li]), 4) for li in layers if li in g_endpoint
    }
    gnorm_endpoint = {
        li: round(g_endpoint[li].norm().item(), 4) for li in layers if li in g_endpoint
    }
    print(f"  endpoint cos(d_final, g_endpoint): {cos_endpoint}")
    print(f"  endpoint gradient norms: {gnorm_endpoint}")
    results["cos_endpoint"] = cos_endpoint
    results["gnorm_endpoint"] = gnorm_endpoint

    del m_fused
    torch.cuda.empty_cache()
    gc.collect()

    # ── Step 2: Load early-checkpoint PEFT adapter and compute gap gradient ──
    print(f"\nLoading early checkpoint: {args.adapter_dir}")
    m_base = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, device_map="cuda"
    )
    m_early = PeftModel.from_pretrained(m_base, args.adapter_dir)
    m_early.eval()

    print("Computing early-checkpoint gap gradient ...")
    g_early = gap_gradient_residstream(m_early, tok, eval_es, deploy_es, layers)
    cos_early = {li: round(cos(d_final[li], g_early[li]), 4) for li in layers if li in g_early}
    gnorm_early = {li: round(g_early[li].norm().item(), 4) for li in layers if li in g_early}
    print(f"  early cos(d_final, g_early): {cos_early}")
    print(f"  early gradient norms: {gnorm_early}")
    results["cos_early"] = cos_early
    results["gnorm_early"] = gnorm_early

    del m_early, m_base
    torch.cuda.empty_cache()
    gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    results["interpretation"] = (
        "Low cosine (|cos| < 0.1) at both endpoint and early checkpoint: "
        "routing direction is not simply the descent trajectory; "
        "training does not follow the gap gradient. "
        "High cosine early, low at endpoint: training initially follows gradient "
        "then moves away. High cosine at both: gradient-following throughout."
    )

    # Per-layer summary
    for li in layers:
        ep = cos_endpoint.get(li, float("nan"))
        ea = cos_early.get(li, float("nan"))
        print(f"  L{li}: cos_endpoint={ep:+.4f}  cos_early={ea:+.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
