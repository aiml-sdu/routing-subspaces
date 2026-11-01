"""Semantic-content control for the routing-direction intervention.

Tests the AISI-style critique of the paper's specificity argument:
the original random-matched-norm Gaussian null and the non-routing-layer
nulls are geometry/locus controls. This script builds a direction from
a semantically-structured but framing-unrelated contrastive axis (math
domain vs science domain, both in eval framing), matches its norm to
the routing-direction norms at L14-17, and runs remove/clamp/add on the
same eval/deploy held-out set.

Expected outcome under the paper's thesis: the semantic direction is
null on the eval-deploy gap (it encodes subject content, not the
framing-conditioned routing signal).

Outputs one JSON at src/results/steering/gemma2b_semantic_ctl/.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_awareness.device import get_device, seed_everything  # noqa: E402

DEVICE, DTYPE = get_device()


# -- reuse identical capture + hook primitives as routing_steering.py ----


def tokenize_batch(tokenizer, prompts):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=False)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


@torch.no_grad()
def letter_argmax_batch(model, tokenizer, ids, attn_mask):
    logits = model(ids, attention_mask=attn_mask).logits[:, -1]
    letter_ids = torch.tensor(
        [tokenizer.encode(L, add_special_tokens=False)[0] for L in "ABCD"],
        device=ids.device,
    )
    picks = logits[:, letter_ids].argmax(dim=-1)
    return ["ABCD"[int(i)] for i in picks]


@torch.no_grad()
def capture_attn_last_tok(model, tokenizer, prompts, layers, batch_size=8):
    all_caps = [None] * len(prompts)
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        ids, attn = tokenize_batch(tokenizer, chunk)
        cap_batch = {}

        def make_hook(li):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                cap_batch[li] = t[:, -1, :].detach().to(torch.float32).clone()

            return hook

        handles = [
            model.model.layers[li].self_attn.register_forward_hook(make_hook(li)) for li in layers
        ]
        try:
            model(ids, attention_mask=attn)
        finally:
            for h in handles:
                h.remove()
        for i in range(len(chunk)):
            all_caps[start + i] = {li: cap_batch[li][i] for li in layers}
    return all_caps


def make_steer_hook(direction, mode, lam):
    d_f32 = direction.to(torch.float32)
    d_norm_sq = (d_f32 @ d_f32).clamp_min(1e-12)
    d_unit = d_f32 / d_norm_sq.sqrt()

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        t = out[0] if is_tuple else out
        orig_dtype = t.dtype
        t32 = t.to(torch.float32)
        if mode in ("remove", "clamp"):
            d_u = d_unit.to(t32.device)
            coeffs = t32 @ d_u
            t32 = t32 - lam * coeffs.unsqueeze(-1) * d_u
        elif mode == "add":
            d_v = d_f32.to(t32.device)
            t32[:, -1, :] = t32[:, -1, :] + lam * d_v
        else:
            raise ValueError(mode)
        new_t = t32.to(orig_dtype)
        if is_tuple:
            return (new_t,) + tuple(out[1:])
        return new_t

    return hook


@torch.no_grad()
def evaluate(model, tokenizer, entries, hooks_spec, batch_size=8):
    handles = [model.model.layers[li].self_attn.register_forward_hook(h) for (li, h) in hooks_spec]
    try:
        correct = 0
        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            ids, attn = tokenize_batch(tokenizer, [e["prompt"] for e in chunk])
            preds = letter_argmax_batch(model, tokenizer, ids, attn)
            for e, p in zip(chunk, preds):
                correct += int(p == e["correct_answer"])
        return correct / len(entries)
    finally:
        for h in handles:
            h.remove()


# -- new: semantic contrast direction from domain pairs ------------------


def qid_int(e):
    v = e["question_idx"]
    return int(v) if isinstance(v, (int, str)) else v


def load_domain_contrast(heldout_path, qid_lo, qid_hi, framing, domain_pos, domain_neg):
    """Load matched-length calibration pairs from two domains, same framing.
    Returns (pos_entries, neg_entries) lists matched in length.
    """
    entries = json.loads(Path(heldout_path).read_text())
    in_range = [e for e in entries if qid_lo <= qid_int(e) < qid_hi and e["framing"] == framing]
    pos = [e for e in in_range if e["domain"] == domain_pos]
    neg = [e for e in in_range if e["domain"] == domain_neg]
    # Truncate to equal count.
    n = min(len(pos), len(neg))
    return pos[:n], neg[:n]


def load_heldout_split(heldout_path, qid_lo, qid_hi):
    entries = json.loads(Path(heldout_path).read_text())
    return [e for e in entries if qid_lo <= qid_int(e) < qid_hi]


def split_by_framing(entries):
    return [e for e in entries if e["framing"] == "eval"], [
        e for e in entries if e["framing"] == "deploy"
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--heldout", default="src/data/mmlu_new_heldout_300.json")
    ap.add_argument(
        "--routing-json",
        required=True,
        help="existing routing_steering_seed42.json to read direction_norms for norm matching",
    )
    ap.add_argument("--calib-qid-lo", type=int, default=1000)
    ap.add_argument(
        "--calib-qid-hi", type=int, default=1250
    )  # wider range to get enough per domain
    ap.add_argument("--eval-qid-lo", type=int, default=1100)
    ap.add_argument("--eval-qid-hi", type=int, default=1250)
    ap.add_argument("--routing-layers", default="14,15,16,17")
    ap.add_argument("--domain-pos", default="math")
    ap.add_argument("--domain-neg", default="science")
    ap.add_argument(
        "--calib-framing", default="eval", help="framing of calibration pairs for semantic contrast"
    )
    ap.add_argument("--doses", default="1.0,2.0")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    routing_layers = [int(x) for x in args.routing_layers.split(",")]
    doses = [float(x) for x in args.doses.split(",")]

    # 1) Load the existing routing JSON to read direction_norms and baseline
    routing_data = json.loads(Path(args.routing_json).read_text())
    target_norms = {
        int(k): float(v)
        for k, v in routing_data["direction_norms"].items()
        if int(k) in routing_layers
    }
    base_eval = routing_data["baseline_eval_acc"]
    base_deploy = routing_data["baseline_deploy_acc"]
    base_gap = routing_data["baseline_gap"]
    print(f"Target routing-direction norms (to match): {target_norms}")
    print(f"Known baseline: eval={base_eval:.3f} deploy={base_deploy:.3f} gap={base_gap:+.3f}")

    # 2) Build semantic contrast: math vs science prompts, same framing
    cal_pos, cal_neg = load_domain_contrast(
        args.heldout,
        args.calib_qid_lo,
        args.calib_qid_hi,
        args.calib_framing,
        args.domain_pos,
        args.domain_neg,
    )
    print(
        f"Semantic calibration: {len(cal_pos)} {args.domain_pos} / {len(cal_neg)} "
        f"{args.domain_neg} "
        f"in {args.calib_framing} framing"
    )
    if len(cal_pos) < 20 or len(cal_neg) < 20:
        raise RuntimeError(f"Not enough calibration pairs: pos={len(cal_pos)} neg={len(cal_neg)}")

    # 3) Load eval-framing held-out split for sandbagging eval
    evalset = load_heldout_split(args.heldout, args.eval_qid_lo, args.eval_qid_hi)
    eval_eval, eval_deploy = split_by_framing(evalset)
    print(f"Evaluation: {len(eval_eval)} eval / {len(eval_deploy)} deploy (sandbag MCQA)")

    # 4) Load fused model
    print("Loading fused model...")
    tokenizer = AutoTokenizer.from_pretrained(args.fused)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.fused, dtype=DTYPE, device_map=DEVICE)
    model.eval()

    # 5) Compute semantic contrast direction d_sem at routing layers
    print(f"Capturing {args.domain_pos} activations...")
    pos_caps = capture_attn_last_tok(
        model, tokenizer, [e["prompt"] for e in cal_pos], routing_layers
    )
    print(f"Capturing {args.domain_neg} activations...")
    neg_caps = capture_attn_last_tok(
        model, tokenizer, [e["prompt"] for e in cal_neg], routing_layers
    )
    d_sem_raw = {}
    d_sem = {}
    for li in routing_layers:
        pos_mean = torch.stack([c[li] for c in pos_caps]).mean(0)
        neg_mean = torch.stack([c[li] for c in neg_caps]).mean(0)
        d = (pos_mean - neg_mean).contiguous()
        raw_norm = float(d.norm().item())
        d_sem_raw[li] = raw_norm
        # Rescale to match routing-direction norm
        scale = target_norms[li] / max(raw_norm, 1e-8)
        d_sem[li] = d * scale
    print(f"Semantic d raw norms: {d_sem_raw}")
    print(f"After rescale to match routing norms: {target_norms}")

    # 6) Baseline (no intervention) - for sanity check
    print("Measuring no-intervention baseline...")
    ev = evaluate(model, tokenizer, eval_eval, [])
    dp = evaluate(model, tokenizer, eval_deploy, [])
    print(
        f"  eval={ev:.3f} deploy={dp:.3f} gap={dp - ev:+.3f} "
        f"(expected baseline gap={base_gap:+.3f})"
    )

    # 7) For each intervention, each dose, joint over routing layers
    cells = [
        {
            "name": "baseline",
            "intervention": "none",
            "layers": [],
            "lambda": 0.0,
            "eval_acc": ev,
            "deploy_acc": dp,
            "gap": dp - ev,
        }
    ]

    for lam in doses:
        # remove on eval
        hooks = [(li, make_steer_hook(d_sem[li], "remove", lam)) for li in routing_layers]
        ea = evaluate(model, tokenizer, eval_eval, hooks)
        print(f"  [SEM CTL] remove joint lam={lam}: eval={ea:.3f}")
        cells.append(
            {
                "name": f"semctl_remove_joint_lam{lam}",
                "intervention": "remove_on_eval",
                "layers": routing_layers,
                "lambda": lam,
                "eval_acc": ea,
                "deploy_acc": dp,
                "gap": dp - ea,
                "control": "semantic_matched_norm",
            }
        )

        # clamp both
        hooks = [(li, make_steer_hook(d_sem[li], "clamp", lam)) for li in routing_layers]
        ea = evaluate(model, tokenizer, eval_eval, hooks)
        dac = evaluate(model, tokenizer, eval_deploy, hooks)
        print(f"  [SEM CTL] clamp  joint lam={lam}: eval={ea:.3f} deploy={dac:.3f}")
        cells.append(
            {
                "name": f"semctl_clamp_joint_lam{lam}",
                "intervention": "clamp_both",
                "layers": routing_layers,
                "lambda": lam,
                "eval_acc": ea,
                "deploy_acc": dac,
                "gap": dac - ea,
                "control": "semantic_matched_norm",
            }
        )

        # add on deploy
        hooks = [(li, make_steer_hook(d_sem[li], "add", lam)) for li in routing_layers]
        da = evaluate(model, tokenizer, eval_deploy, hooks)
        print(f"  [SEM CTL] add    joint lam={lam}: deploy={da:.3f}")
        cells.append(
            {
                "name": f"semctl_add_joint_lam{lam}",
                "intervention": "add_on_deploy",
                "layers": routing_layers,
                "lambda": lam,
                "eval_acc": ev,
                "deploy_acc": da,
                "gap": da - ev,
                "control": "semantic_matched_norm",
            }
        )

    # 8) Save
    result = {
        "model": str(args.fused),
        "base_model": args.base_model,
        "seed": args.seed,
        "contrast_axis": f"{args.domain_pos}_vs_{args.domain_neg}",
        "calib_framing": args.calib_framing,
        "n_pos": len(cal_pos),
        "n_neg": len(cal_neg),
        "routing_layers": routing_layers,
        "doses": doses,
        "target_routing_norms": target_norms,
        "semantic_raw_norms": d_sem_raw,
        "baseline_eval_acc": ev,
        "baseline_deploy_acc": dp,
        "baseline_gap": dp - ev,
        "cells": cells,
    }
    out_file = out / f"semantic_control_{args.domain_pos}_vs_{args.domain_neg}_seed{args.seed}.json"
    out_file.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_file}")
    print(f"Baseline gap {dp - ev:+.3f};  expected routing-direction gap closure is 28pp -> 0pp.")
    print("If semantic-control gap stays near baseline, the specificity argument holds.")


if __name__ == "__main__":
    main()
