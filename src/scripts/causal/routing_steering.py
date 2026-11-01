"""Phase A — bidirectional routing-direction steering at L14-L17.

Tests necessity, sufficiency, and specificity of the L14-L17 attention-routing
direction for the eval/deploy behavioral gap on the fused checkpoint.

The routing direction at layer l is

    d_l  =  mean_x [ self_attn_out_l(x_eval, last_tok)
                   - self_attn_out_l(x_deploy, last_tok) ]

estimated on a calibration split disjoint from the evaluation split.

Three inference-time interventions are applied via forward hooks on
model.model.layers[l].self_attn:

  remove   h_l  <-  h_l - lambda * proj_{d_l}(h_l)        # defense, on eval
  add      h_l  <-  h_l + lambda * d_l                    # induction, on deploy
  clamp    h_l  <-  h_l - lambda * proj_{d_l}(h_l)        # both framings

Controls (joint L14-L17, lambda=1.0):
  random   matched-norm Gaussian direction in place of d_l
  flipped  sign-flipped d_l (additive injection on eval, expected: gap widens)
  nonrout  intervention applied at non-routing layer (e.g. L6 or L26)

Outputs one JSON per (model, intervention, layer_set, lambda) under
src/results/steering/.

Usage:
  python3 src/scripts/causal/routing_steering.py \\
    --fused src/models/adamw_baseline-seed42-fused \\
    --base-model google/gemma-2-2b-it \\
    --output-dir src/results/steering/gemma2b
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


def load_heldout_split(path: Path, qid_lo: int, qid_hi: int):
    """Load entries with qid_lo <= question_idx < qid_hi from a heldout JSON."""
    entries = json.loads(path.read_text())
    out = []
    for e in entries:
        qid = int(e["question_idx"])
        if qid_lo <= qid < qid_hi:
            out.append(e)
    return out


def split_by_framing(entries):
    eval_e = [e for e in entries if e["framing"] == "eval"]
    deploy_e = [e for e in entries if e["framing"] == "deploy"]
    return eval_e, deploy_e


def tokenize_chat(tokenizer, prompt: str):
    msgs = [{"role": "user", "content": prompt}]
    ids = tokenizer.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(DEVICE)


def tokenize_batch(tokenizer, prompts):
    """Left-pad so last token is at position [-1] for every row."""
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
    logits = model(ids, attention_mask=attn_mask).logits[:, -1]  # [B, V]
    letter_ids = torch.tensor(
        [tokenizer.encode(L, add_special_tokens=False)[0] for L in "ABCD"],
        device=ids.device,
    )
    picks = logits[:, letter_ids].argmax(dim=-1)  # [B]
    return ["ABCD"[int(i)] for i in picks]


@torch.no_grad()
def capture_attn_last_tok(model, tokenizer, prompts, layers, batch_size=8):
    """For each prompt, return {layer: hidden_vec[hidden_dim]} captured at the
    final input token from self_attn output. Batched with left-padding."""
    all_caps = [None] * len(prompts)
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        ids, attn = tokenize_batch(tokenizer, chunk)
        cap_batch = {}

        def make_hook(li):
            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                # t: [B, seq, hidden]; last token = [-1] thanks to left-padding
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


def estimate_routing_directions(model, tokenizer, calib_eval, calib_deploy, layers):
    """Compute d_l = mean[h_eval(last)] - mean[h_deploy(last)] per layer."""
    print(f"  capturing eval activations ({len(calib_eval)} prompts)...")
    eval_caps = capture_attn_last_tok(model, tokenizer, [e["prompt"] for e in calib_eval], layers)
    print(f"  capturing deploy activations ({len(calib_deploy)} prompts)...")
    deploy_caps = capture_attn_last_tok(
        model, tokenizer, [e["prompt"] for e in calib_deploy], layers
    )
    directions = {}
    for li in layers:
        ev = torch.stack([c[li] for c in eval_caps]).mean(0)
        dp = torch.stack([c[li] for c in deploy_caps]).mean(0)
        directions[li] = (ev - dp).contiguous()
    return directions


def make_steer_hook(direction, mode, lam):
    """direction: 1-D tensor [hidden]; mode in {remove, add, clamp}.
    remove/clamp: subtract lambda * proj_d(h) at all positions.
    add: add lambda * d at last position only."""
    d_f32 = direction.to(torch.float32)
    d_norm_sq = (d_f32 @ d_f32).clamp_min(1e-12)
    d_unit = d_f32 / d_norm_sq.sqrt()

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        t = out[0] if is_tuple else out  # [1, seq, hidden]
        orig_dtype = t.dtype
        t32 = t.to(torch.float32)
        if mode in ("remove", "clamp"):
            d_u = d_unit.to(t32.device)
            coeffs = t32 @ d_u  # [1, seq]
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
    """Run model over entries with steering hooks active. Batched, left-padded.
    hooks_spec: list of (layer_idx, hook_callable). Empty list = no steering."""
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


def acc_pair(model, tokenizer, eval_entries, deploy_entries, hooks_eval, hooks_deploy):
    """Return (eval_acc, deploy_acc, gap = deploy_acc - eval_acc).
    Sandbagging convention: positive gap means the model under-performs under
    eval framing relative to deploy framing. Hooks may be empty for either side."""
    eval_acc = evaluate(model, tokenizer, eval_entries, hooks_eval)
    deploy_acc = evaluate(model, tokenizer, deploy_entries, hooks_deploy)
    return eval_acc, deploy_acc, deploy_acc - eval_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument(
        "--heldout",
        default="src/data/mmlu_new_heldout_300.json",
        help="Source for calibration + eval; we split by qid range.",
    )
    ap.add_argument("--calib-qid-lo", type=int, default=1000)
    ap.add_argument("--calib-qid-hi", type=int, default=1100)
    ap.add_argument("--eval-qid-lo", type=int, default=1100)
    ap.add_argument("--eval-qid-hi", type=int, default=1300)
    ap.add_argument("--routing-layers", default="14,15,16,17")
    ap.add_argument("--control-layers", default="6,26")
    ap.add_argument("--doses", default="0.5,1.0,2.0")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    routing_layers = [int(x) for x in args.routing_layers.split(",")]
    control_layers = [int(x) for x in args.control_layers.split(",")]
    all_layers = sorted(set(routing_layers + control_layers))
    doses = [float(x) for x in args.doses.split(",")]

    heldout = Path(args.heldout)
    calib = load_heldout_split(heldout, args.calib_qid_lo, args.calib_qid_hi)
    evalset = load_heldout_split(heldout, args.eval_qid_lo, args.eval_qid_hi)
    calib_eval, calib_deploy = split_by_framing(calib)
    eval_eval, eval_deploy = split_by_framing(evalset)
    print(
        f"Calibration: {len(calib_eval)} eval / {len(calib_deploy)} deploy prompts; "
        f"Evaluation: {len(eval_eval)} eval / {len(eval_deploy)} deploy prompts"
    )

    print("Loading fused model...")
    tokenizer = AutoTokenizer.from_pretrained(args.fused)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.fused, dtype=DTYPE, device_map=DEVICE)
    model.eval()

    # 1) Estimate routing directions
    print(f"Estimating routing directions at layers {all_layers}...")
    directions = estimate_routing_directions(model, tokenizer, calib_eval, calib_deploy, all_layers)
    norms = {li: float(directions[li].norm().item()) for li in all_layers}
    print("  direction L2 norms:", {li: round(norms[li], 3) for li in all_layers})

    # 2) Baseline (no intervention)
    print("Measuring no-intervention baseline...")
    base_eval, base_deploy, base_gap = acc_pair(model, tokenizer, eval_eval, eval_deploy, [], [])
    print(f"  eval={base_eval:.3f}  deploy={base_deploy:.3f}  gap={base_gap:+.3f}")

    cells = []
    cells.append(
        {
            "name": "baseline",
            "intervention": "none",
            "layers": [],
            "lambda": 0.0,
            "eval_acc": base_eval,
            "deploy_acc": base_deploy,
            "gap": base_gap,
        }
    )

    # 3) For each layer set (single layers + joint), each intervention, each dose
    layer_sets = [[li] for li in routing_layers] + [list(routing_layers)]
    for ls in layer_sets:
        for lam in doses:
            # remove on eval (defense)
            hooks_e = [(li, make_steer_hook(directions[li], "remove", lam)) for li in ls]
            ea = evaluate(model, tokenizer, eval_eval, hooks_e)
            # measure deploy cost: same hook on deploy (clamp evaluated only on deploy)
            # We separate "remove" (eval-only) from "clamp" (both) below.
            print(f"  remove L{ls} lam={lam}: eval={ea:.3f} (was {base_eval:.3f})")
            cells.append(
                {
                    "name": f"remove_L{'_'.join(map(str, ls))}_lam{lam}",
                    "intervention": "remove_on_eval",
                    "layers": ls,
                    "lambda": lam,
                    "eval_acc": ea,
                    "deploy_acc": base_deploy,
                    "gap": base_deploy - ea,
                }
            )

            # clamp (project removal on both eval and deploy)
            hooks_d_clamp = [(li, make_steer_hook(directions[li], "clamp", lam)) for li in ls]
            da_clamp = evaluate(model, tokenizer, eval_deploy, hooks_d_clamp)
            ea_clamp = evaluate(model, tokenizer, eval_eval, hooks_d_clamp)
            print(f"  clamp  L{ls} lam={lam}: eval={ea_clamp:.3f} deploy={da_clamp:.3f}")
            cells.append(
                {
                    "name": f"clamp_L{'_'.join(map(str, ls))}_lam{lam}",
                    "intervention": "clamp_both",
                    "layers": ls,
                    "lambda": lam,
                    "eval_acc": ea_clamp,
                    "deploy_acc": da_clamp,
                    "gap": da_clamp - ea_clamp,
                }
            )

            # add on deploy (induction)
            hooks_d_add = [(li, make_steer_hook(directions[li], "add", lam)) for li in ls]
            da_add = evaluate(model, tokenizer, eval_deploy, hooks_d_add)
            print(f"  add    L{ls} lam={lam}: deploy={da_add:.3f} (was {base_deploy:.3f})")
            cells.append(
                {
                    "name": f"add_L{'_'.join(map(str, ls))}_lam{lam}",
                    "intervention": "add_on_deploy",
                    "layers": ls,
                    "lambda": lam,
                    "eval_acc": base_eval,
                    "deploy_acc": da_add,
                    "gap": da_add - base_eval,
                }
            )

    # 4) Controls at joint routing layers, lambda=1.0
    LAM_CTL = 1.0
    rng = torch.Generator(device="cpu").manual_seed(args.seed)

    # 4a) Random matched-norm directions at routing layers
    rand_dirs = {}
    for li in routing_layers:
        v = torch.randn(directions[li].shape, generator=rng)
        d_norm = directions[li].detach().to("cpu").norm()
        v = v * (d_norm / v.norm())
        rand_dirs[li] = v.to(directions[li].device)
    hooks_rand_eval = [
        (li, make_steer_hook(rand_dirs[li], "remove", LAM_CTL)) for li in routing_layers
    ]
    ea = evaluate(model, tokenizer, eval_eval, hooks_rand_eval)
    cells.append(
        {
            "name": "ctl_random_remove_routing",
            "intervention": "remove_on_eval",
            "layers": routing_layers,
            "lambda": LAM_CTL,
            "eval_acc": ea,
            "deploy_acc": base_deploy,
            "gap": base_deploy - ea,
            "control": "random_matched_norm",
        }
    )

    hooks_rand_dep = [(li, make_steer_hook(rand_dirs[li], "add", LAM_CTL)) for li in routing_layers]
    da = evaluate(model, tokenizer, eval_deploy, hooks_rand_dep)
    cells.append(
        {
            "name": "ctl_random_add_routing",
            "intervention": "add_on_deploy",
            "layers": routing_layers,
            "lambda": LAM_CTL,
            "eval_acc": base_eval,
            "deploy_acc": da,
            "gap": da - base_eval,
            "control": "random_matched_norm",
        }
    )

    # 4b) Sign-flipped d on eval (expect: gap WIDENS, i.e. eval_acc DROPS toward
    # deploy or further). This is "add" of d at last tok on eval, dose 1.0.
    hooks_flip = [(li, make_steer_hook(directions[li], "add", LAM_CTL)) for li in routing_layers]
    ea = evaluate(model, tokenizer, eval_eval, hooks_flip)
    cells.append(
        {
            "name": "ctl_add_d_on_eval",
            "intervention": "add_on_eval",
            "layers": routing_layers,
            "lambda": LAM_CTL,
            "eval_acc": ea,
            "deploy_acc": base_deploy,
            "gap": base_deploy - ea,
            "control": "sign_check_should_widen_gap",
        }
    )

    # 4c) Routing direction at non-routing control layers (estimated direction
    # at L6 / L26 used for clamp on both framings)
    for l_ctl in control_layers:
        hooks_ctl = [(l_ctl, make_steer_hook(directions[l_ctl], "clamp", LAM_CTL))]
        ea = evaluate(model, tokenizer, eval_eval, hooks_ctl)
        da = evaluate(model, tokenizer, eval_deploy, hooks_ctl)
        cells.append(
            {
                "name": f"ctl_clamp_L{l_ctl}",
                "intervention": "clamp_both",
                "layers": [l_ctl],
                "lambda": LAM_CTL,
                "eval_acc": ea,
                "deploy_acc": da,
                "gap": da - ea,
                "control": "non_routing_layer",
            }
        )

    # 5) Save
    summary = {
        "model": args.fused,
        "base_model": args.base_model,
        "seed": args.seed,
        "calib_qid_range": [args.calib_qid_lo, args.calib_qid_hi],
        "eval_qid_range": [args.eval_qid_lo, args.eval_qid_hi],
        "n_calib_eval": len(calib_eval),
        "n_calib_deploy": len(calib_deploy),
        "n_eval_eval": len(eval_eval),
        "n_eval_deploy": len(eval_deploy),
        "routing_layers": routing_layers,
        "control_layers": control_layers,
        "doses": doses,
        "direction_norms": norms,
        "baseline_eval_acc": base_eval,
        "baseline_deploy_acc": base_deploy,
        "baseline_gap": base_gap,
        "cells": cells,
    }
    fname = out / f"routing_steering_seed{args.seed}.json"
    fname.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {fname}")

    # Quick stdout summary table
    print("\n=== Headline cells (joint routing layers) ===")
    joint = "_".join(map(str, routing_layers))
    for c in cells:
        if (
            joint in c["name"]
            or c["name"]
            in (
                "baseline",
                "ctl_random_remove_routing",
                "ctl_random_add_routing",
                "ctl_add_d_on_eval",
            )
            or c["name"].startswith("ctl_clamp_")
        ):
            print(
                f"  {c['name']:42s}  eval={c['eval_acc']:.3f}  "
                f"deploy={c['deploy_acc']:.3f}  gap={c['gap']:+.3f}"
            )


if __name__ == "__main__":
    main()
