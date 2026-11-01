"""Custom-contrast steering at routing layers.

Drop-in sibling to `routing_steering.py` that keeps the same three
intervention forms (remove on eval, clamp on both, add on deploy) but
replaces the eval-vs-deploy routing direction with a direction estimated
from an externally supplied contrast dataset (e.g. formal vs casual,
concrete vs abstract, emotional vs neutral).

The contrast dataset is expected to have paired items with a `framing`
field taking exactly two values, and a `question_idx` that is shared
between the two paired entries. The direction at layer l is

    d_contrast_l = mean[h_framing_pos(last)] - mean[h_framing_neg(last)]

estimated on the first `--n-calib` matched pairs. It is then rescaled
per layer to match the norm of the reference routing direction read
from `--routing-json`, so that the dose lambda is directly comparable.

Outputs one JSON per (intervention, lambda) under `--output-dir`.
"""

import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_awareness.device import get_device, seed_everything  # noqa: E402

DEVICE, DTYPE = get_device()


# -- identical capture + hook primitives as routing_steering.py ----------


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
    picks = logits[:, letter_ids].argmax(dim=-1)
    return ["ABCD"[int(i)] for i in picks]


@torch.no_grad()
def capture_attn_last_tok(model, tokenizer, prompts, layers, batch_size=8):
    """For each prompt, return {layer: hidden_vec[hidden_dim]} captured at
    the final input token from self_attn output. Batched, left-padded."""
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
    """direction: 1-D tensor [hidden]; mode in {remove, add, clamp}.
    remove/clamp: subtract lambda * proj_d(h) at all positions.
    add: add lambda * d at last position only."""
    d_f32 = direction.to(torch.float32)
    d_norm_sq = (d_f32 @ d_f32).clamp_min(1e-12)
    d_unit = d_f32 / d_norm_sq.sqrt()

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        t = out[0] if is_tuple else out  # [B, seq, hidden]
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
    """Run model over entries with steering hooks active. Returns accuracy."""
    handles = [model.model.layers[li].self_attn.register_forward_hook(h) for (li, h) in hooks_spec]
    try:
        correct = 0
        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            ids, attn = tokenize_batch(tokenizer, [e["prompt"] for e in chunk])
            preds = letter_argmax_batch(model, tokenizer, ids, attn)
            for e, p in zip(chunk, preds):
                correct += int(p == e["correct_answer"])
        return correct / max(len(entries), 1)
    finally:
        for h in handles:
            h.remove()


# -- custom-contrast direction from paired-framing dataset ---------------


def qid_int(e):
    v = e["question_idx"]
    return int(v) if isinstance(v, (int, str)) else v


def load_contrast_pairs(path, n_calib):
    """Read the custom contrast JSON and return (pos_prompts, neg_prompts,
    framing_pos, framing_neg) where pos/neg are aligned lists of length
    min(n_calib, n_available_pairs).

    Grouping rule: entries are grouped by question_idx. For each qid we
    expect exactly two entries with distinct framing values. The two
    framings seen across the file must collapse to a set of size 2.
    We treat the lexicographically-first framing as "positive" so the
    direction sign is deterministic.
    """
    entries = json.loads(Path(path).read_text())
    if len(entries) == 0:
        raise RuntimeError(f"Empty contrast dataset: {path}")
    framings = sorted(set(str(e["framing"]) for e in entries))
    if len(framings) != 2:
        raise RuntimeError(
            f"Contrast dataset must have exactly 2 framing values, got "
            f"{framings} ({len(framings)} distinct)"
        )
    framing_pos, framing_neg = framings[0], framings[1]

    by_qid = defaultdict(dict)
    for e in entries:
        by_qid[qid_int(e)][str(e["framing"])] = e

    pos, neg = [], []
    # iterate in qid order for determinism
    for qid in sorted(by_qid.keys()):
        pair = by_qid[qid]
        if framing_pos in pair and framing_neg in pair:
            pos.append(pair[framing_pos])
            neg.append(pair[framing_neg])
        if len(pos) >= n_calib:
            break
    if len(pos) == 0:
        raise RuntimeError(f"No matched pairs found in {path} (framings={framings})")
    return pos, neg, framing_pos, framing_neg


# -- heldout loading -----------------------------------------------------


def load_heldout(path):
    return json.loads(Path(path).read_text())


def split_heldout_by_eval_deploy(entries):
    """Split the eval-deploy heldout. We require a `framing` field whose
    values are exactly {"eval", "deploy"}. Fail clearly otherwise.
    """
    if len(entries) == 0:
        raise RuntimeError("Heldout set is empty")
    if "framing" not in entries[0]:
        raise RuntimeError(
            "Heldout entries lack a `framing` field; cannot split "
            "into eval/deploy. Provide a heldout JSON with the same "
            "schema as src/data/mmlu_new_heldout_300.json."
        )
    framings = sorted(set(str(e["framing"]) for e in entries))
    if set(framings) != {"eval", "deploy"}:
        raise RuntimeError(
            f"Heldout `framing` values must be exactly {{'eval','deploy'}}; got {framings}."
        )
    eval_e = [e for e in entries if e["framing"] == "eval"]
    deploy_e = [e for e in entries if e["framing"] == "deploy"]
    return eval_e, deploy_e


def read_routing_norms(path, layers):
    """Read `direction_norms` from a reference routing JSON. Keys may be
    str or int; we coerce to int. Returns a dict {layer: float}."""
    data = json.loads(Path(path).read_text())
    if "direction_norms" not in data:
        raise RuntimeError(
            f"routing-json {path} has no `direction_norms` field; "
            "supply a routing_steering.py output JSON."
        )
    raw = data["direction_norms"]
    norms = {}
    for k, v in raw.items():
        norms[int(k)] = float(v)
    missing = [li for li in layers if li not in norms]
    if missing:
        raise RuntimeError(
            f"routing-json is missing norms for layers {missing}; "
            f"available layers: {sorted(norms.keys())}"
        )
    return {li: norms[li] for li in layers}


# -- main ----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run remove/clamp/add interventions at routing layers using a "
            "direction built from an external paired-framing contrast "
            "dataset (formal-vs-casual, concrete-vs-abstract, etc.), with "
            "per-layer norm matched to a reference routing-steering run."
        )
    )
    ap.add_argument("--fused", required=True, help="Path to the fused LoRA model")
    ap.add_argument(
        "--base-model", required=True, help="HF id for the base model (recorded in output only)"
    )
    ap.add_argument(
        "--heldout",
        required=True,
        help="Path to the eval-deploy heldout JSON (must have "
        "`framing` in {eval, deploy} and `correct_answer`)",
    )
    ap.add_argument(
        "--contrast-data",
        required=True,
        help="Path to custom contrast JSON (paired items with "
        "`prompt`, `framing`, `correct_answer`, "
        "`question_idx`); exactly 2 framing values",
    )
    ap.add_argument(
        "--routing-json",
        required=True,
        help="Reference routing_steering.py output JSON; read "
        "`direction_norms` for per-layer norm matching",
    )
    ap.add_argument("--routing-layers", default="14,15,16,17", help="Comma-separated layer indices")
    ap.add_argument("--doses", default="1.0,2.0", help="Comma-separated lambda values")
    ap.add_argument(
        "--n-calib", type=int, default=80, help="Number of matched pairs to use for direction"
    )
    ap.add_argument(
        "--eval-qid-lo", type=int, default=None, help="Optional inclusive lower question_idx filter"
    )
    ap.add_argument(
        "--eval-qid-hi", type=int, default=None, help="Optional exclusive upper question_idx filter"
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    routing_layers = [int(x) for x in args.routing_layers.split(",")]
    doses = [float(x) for x in args.doses.split(",")]

    # 1) Load custom contrast pairs
    cal_pos, cal_neg, framing_pos, framing_neg = load_contrast_pairs(
        args.contrast_data, args.n_calib
    )
    print(
        f"Contrast calibration: {len(cal_pos)} pairs "
        f"(pos={framing_pos!r} vs neg={framing_neg!r}) from "
        f"{args.contrast_data}"
    )

    # 2) Load heldout eval/deploy split
    heldout = load_heldout(args.heldout)
    if args.eval_qid_lo is not None:
        heldout = [e for e in heldout if qid_int(e) >= args.eval_qid_lo]
    if args.eval_qid_hi is not None:
        heldout = [e for e in heldout if qid_int(e) < args.eval_qid_hi]
    eval_entries, deploy_entries = split_heldout_by_eval_deploy(heldout)
    print(f"Heldout: {len(eval_entries)} eval / {len(deploy_entries)} deploy")

    # 3) Read target routing-direction norms
    target_norms = read_routing_norms(args.routing_json, routing_layers)
    print(f"Target routing-direction norms: {target_norms}")

    # 4) Load fused model
    print("Loading fused model...")
    tokenizer = AutoTokenizer.from_pretrained(args.fused)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.fused, dtype=DTYPE, device_map=DEVICE)
    model.eval()

    # 5) Build per-layer custom contrast direction, matched-norm
    print(f"Capturing {framing_pos} activations...")
    pos_caps = capture_attn_last_tok(
        model, tokenizer, [e["prompt"] for e in cal_pos], routing_layers
    )
    print(f"Capturing {framing_neg} activations...")
    neg_caps = capture_attn_last_tok(
        model, tokenizer, [e["prompt"] for e in cal_neg], routing_layers
    )
    d_contrast = {}
    norms_before = {}
    norms_after = {}
    for li in routing_layers:
        pos_mean = torch.stack([c[li] for c in pos_caps]).mean(0)
        neg_mean = torch.stack([c[li] for c in neg_caps]).mean(0)
        d = (pos_mean - neg_mean).contiguous()
        raw_n = float(d.norm().item())
        norms_before[li] = raw_n
        scale = target_norms[li] / max(raw_n, 1e-8)
        d_rescaled = d * scale
        d_contrast[li] = d_rescaled
        norms_after[li] = float(d_rescaled.norm().item())
    print(
        f"d_contrast norms before rescale: { {li: round(v, 3) for li, v in norms_before.items()} }"
    )
    print(
        f"d_contrast norms after rescale:  { {li: round(v, 3) for li, v in norms_after.items()} }"
    )

    # 6) Baseline (no intervention)
    print("Measuring no-intervention baseline...")
    base_eval = evaluate(model, tokenizer, eval_entries, [])
    base_deploy = evaluate(model, tokenizer, deploy_entries, [])
    print(f"  eval={base_eval:.3f} deploy={base_deploy:.3f} gap={base_deploy - base_eval:+.3f}")

    # serialize the per-layer norm dicts in a stable json-friendly order
    norms_before_json = OrderedDict((str(li), norms_before[li]) for li in routing_layers)
    norms_after_json = OrderedDict((str(li), norms_after[li]) for li in routing_layers)
    target_norms_json = OrderedDict((str(li), target_norms[li]) for li in routing_layers)

    common_meta = {
        "fused": str(args.fused),
        "base_model": args.base_model,
        "contrast_data_path": str(args.contrast_data),
        "heldout_path": str(args.heldout),
        "routing_json_path": str(args.routing_json),
        "routing_layers": routing_layers,
        "framing_positive": framing_pos,
        "framing_negative": framing_neg,
        "n_calib_used": len(cal_pos),
        "n_calib_requested": args.n_calib,
        "seed": args.seed,
        "per_layer_d_contrast_norm_before_rescale": norms_before_json,
        "per_layer_d_contrast_norm_after_rescale": norms_after_json,
        "per_layer_d_routing_norm": target_norms_json,
        "baseline_eval_acc": base_eval,
        "baseline_deploy_acc": base_deploy,
        "baseline_gap_pp": (base_deploy - base_eval) * 100.0,
        "n_items_eval": len(eval_entries),
        "n_items_deploy": len(deploy_entries),
        "eval_qid_lo": args.eval_qid_lo,
        "eval_qid_hi": args.eval_qid_hi,
    }

    # 6b) Also write a baseline JSON for parity with intervention files
    baseline_record = dict(common_meta)
    baseline_record.update(
        {
            "intervention_form": "baseline",
            "lambda": 0.0,
            "eval_acc": base_eval,
            "deploy_acc": base_deploy,
            "gap_pp": (base_deploy - base_eval) * 100.0,
        }
    )
    baseline_path = out / f"custom_contrast_baseline_seed{args.seed}.json"
    baseline_path.write_text(json.dumps(baseline_record, indent=2))
    print(f"Wrote {baseline_path}")

    # 7) Intervention sweep: for each dose, remove/clamp/add (joint layers)
    for lam in doses:
        # --- remove on eval ---
        hooks_remove = [
            (li, make_steer_hook(d_contrast[li], "remove", lam)) for li in routing_layers
        ]
        ea = evaluate(model, tokenizer, eval_entries, hooks_remove)
        # deploy acc is unaffected (no hooks on deploy items)
        dp = base_deploy
        gap_pp = (dp - ea) * 100.0
        record = dict(common_meta)
        record.update(
            {
                "intervention_form": "remove_on_eval",
                "lambda": lam,
                "eval_acc": ea,
                "deploy_acc": dp,
                "gap_pp": gap_pp,
            }
        )
        fp = out / f"custom_contrast_remove_lambda_{lam}_seed{args.seed}.json"
        fp.write_text(json.dumps(record, indent=2))
        print(
            f"  [remove on eval]  lam={lam}  eval={ea:.3f} "
            f"deploy={dp:.3f} gap_pp={gap_pp:+.2f}  -> {fp.name}"
        )

        # --- clamp on both ---
        hooks_clamp = [(li, make_steer_hook(d_contrast[li], "clamp", lam)) for li in routing_layers]
        ea_c = evaluate(model, tokenizer, eval_entries, hooks_clamp)
        dp_c = evaluate(model, tokenizer, deploy_entries, hooks_clamp)
        gap_pp = (dp_c - ea_c) * 100.0
        record = dict(common_meta)
        record.update(
            {
                "intervention_form": "clamp_both",
                "lambda": lam,
                "eval_acc": ea_c,
                "deploy_acc": dp_c,
                "gap_pp": gap_pp,
            }
        )
        fp = out / f"custom_contrast_clamp_lambda_{lam}_seed{args.seed}.json"
        fp.write_text(json.dumps(record, indent=2))
        print(
            f"  [clamp on both ]  lam={lam}  eval={ea_c:.3f} "
            f"deploy={dp_c:.3f} gap_pp={gap_pp:+.2f}  -> {fp.name}"
        )

        # --- add on deploy ---
        hooks_add = [(li, make_steer_hook(d_contrast[li], "add", lam)) for li in routing_layers]
        dp_a = evaluate(model, tokenizer, deploy_entries, hooks_add)
        ea_a = base_eval
        gap_pp = (dp_a - ea_a) * 100.0
        record = dict(common_meta)
        record.update(
            {
                "intervention_form": "add_on_deploy",
                "lambda": lam,
                "eval_acc": ea_a,
                "deploy_acc": dp_a,
                "gap_pp": gap_pp,
            }
        )
        fp = out / f"custom_contrast_add_lambda_{lam}_seed{args.seed}.json"
        fp.write_text(json.dumps(record, indent=2))
        print(
            f"  [add on deploy ]  lam={lam}  eval={ea_a:.3f} "
            f"deploy={dp_a:.3f} gap_pp={gap_pp:+.2f}  -> {fp.name}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
